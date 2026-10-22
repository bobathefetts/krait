// tests/core/nemotron_h_layout_test.cpp - Nemotron 3.5 Super (nemotron_h_moe) on synthetic GGUFs: no model, no GPU.
//
//   * geometry_from_gguf: the block kinds from the per-layer keys (Mamba-2 / attention / MoE / MTP), the derived
//     Mamba-2 sizes (d_in_proj, conv channels), the attention layer list; refusals for another architecture, zero or
//     inconsistent geometry, a value that does not fit an int and grouped expert routing;
//   * load_model's layout check on a tiny complete file (passes), and with a wrong shape, a missing tensor, an
//     unsupported quant type and a tensor the layout does not name (each refused, naming the tensor);
//   * the GGUF reader keeps short per-layer arrays whole (the 89-block keys) and samples long ones;
//   * the model family names nemotron_h_moe.
//
// `nemotron_h_layout_test --write DIR` writes tiny-ok.gguf, tiny-missing.gguf and tiny-arch.gguf for the CLI tests
// (strata-nh-layout and strata-nh-engine exit codes, CMakeLists.txt).
#include "gguf_fixture.hpp"

#include "strata/artifact/gguf_split.hpp"
#include "strata/artifact/model_family.hpp"
#include "strata/nemotron_h/layout.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace nh = strata::nemotron_h;
using fixture::Kv;
using fixture::Tensor;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-nh-layout-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

// a tiny model of the family: block 0 Mamba-2, 1 attention, 2 MoE, 3 the MTP block (every tensor F32)
struct Tiny {
    uint64_t E = 64, V = 32, H = 2, HKV = 1, D = 16, conv = 4, inner = 32, state = 8, groups = 2, ssm_heads = 4;
    uint64_t NE = 4, used = 2, FF = 32, FS = 32, LAT = 32, group_count = 1;
    std::string arch = "nemotron_h_moe";
    uint64_t block_count = 4;
};

std::vector<Kv> metadata(const Tiny& t) {
    const std::string p = "nemotron_h_moe.";
    return {fixture::str("general.architecture", t.arch),
            fixture::u32(p + "block_count", t.block_count),
            fixture::u32(p + "nextn_predict_layers", 1),
            fixture::u32(p + "embedding_length", t.E),
            fixture::u32(p + "vocab_size", t.V),
            fixture::u32(p + "attention.head_count", t.H),
            fixture::u32s(p + "attention.head_count_kv", {0, t.HKV, 0, 0}),
            fixture::u32(p + "attention.key_length", t.D),
            fixture::u32(p + "attention.value_length", t.D),
            fixture::u32s(p + "feed_forward_length", {0, 0, t.FF, t.FF}),
            fixture::u32(p + "ssm.conv_kernel", t.conv),
            fixture::u32(p + "ssm.inner_size", t.inner),
            fixture::u32(p + "ssm.state_size", t.state),
            fixture::u32(p + "ssm.group_count", t.groups),
            fixture::u32(p + "ssm.time_step_rank", t.ssm_heads),
            fixture::u32(p + "expert_count", t.NE),
            fixture::u32(p + "expert_used_count", t.used),
            fixture::u32(p + "expert_feed_forward_length", t.FF),
            fixture::u32(p + "expert_shared_feed_forward_length", t.FS),
            fixture::u32(p + "expert_shared_count", 1),
            fixture::u32(p + "moe_latent_size", t.LAT),
            fixture::u32(p + "expert_group_count", t.group_count),
            fixture::u32(p + "expert_weights_norm", 1),
            fixture::f32(p + "expert_weights_scale", 2.5f),
            fixture::f32(p + "attention.layer_norm_epsilon", 1e-5f)};
}

std::vector<Tensor> tensors(const Tiny& t) {
    const uint64_t E = t.E, in_proj = 2 * t.inner + 2 * t.groups * t.state + t.ssm_heads,
                   conv_dim = t.inner + 2 * t.groups * t.state;
    std::vector<Tensor> ts;
    uint8_t seed = 1;
    auto add = [&](const std::string& n, std::vector<uint64_t> s) {
        Tensor x;
        x.name = n;
        x.shape = std::move(s);
        x.seed = seed++;
        ts.push_back(x);
    };
    add("token_embd.weight", {E, t.V});
    add("output_norm.weight", {E});
    add("output.weight", {E, t.V});
    for (int i = 0; i < 4; ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        add(b + "attn_norm.weight", {E});
        if (i == 0) {
            add(b + "ssm_in.weight", {E, in_proj});
            add(b + "ssm_conv1d.weight", {t.conv, conv_dim});
            add(b + "ssm_conv1d.bias", {conv_dim});
            add(b + "ssm_dt.bias", {t.ssm_heads});
            add(b + "ssm_a", {1, t.ssm_heads});
            add(b + "ssm_d", {1, t.ssm_heads});
            add(b + "ssm_norm.weight", {t.inner / t.groups, t.groups});
            add(b + "ssm_out.weight", {t.inner, E});
        }
        if (i == 1 || i == 3) {
            add(b + "attn_q.weight", {E, t.H * t.D});
            add(b + "attn_k.weight", {E, t.HKV * t.D});
            add(b + "attn_v.weight", {E, t.HKV * t.D});
            add(b + "attn_output.weight", {t.H * t.D, E});
        }
        if (i == 2 || i == 3) {
            add(b + "ffn_gate_inp.weight", {E, t.NE});
            add(b + "exp_probs_b.bias", {t.NE});
            add(b + "ffn_latent_down.weight", {E, t.LAT});
            add(b + "ffn_latent_up.weight", {t.LAT, E});
            add(b + "ffn_up_exps.weight", {t.LAT, t.FF, t.NE});
            add(b + "ffn_down_exps.weight", {t.FF, t.LAT, t.NE});
            add(b + "ffn_up_shexp.weight", {E, t.FS});
            add(b + "ffn_down_shexp.weight", {t.FS, E});
        }
        if (i == 3) {
            add(b + "nextn.enorm.weight", {E});
            add(b + "nextn.hnorm.weight", {E});
            add(b + "nextn.eh_proj.weight", {2 * E, E});
            add(b + "nextn.shared_head_norm.weight", {E});
            add(b + "post_attention_norm.weight", {E});
        }
    }
    return ts;
}

Tensor* find(std::vector<Tensor>& ts, const std::string& n) {
    for (auto& x : ts)
        if (x.name == n) return &x;
    return nullptr;
}

bool any_error_names(const nh::Model& m, const std::string& needle) {
    for (const auto& e : m.report.errors)
        if (e.find(needle) != std::string::npos) return true;
    return false;
}

bool throws_with(const std::function<void()>& f, const std::string& needle) {
    try {
        f();
    } catch (const std::exception& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

nh::Model load(const fs::path& p) {
    const strata::GgufModel gm = strata::GgufModel::open(p.string());
    return nh::load_model(gm);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && !std::strcmp(argv[1], "--write")) {   // fixtures for the CLI tests
        const fs::path d = argv[2];
        fs::create_directories(d);
        const Tiny t;
        fixture::write(d / "tiny-ok.gguf", metadata(t), tensors(t));
        auto ts = tensors(t);
        for (size_t i = 0; i < ts.size(); ++i)
            if (ts[i].name == "blk.0.ssm_in.weight") ts.erase(ts.begin() + (long) i);
        fixture::write(d / "tiny-missing.gguf", metadata(t), ts);
        Tiny q = t;
        q.arch = "qwen35moe";
        fixture::write(d / "tiny-arch.gguf", metadata(q), tensors(t));
        std::printf("wrote %s/tiny-{ok,missing,arch}.gguf\n", d.string().c_str());
        return 0;
    }
    TempDir tmp;
    const Tiny t;

    std::printf("geometry\n");
    {
        const fs::path p = tmp.path / "ok.gguf";
        fixture::write(p, metadata(t), tensors(t));
        const strata::GgufModel gm = strata::GgufModel::open(p.string());
        const nh::Geometry g = nh::geometry_from_gguf(gm.meta());
        check(g.n_block == 4 && g.n_layer == 3 && g.n_mtp == 1, "blocks 4 = 3 main + 1 MTP");
        check(g.kind.size() == 4 && g.kind[0] == nh::Kind::Mamba && g.kind[1] == nh::Kind::Attention &&
                  g.kind[2] == nh::Kind::Moe && g.kind[3] == nh::Kind::Mtp,
              "kinds from head_count_kv / feed_forward_length: mamba2, attention, moe, mtp");
        check(g.attention_layers == std::vector<int>{1} && g.n_head_kv == 1, "attention layers [1], n_head_kv 1");
        check(g.ssm_d_in_proj == 2 * 32 + 2 * 2 * 8 + 4 && g.ssm_conv_dim == 32 + 2 * 2 * 8 && g.ssm_head_dim == 8,
              "Mamba-2 d_in_proj 100, conv channels 64, head dim 8");
        check(g.rms_eps == g.norm_eps && g.expert_weights_norm && g.expert_weights_scale == 2.5f,
              "rms eps falls back to layer_norm_epsilon; expert weights norm + scale");
    }
    auto refuses = [&](const char* what, const std::function<void(Tiny&)>& edit, const std::string& needle) {
        Tiny b = t;
        edit(b);
        const fs::path p = tmp.path / "bad.gguf";
        fixture::write(p, metadata(b), tensors(t));
        check(throws_with([&] { (void) load(p); }, needle), std::string("refused: ") + what);
    };
    refuses("another architecture", [](Tiny& b) { b.arch = "qwen35moe"; }, "expected 'nemotron_h_moe'");
    refuses("inner size not a multiple of the SSM heads", [](Tiny& b) { b.ssm_heads = 5; }, "inconsistent model geometry");
    refuses("zero embedding length", [](Tiny& b) { b.E = 0; }, "inconsistent model geometry");
    refuses("a value that does not fit an int", [](Tiny& b) { b.block_count = (1ull << 31); }, "out of range");
    refuses("grouped expert routing", [](Tiny& b) { b.group_count = 2; }, "grouped expert routing");
    refuses("more experts used than exist", [](Tiny& b) { b.used = 9; }, "expert geometry");

    std::printf("layout\n");
    {
        const fs::path p = tmp.path / "ok.gguf";
        fixture::write(p, metadata(t), tensors(t));
        const nh::Model m = load(p);
        check(m.report.pass() && m.report.expected == (int) tensors(t).size() && m.report.in_file == m.report.expected,
              "a complete tiny file passes (" + std::to_string(m.report.expected) + " tensors mapped)");
        check(m.layers.size() == 4 && m.layers[0].ssm_in && m.layers[1].wq && m.layers[2].up_exps &&
                  m.layers[3].eh_proj && m.layers[3].down_exps && m.output,
              "roles bound: ssm_in, attn_q, ffn_up_exps, nextn.eh_proj, MTP experts, output");
    }
    auto bad_layout = [&](const char* what, const std::function<void(std::vector<Tensor>&)>& edit,
                          const std::string& needle) {
        auto ts = tensors(t);
        edit(ts);
        const fs::path p = tmp.path / "bad-layout.gguf";
        fixture::write(p, metadata(t), ts);
        const nh::Model m = load(p);
        check(!m.report.pass() && any_error_names(m, needle), std::string("refused: ") + what + " (" + needle + ")");
    };
    bad_layout("a wrong shape", [](std::vector<Tensor>& ts) { find(ts, "blk.2.ffn_up_exps.weight")->shape = {32, 32, 3}; },
               "blk.2.ffn_up_exps.weight: shape");
    bad_layout("a missing tensor", [](std::vector<Tensor>& ts) {
        for (size_t i = 0; i < ts.size(); ++i)
            if (ts[i].name == "blk.1.attn_k.weight") ts.erase(ts.begin() + (long) i);
    }, "missing tensor blk.1.attn_k.weight");
    bad_layout("a quant type the engine does not read (Q4_0 router)",
               [](std::vector<Tensor>& ts) { find(ts, "blk.2.ffn_gate_inp.weight")->type = 2; },
               "blk.2.ffn_gate_inp.weight: type");
    bad_layout("a norm that is not F32", [](std::vector<Tensor>& ts) { find(ts, "output_norm.weight")->type = 1; },
               "output_norm.weight: type");
    bad_layout("a tensor the layout does not name", [](std::vector<Tensor>& ts) {
        Tensor x;
        x.name = "blk.0.extra.weight";
        x.shape = {64};
        ts.push_back(x);
    }, "unmapped tensor in the file: blk.0.extra.weight");

    std::printf("gguf reader\n");
    {
        std::vector<uint64_t> shortv(100), longv(5000);
        for (size_t i = 0; i < shortv.size(); ++i) shortv[i] = i * 3;
        for (size_t i = 0; i < longv.size(); ++i) longv[i] = i;
        const fs::path p = tmp.path / "arrays.gguf";
        fixture::write(p, {fixture::str("general.architecture", "x"), fixture::u32s("x.short", shortv),
                           fixture::u32s("x.long", longv)}, {});
        const strata::GgufFile f(p.string());
        const strata::MetaValue* s = f.get("x.short");
        const strata::MetaValue* l = f.get("x.long");
        check(s && s->count == 100 && s->items.size() == 100 && s->items[99].u == 297, "a 100-entry array is kept whole");
        check(l && l->count == 5000 && l->items.size() == 64, "a 5000-entry array keeps its count and a 64-entry sample");
    }

    std::printf("model family\n");
    {
        using strata::ModelFamily;
        const ModelFamily f = strata::model_family_from_arch("nemotron_h_moe");
        check(f == ModelFamily::NemotronHMoe && std::string(strata::model_family_arch(f)) == "nemotron_h_moe" &&
                  std::string(strata::model_family_title(f)) == "Nemotron 3.5 Super",
              "nemotron_h_moe -> NemotronHMoe, title Nemotron 3.5 Super");
        check(strata::model_family_not_supported(f).find("krait/krait") != std::string::npos,
              "the strata engine's refusal points to Krait (krait/krait)");
    }

    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
