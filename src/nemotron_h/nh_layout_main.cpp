// src/nemotron_h/nh_layout_main.cpp - strata-nh-layout: open a Nemotron 3.5 Super GGUF (all shards, mmap), map every
// tensor (include/strata/nemotron_h/layout.hpp) and check names, shapes, quant types, bounds and completeness; open
// the vision projector (mmproj) too and check its tensors, loaded but unused.  Exit 0 only when everything passes.
//
// Usage: strata-nh-layout --model <any shard> [--mmproj <mmproj.gguf>] [--table <file>]
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/nemotron_h/layout.hpp"
#ifndef NH_LAYOUT_BIN
#error "NH_LAYOUT_BIN comes from nemotron/project.env through CMakeLists.txt"
#endif

#include <cstdio>
#include <cstring>
#include <string>

using namespace strata;
using namespace strata::nemotron_h;

static int check_mmproj(const std::string& path, FILE* tab) {
    const GgufFile f(path);
    const MetaValue* arch = f.get("general.architecture");
    const std::string a = arch ? arch->s : "";
    int bad = 0, n = 0;
    uint64_t bytes = 0;
    std::map<std::string, int> types;
    const uint64_t payload = f.file_size() - f.data_start();
    for (const auto& t : f.tensors()) {
        ++n;
        const uint64_t b = tensor_payload_bytes(t);
        types[t.type_name()]++;
        bytes += b;
        if (b == 0 || t.offset > payload || b > payload - t.offset) {
            std::printf("mmproj: %s out of bounds or unknown type %s\n", t.name.c_str(), t.type_name());
            ++bad;
        }
        if (tab) std::fprintf(tab, "mmproj %-48s %-7s %12llu\n", t.name.c_str(), t.type_name(), (unsigned long long) b);
    }
    std::printf("mmproj: %s  architecture=%s  tensors=%d  bytes=%llu  types:", path.c_str(), a.c_str(), n,
                (unsigned long long) bytes);
    for (const auto& kv : types) std::printf(" %s=%d", kv.first.c_str(), kv.second);
    std::printf("\nmmproj: %s (loaded, unused by the text path)\n", (bad == 0 && a == "clip") ? "PASS" : "FAIL");
    return (bad == 0 && a == "clip") ? 0 : 1;
}

int main(int argc, char** argv) {
    std::string model, mmproj, table;
    for (int i = 1; i + 1 < argc; ++i) {
        if (!std::strcmp(argv[i], "--model")) model = argv[++i];
        else if (!std::strcmp(argv[i], "--mmproj")) mmproj = argv[++i];
        else if (!std::strcmp(argv[i], "--table")) table = argv[++i];
    }
    if (model.empty()) {
        std::fprintf(stderr, "usage: " NH_LAYOUT_BIN " --model <shard> [--mmproj f] [--table f]\n");
        return 2;
    }
    try {
        const GgufModel gm = GgufModel::open(model);
        const Model m = load_model(gm);
        const Geometry& g = m.g;
        std::printf("model: %zu shard(s)\n", gm.size());
        for (size_t s = 0; s < gm.size(); ++s)
            std::printf("  shard%zu %s tensors=%zu size=%llu\n", s + 1, gm.shard(s).path().c_str(),
                        gm.shard(s).tensors().size(), (unsigned long long) gm.shard(s).file_size());
        int nk[4] = {0, 0, 0, 0};
        for (Kind k : g.kind) nk[(int) k]++;
        std::printf("geometry: blocks=%d (main %d + mtp %d) mamba2=%d attention=%d moe=%d n_embd=%d vocab=%d\n", g.n_block,
                    g.n_layer, g.n_mtp, nk[0], nk[1], nk[2], g.n_embd, g.n_vocab);
        std::printf("  attention layers:");
        for (int l : g.attention_layers) std::printf(" %d", l);
        std::printf("  heads %d/%d x %d\n", g.n_head, g.n_head_kv, g.head_dim);
        std::printf("  mamba2: d_conv %d d_inner %d d_state %d groups %d heads %d x %d in_proj %d conv_dim %d\n",
                    g.ssm_d_conv, g.ssm_d_inner, g.ssm_d_state, g.ssm_n_group, g.ssm_n_head, g.ssm_head_dim,
                    g.ssm_d_in_proj, g.ssm_conv_dim);
        std::printf("  moe: experts %d top-%d n_ff %d latent %d shared %d x %d norm_w %d scale %.3f groups %d\n",
                    g.n_expert, g.n_expert_used, g.n_ff_exp, g.moe_latent, g.n_expert_shared, g.n_ff_shexp,
                    (int) g.expert_weights_norm, g.expert_weights_scale, g.expert_group_count);
        std::printf("  eps: norm %.3g rms %.3g\n", g.norm_eps, g.rms_eps);
        const LayoutReport& r = m.report;
        std::printf("layout: expected %d, found %d, ok %d, tensors in file %d\n", r.expected, r.found, r.ok, r.in_file);
        std::printf("  bytes: all %llu, main routed experts %llu, mtp block %llu\n", (unsigned long long) r.bytes,
                    (unsigned long long) r.bytes_experts, (unsigned long long) r.bytes_mtp);
        for (const auto& kv : r.types_by_role) {
            std::printf("  %-28s", kv.first.c_str());
            for (const auto& t : kv.second) std::printf(" %s=%d", t.first.c_str(), t.second);
            std::printf("\n");
        }
        for (const auto& e : r.errors) std::printf("ERROR %s\n", e.c_str());
        std::printf("layout: %s\n", r.pass() ? "PASS" : "FAIL");
        FILE* tab = table.empty() ? nullptr : std::fopen(table.c_str(), "w");
        if (tab)
            for (const auto& row : r.rows) std::fprintf(tab, "%s\n", row.c_str());
        int rc = r.pass() ? 0 : 1;
        if (!mmproj.empty()) rc |= check_mmproj(mmproj, tab);
        if (tab) std::fclose(tab);
        return rc;
    } catch (const std::exception& e) {
        std::printf("ERROR %s\nlayout: FAIL\n", e.what());
        return 1;
    }
}
