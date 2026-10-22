// include/strata/nemotron_h/layout.hpp - NVIDIA Nemotron 3.5 Super (GGUF architecture nemotron_h_moe): the geometry from
// the GGUF metadata, every tensor of the file mapped to its role, and the layout check (names, shapes, quant types,
// bounds, nothing left unmapped).  Header-only; the reference engine (src/nemotron_h/nhref.cu) and the layout tool
// (src/nemotron_h/nh_layout_main.cpp, strata-nh-layout) both use it.
//
// Source of truth: llama.cpp b11806 (0682a576a772de436caa5d1c1e94036486d6ebd0)
//   hparams   src/models/nemotron-h.cpp:5-45   (a layer is recurrent iff n_head_kv(il) == 0 and n_ff(il) == 0, :12-16;
//                                               the rms eps falls back to layer_norm_epsilon, :18-21)
//   tensors   src/models/nemotron-h.cpp:47-198 (global :67-76, attn_norm :83, Mamba-2 :86-101, attention :103-109,
//                                               MoE :110-140, the appended MTP/NextN block :160-197)
//   shapes    d_in_proj = 2*d_inner + 2*n_group*d_state + n_ssm_head (:63); conv channels d_inner + 2*n_group*d_state
//             (:89); ssm_norm {d_inner/n_group, n_group} (:98); experts up {latent, n_ff_exp, n_expert}, down
//             {n_ff_exp, latent, n_expert} (:129-130); router {n_embd, n_expert} (:123)
// The weights stay in the GGUF mapping (mmap, read-only); nothing is copied here.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::nemotron_h {

enum class Kind : int { Mamba = 0, Attention = 1, Moe = 2, Mtp = 3 };

inline const char* kind_name(Kind k) {
    switch (k) {
    case Kind::Mamba: return "mamba2";
    case Kind::Attention: return "attention";
    case Kind::Moe: return "moe";
    case Kind::Mtp: return "mtp";
    }
    return "?";
}

/// The model's shape, read from the metadata shard (keys "nemotron_h_moe.*").
struct Geometry {
    int n_block = 0;       ///< block_count: the main layers plus the MTP blocks (89)
    int n_layer = 0;       ///< main layers (88)
    int n_mtp = 0;         ///< nextn_predict_layers (1)
    int n_embd = 0, n_vocab = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0;   ///< attention (32, 2, 128); n_head_kv of the attention layers
    int ssm_d_conv = 0, ssm_d_inner = 0, ssm_d_state = 0, ssm_n_group = 0, ssm_n_head = 0, ssm_head_dim = 0;
    int ssm_d_in_proj = 0, ssm_conv_dim = 0;
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0, n_expert_shared = 0, moe_latent = 0;
    int expert_group_count = 0;
    bool expert_weights_norm = false;
    float expert_weights_scale = 0.f;
    float norm_eps = 0.f, rms_eps = 0.f;
    std::vector<Kind> kind;  ///< per block (n_block)
    std::vector<int> attention_layers;
};

inline const MetaValue& need(const GgufFile& g, const std::string& key) {
    const MetaValue* v = g.get(key);
    if (!v) throw std::runtime_error("nemotron_h: GGUF key '" + key + "' missing");
    return *v;
}

/// Element i of a per-layer key that is an array (or a scalar for every layer).
inline uint64_t per_layer_u(const MetaValue& v, int i, const std::string& key) {
    if (v.type != MetaType::ARRAY) return v.u;
    if ((uint64_t) i >= v.count || (size_t) i >= v.items.size())
        throw std::runtime_error("nemotron_h: per-layer key '" + key + "' too short");
    return v.items[(size_t) i].u;
}

inline Geometry geometry_from_gguf(const GgufFile& g) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch || arch->s != "nemotron_h_moe")
        throw std::runtime_error("nemotron_h: general.architecture is '" + (arch ? arch->s : std::string()) +
                                 "', expected 'nemotron_h_moe'");
    const std::string p = "nemotron_h_moe.";
    auto U = [&](const char* k) { return (int) need(g, p + k).u; };
    Geometry m;
    m.n_block = U("block_count");
    m.n_mtp = g.get(p + "nextn_predict_layers") ? U("nextn_predict_layers") : 0;
    m.n_layer = m.n_block - m.n_mtp;
    m.n_embd = U("embedding_length");
    m.n_vocab = U("vocab_size");
    m.n_head = U("attention.head_count");
    m.head_dim = U("attention.key_length");
    if (U("attention.value_length") != m.head_dim) throw std::runtime_error("nemotron_h: key_length != value_length");
    m.ssm_d_conv = U("ssm.conv_kernel");
    m.ssm_d_inner = U("ssm.inner_size");
    m.ssm_d_state = U("ssm.state_size");
    m.ssm_n_group = U("ssm.group_count");
    m.ssm_n_head = U("ssm.time_step_rank");
    m.ssm_head_dim = m.ssm_d_inner / m.ssm_n_head;
    m.ssm_d_in_proj = 2 * m.ssm_d_inner + 2 * m.ssm_n_group * m.ssm_d_state + m.ssm_n_head;   // nemotron-h.cpp:63
    m.ssm_conv_dim = m.ssm_d_inner + 2 * m.ssm_n_group * m.ssm_d_state;                       // nemotron-h.cpp:89
    m.n_expert = U("expert_count");
    m.n_expert_used = U("expert_used_count");
    m.n_ff_exp = U("expert_feed_forward_length");
    m.n_ff_shexp = U("expert_shared_feed_forward_length");
    m.n_expert_shared = U("expert_shared_count");
    m.moe_latent = U("moe_latent_size");
    m.expert_group_count = U("expert_group_count");
    m.expert_weights_norm = need(g, p + "expert_weights_norm").u != 0;
    m.expert_weights_scale = (float) need(g, p + "expert_weights_scale").f;
    m.norm_eps = (float) need(g, p + "attention.layer_norm_epsilon").f;
    const MetaValue* rms = g.get(p + "attention.layer_norm_rms_epsilon");
    m.rms_eps = rms ? (float) rms->f : m.norm_eps;   // nemotron-h.cpp:18-21
    const std::string k_kv = p + "attention.head_count_kv", k_ff = p + "feed_forward_length";
    const MetaValue& kv = need(g, k_kv);   // references into the metadata map
    const MetaValue& ff = need(g, k_ff);
    m.kind.resize((size_t) m.n_block);
    for (int i = 0; i < m.n_block; ++i) {
        const uint64_t hkv = per_layer_u(kv, i, "head_count_kv"), nff = per_layer_u(ff, i, "feed_forward_length");
        if (i >= m.n_layer) m.kind[(size_t) i] = Kind::Mtp;
        else if (hkv == 0 && nff == 0) m.kind[(size_t) i] = Kind::Mamba;   // nemotron-h.cpp:14-16
        else if (nff == 0) m.kind[(size_t) i] = Kind::Attention;
        else m.kind[(size_t) i] = Kind::Moe;
        if (m.kind[(size_t) i] == Kind::Attention) {
            if (m.n_head_kv && (int) hkv != m.n_head_kv) throw std::runtime_error("nemotron_h: n_head_kv varies");
            m.n_head_kv = (int) hkv;
            m.attention_layers.push_back(i);
        }
        if (m.kind[(size_t) i] == Kind::Moe && (int) nff != m.n_ff_exp)
            throw std::runtime_error("nemotron_h: per-layer feed_forward_length != expert_feed_forward_length");
    }
    if (m.expert_group_count > 1) throw std::runtime_error("nemotron_h: grouped expert routing is not implemented");
    return m;
}

/// One tensor as the engine sees it: its GGUF entry, the shard holding it and its bytes in the mapping.
struct Tensor {
    const TensorInfo* info = nullptr;
    const uint8_t* data = nullptr;
    size_t shard = 0;
    uint32_t type = 0;
    uint64_t bytes = 0;
    uint64_t ne(int d) const { return info && (size_t) d < info->shape.size() ? info->shape[(size_t) d] : 1; }
    explicit operator bool() const { return info != nullptr; }
};

struct Layer {
    Kind kind = Kind::Mamba;
    Tensor attn_norm;
    // Mamba-2
    Tensor ssm_in, ssm_conv1d, ssm_conv1d_b, ssm_dt_b, ssm_a, ssm_d, ssm_norm, ssm_out;
    // attention (also the MTP block's attention sub-layer)
    Tensor wq, wk, wv, wo;
    // MoE (also the MTP block's MoE sub-layer)
    Tensor gate_inp, exp_probs_b, latent_down, latent_up, up_exps, down_exps, up_shexp, down_shexp;
    // MTP only
    Tensor post_attn_norm, eh_proj, enorm, hnorm, shared_head_norm;
};

/// The quant types the reference engine dequantises (ggml type ids).
inline bool supported_matrix_type(uint32_t t) {
    switch (t) {
    case 0: case 1: case 30:            // F32, F16, BF16
    case 6: case 8:                      // Q5_0, Q8_0
    case 12: case 13: case 14:           // Q4_K, Q5_K, Q6_K
    case 20:                             // IQ4_NL
        return true;
    default:
        return false;
    }
}

struct LayoutReport {
    int expected = 0, found = 0, ok = 0, in_file = 0;
    std::vector<std::string> errors;
    std::vector<std::string> rows;   ///< one line per tensor: name kind role shape type bytes shard
    std::map<std::string, std::map<std::string, int>> types_by_role;
    uint64_t bytes = 0, bytes_mtp = 0, bytes_experts = 0;
    bool pass() const { return errors.empty() && expected == found && found == ok && ok == in_file; }
};

/// The model: geometry, the global tensors and every block's tensors, all pointing into the GGUF mapping.
struct Model {
    Geometry g;
    Tensor tok_embd, output_norm, output;
    std::vector<Layer> layers;   ///< n_block entries; blk.88 (MTP) is loaded but the main forward does not use it
    LayoutReport report;
};

namespace detail {

inline std::string shape_str(const std::vector<uint64_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? ", " : "") + std::to_string(s[i]);
    return r + "]";
}

struct Binder {
    const GgufModel& gm;
    LayoutReport& rep;
    std::set<std::string>& seen;
    Tensor bind(const std::string& name, const std::vector<uint64_t>& shape, bool matrix, const std::string& role,
                const char* kind, bool mtp) {
        ++rep.expected;
        size_t s = 0;
        const TensorInfo* t = gm.find(name, &s);
        Tensor r;
        if (!t) {
            rep.errors.push_back("missing tensor " + name);
            return r;
        }
        ++rep.found;
        seen.insert(name);
        r.info = t;
        r.shard = s;
        r.type = t->type;
        r.bytes = tensor_payload_bytes(*t);
        r.data = gm.shard(s).tensor_data(*t);
        bool good = true;
        if (t->shape != shape) {
            rep.errors.push_back(name + ": shape " + shape_str(t->shape) + ", expected " + shape_str(shape));
            good = false;
        }
        if (matrix ? !supported_matrix_type(t->type) : t->type != 0) {
            rep.errors.push_back(name + ": type " + t->type_name() + " not supported for this role");
            good = false;
        }
        if (!gm.in_bounds(*t, s)) {
            rep.errors.push_back(name + ": payload out of the file's bounds");
            good = false;
        }
        if (good) ++rep.ok;
        rep.bytes += r.bytes;
        if (mtp) rep.bytes_mtp += r.bytes;
        else if (role == "ffn_up_exps" || role == "ffn_down_exps") rep.bytes_experts += r.bytes;
        rep.types_by_role[std::string(mtp ? "mtp:" : "") + role][t->type_name()]++;
        char line[512];
        std::snprintf(line, sizeof line, "%-36s %-9s %-18s %-22s %-7s %12llu shard%zu", name.c_str(), kind,
                      role.c_str(), shape_str(t->shape).c_str(), t->type_name(), (unsigned long long) r.bytes, s + 1);
        rep.rows.emplace_back(line);
        return r;
    }
};

}  // namespace detail

/// Map every tensor of the model and check it.  Throws only when the metadata is unusable; layout problems land in
/// `report.errors` (the caller decides: the reference engine refuses to run unless report.pass()).
inline Model load_model(const GgufModel& gm) {
    Model m;
    m.g = geometry_from_gguf(gm.meta());
    const Geometry& g = m.g;
    std::set<std::string> seen;
    detail::Binder b{gm, m.report, seen};
    using U = uint64_t;
    const U E = (U) g.n_embd, V = (U) g.n_vocab;
    // nemotron-h.cpp:67-76 (output is required here: this GGUF has its own, untied)
    m.tok_embd = b.bind("token_embd.weight", {E, V}, true, "token_embd", "global", false);
    m.output_norm = b.bind("output_norm.weight", {E}, false, "output_norm", "global", false);
    m.output = b.bind("output.weight", {E, V}, true, "output", "global", false);
    m.layers.resize((size_t) g.n_block);
    for (int i = 0; i < g.n_block; ++i) {
        Layer& L = m.layers[(size_t) i];
        L.kind = g.kind[(size_t) i];
        const std::string pre = "blk." + std::to_string(i) + ".";
        const char* kn = kind_name(L.kind);
        const bool mtp = L.kind == Kind::Mtp;
        auto T = [&](const char* suffix, std::vector<U> shape, bool matrix) {
            std::string role = suffix;
            const size_t dot = role.rfind(".weight");
            if (dot != std::string::npos && dot + 7 == role.size()) role.resize(dot);
            return b.bind(pre + suffix, shape, matrix, role, kn, mtp);
        };
        L.attn_norm = T("attn_norm.weight", {E}, false);   // nemotron-h.cpp:83 (all blocks), :171 (MTP)
        const U H = (U) g.n_head, HKV = (U) g.n_head_kv, D = (U) g.head_dim;
        if (L.kind == Kind::Mamba) {   // nemotron-h.cpp:86-101
            const U NH = (U) g.ssm_n_head;
            L.ssm_in = T("ssm_in.weight", {E, (U) g.ssm_d_in_proj}, true);
            L.ssm_conv1d = T("ssm_conv1d.weight", {(U) g.ssm_d_conv, (U) g.ssm_conv_dim}, false);
            L.ssm_conv1d_b = T("ssm_conv1d.bias", {(U) g.ssm_conv_dim}, false);
            L.ssm_dt_b = T("ssm_dt.bias", {NH}, false);
            L.ssm_a = T("ssm_a", {1, NH}, false);
            L.ssm_d = T("ssm_d", {1, NH}, false);
            L.ssm_norm = T("ssm_norm.weight", {(U) (g.ssm_d_inner / g.ssm_n_group), (U) g.ssm_n_group}, false);
            L.ssm_out = T("ssm_out.weight", {(U) g.ssm_d_inner, E}, true);
        }
        if (L.kind == Kind::Attention || mtp) {   // nemotron-h.cpp:103-109, :172-174 (create_tensor_qkv: separate q/k/v here)
            L.wq = T("attn_q.weight", {E, H * D}, true);
            L.wk = T("attn_k.weight", {E, HKV * D}, true);
            L.wv = T("attn_v.weight", {E, HKV * D}, true);
            L.wo = T("attn_output.weight", {H * D, E}, true);
        }
        if (L.kind == Kind::Moe || mtp) {   // nemotron-h.cpp:110-140, :177-186
            const U NE = (U) g.n_expert, LAT = (U) g.moe_latent, FF = (U) g.n_ff_exp, FS = (U) g.n_ff_shexp;
            L.gate_inp = T("ffn_gate_inp.weight", {E, NE}, true);
            L.exp_probs_b = T("exp_probs_b.bias", {NE}, false);
            L.latent_down = T("ffn_latent_down.weight", {E, LAT}, true);
            L.latent_up = T("ffn_latent_up.weight", {LAT, E}, true);
            L.up_exps = T("ffn_up_exps.weight", {LAT, FF, NE}, true);
            L.down_exps = T("ffn_down_exps.weight", {FF, LAT, NE}, true);
            L.up_shexp = T("ffn_up_shexp.weight", {E, FS}, true);
            L.down_shexp = T("ffn_down_shexp.weight", {FS, E}, true);
        }
        if (mtp) {   // nemotron-h.cpp:165-168, :177
            L.enorm = T("nextn.enorm.weight", {E}, false);
            L.hnorm = T("nextn.hnorm.weight", {E}, false);
            L.eh_proj = T("nextn.eh_proj.weight", {2 * E, E}, true);
            L.shared_head_norm = T("nextn.shared_head_norm.weight", {E}, false);
            L.post_attn_norm = T("post_attention_norm.weight", {E}, false);
        }
    }
    // nothing in the file may be left unmapped
    LayoutReport& rep = m.report;
    for (size_t s = 0; s < gm.size(); ++s)
        for (const auto& t : gm.shard(s).tensors()) {
            ++rep.in_file;
            if (!seen.count(t.name)) rep.errors.push_back("unmapped tensor in the file: " + t.name);
        }
    if (const MetaValue* tc = gm.meta().get("split.tensors.count"))
        if ((int) tc->u != rep.in_file)
            rep.errors.push_back("split.tensors.count " + std::to_string(tc->u) + " != tensors in the shards " +
                                 std::to_string(rep.in_file));
    return m;
}

}  // namespace strata::nemotron_h
