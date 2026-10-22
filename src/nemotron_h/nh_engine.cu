// src/nemotron_h/nh_engine.cu - strata-nh-engine: the Phase 2 engine for NVIDIA Nemotron 3.5 Super (GGUF architecture
// nemotron_h_moe).  Gate 2 asks for llama.cpp's numerics (token-identical, or top-1 > 99.5% and ppl within 0.5% of
// capture A), so this engine runs llama.cpp's own ggml kernels (ggml-cuda + ggml-cpu from a separate build of the
// reference sources, src/llama.cpp @ 0682a576a, same flags as the reference build) on a graph that Strata builds and
// schedules itself, op for op as llama.cpp b11806 builds it, with the same weight placement and backend scheduler:
//   * q8_1-quantised activations into MMQ/MMVQ (ggml-cuda mul_mat / mul_mat_id), q8_K/q8_0 on the CPU path;
//   * the Mamba-2 scan of ggml_ssm_scan (chunked SSD for > 128 tokens, per-token otherwise: ssm-scan.cu:851-858);
//   * flash attention over an f16 KV cache, n_kv padded to 256 (llama-kv-cache.cpp:1285-1299);
//   * the same reduction orders (MoE: expert outputs summed in selection order, llama-graph.cpp:2560-2575).
// Placement as llama.cpp `-ngl 99 --n-cpu-moe N`: every tensor in VRAM except token_embd and the routed experts of
// blocks < N, which stay in the GGUF mapping (CPU buffers from the mmap); ggml_backend_sched runs those on the CPU,
// or copies the used experts to the GPU for batches >= 32 (op offload; the copy callback of llama-context.cpp:2778).
//
// The forward pass, as llama.cpp b11806 builds it (src/models/nemotron-h.cpp:202-276, mamba-base.cpp:153-305,
// llama-graph.cpp build_norm :1759, build_ffn :1924, build_moe_ffn :2191, build_attn :3212, build_attn_mha :2948,
// build_rs :3882):
//   x = token_embd[tok];  per block: x += mixer(rms(x) * attn_norm)
//   Mamba-2 (40), attention (8: 7 16 25 36 47 58 69 78, no RoPE), LatentMoE (40: sigmoid router, top-22 by p + bias,
//   w = p / sum * 5, latent 1024, relu^2 experts, shared relu^2 expert on the full hidden);  logits = W_out rms(x)
//
// State: per slot (session) the Mamba conv + SSM states (llama_memory_recurrent layout: cache_r_l [3*10240, slots],
// cache_s_l [128*8192, slots]) and the f16 K/V cache (one stream per slot, [256, n_ctx, slots]); a ubatch reads and
// writes its slot in place, so --batch N sessions only need N slots.  The serve line protocol (READY/GEN/T/DONE/STOP/
// PP/BDONE, as src/qwen35/q36_engine.cu) sits on top of decode() (next step).
//
// Usage (golden-set capture, the format of bench/golden/golden_tool.cpp and nhref):
//   strata-nh-engine --model SHARD --list capture_list.txt --tokdir DIR --out RAWDIR [--k 32] [--chunk 512]
//                    [--ctx 32768] [--threads 16] [--n-cpu-moe 68] [--force RAWDIR] [--only id,id]
//                    [--dump-sums FILE --dump-prompt ID]   (every graph node: name, op, shape, sum; bisecting)
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/nemotron_h/layout.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace nh = strata::nemotron_h;

namespace {

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::vector<int32_t> read_i32(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + p);
    std::string b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<int32_t> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

// ---- one output row, exactly as golden_tool.cpp row_writer (and nhref make_row) ----
struct Row {
    std::vector<int32_t> ids;
    std::vector<float> lg;
    float lse = 0, tlp = 0;
};

void make_row(const float* lg, int nv, int K, int target, Row& r) {
    float mx = -INFINITY;
    for (int i = 0; i < nv; ++i) mx = std::max(mx, lg[i]);
    double s = 0;
    for (int i = 0; i < nv; ++i) s += std::exp((double) lg[i] - mx);
    r.lse = (float) (mx + std::log(s));
    std::vector<std::pair<float, int>> hp;
    hp.reserve((size_t) K + 1);
    auto worse = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    for (int i = 0; i < nv; ++i) {
        std::pair<float, int> e{lg[i], i};
        if ((int) hp.size() < K) {
            hp.push_back(e);
            std::push_heap(hp.begin(), hp.end(), worse);
        } else if (worse(e, hp.front())) {
            std::pop_heap(hp.begin(), hp.end(), worse);
            hp.back() = e;
            std::push_heap(hp.begin(), hp.end(), worse);
        }
    }
    std::sort(hp.begin(), hp.end(), worse);
    r.ids.resize((size_t) K);
    r.lg.resize((size_t) K);
    for (int k = 0; k < K; ++k) {
        r.ids[(size_t) k] = hp[(size_t) k].second;
        r.lg[(size_t) k] = hp[(size_t) k].first;
    }
    r.tlp = target >= 0 ? lg[target] - r.lse : NAN;
}

int argmax(const float* lg, int nv) {
    int am = 0;
    for (int i = 1; i < nv; ++i)
        if (lg[i] > lg[am]) am = i;
    return am;
}

// ---- weights ----
struct LayerW {
    nh::Kind kind = nh::Kind::Mamba;
    ggml_tensor* attn_norm = nullptr;
    ggml_tensor *ssm_in = nullptr, *ssm_conv1d = nullptr, *ssm_conv1d_b = nullptr, *ssm_dt_b = nullptr, *ssm_a = nullptr,
                *ssm_d = nullptr, *ssm_norm = nullptr, *ssm_out = nullptr;
    ggml_tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    ggml_tensor *gate_inp = nullptr, *exp_probs_b = nullptr, *latent_down = nullptr, *latent_up = nullptr,
                *up_exps = nullptr, *down_exps = nullptr, *up_shexp = nullptr, *down_shexp = nullptr;
};

struct Weights {
    nh::Geometry g;
    ggml_context* ctx_gpu = nullptr;
    ggml_context* ctx_cpu = nullptr;
    ggml_backend_buffer_t buf_gpu = nullptr;
    std::vector<ggml_backend_buffer_t> bufs_cpu;
    ggml_tensor *tok_embd = nullptr, *output_norm = nullptr, *output = nullptr;
    std::vector<LayerW> L;
    size_t gpu_bytes = 0, cpu_bytes = 0;
    int n_cpu_tensors = 0;
};

// Placement as llama.cpp with `-ngl 99 --n-cpu-moe N` (common/arg.cpp: the override pattern
// blk\.(0..N-1)\.ffn_(up|down|gate|gate_up)_exps -> CPU; with mmap these become CPU_Mapped buffers, and token_embd is
// an input-layer tensor on the CPU: llama-model.cpp dev_input).
void load_weights(const nh::Model& m, int n_cpu_moe, ggml_backend_buffer_type_t buft_gpu, Weights& W) {
    W.g = m.g;
    const int nl = m.g.n_layer;
    const size_t n_t = 8 + (size_t) nl * 24;
    ggml_init_params ip{ggml_tensor_overhead() * n_t, nullptr, true};
    W.ctx_gpu = ggml_init(ip);
    W.ctx_cpu = ggml_init(ip);
    std::vector<std::pair<ggml_tensor*, const nh::Tensor*>> gpu, cpu;
    auto mk = [&](const nh::Tensor& t, bool on_cpu) -> ggml_tensor* {
        if (!t) return nullptr;
        int64_t ne[4] = {1, 1, 1, 1};
        const int nd = (int) t.info->shape.size();
        for (int i = 0; i < nd; ++i) ne[i] = (int64_t) t.info->shape[(size_t) i];
        ggml_tensor* x = ggml_new_tensor(on_cpu ? W.ctx_cpu : W.ctx_gpu, (ggml_type) t.type, nd, ne);
        ggml_set_name(x, t.info->name.c_str());
        if (ggml_nbytes(x) != t.bytes)
            throw std::runtime_error("nh_engine: byte count of " + t.info->name + " differs from the GGUF entry");
        (on_cpu ? cpu : gpu).push_back({x, &t});
        return x;
    };
    W.tok_embd = mk(m.tok_embd, true);
    W.output_norm = mk(m.output_norm, false);
    W.output = mk(m.output, false);
    W.L.resize((size_t) nl);
    for (int il = 0; il < nl; ++il) {
        const nh::Layer& s = m.layers[(size_t) il];
        LayerW& d = W.L[(size_t) il];
        d.kind = s.kind;
        d.attn_norm = mk(s.attn_norm, false);
        d.ssm_in = mk(s.ssm_in, false);
        d.ssm_conv1d = mk(s.ssm_conv1d, false);
        d.ssm_conv1d_b = mk(s.ssm_conv1d_b, false);
        d.ssm_dt_b = mk(s.ssm_dt_b, false);
        d.ssm_a = mk(s.ssm_a, false);
        d.ssm_d = mk(s.ssm_d, false);
        d.ssm_norm = mk(s.ssm_norm, false);
        d.ssm_out = mk(s.ssm_out, false);
        d.wq = mk(s.wq, false);
        d.wk = mk(s.wk, false);
        d.wv = mk(s.wv, false);
        d.wo = mk(s.wo, false);
        d.gate_inp = mk(s.gate_inp, false);
        d.exp_probs_b = mk(s.exp_probs_b, false);
        d.latent_down = mk(s.latent_down, false);
        d.latent_up = mk(s.latent_up, false);
        const bool exps_cpu = il < n_cpu_moe;
        d.down_exps = mk(s.down_exps, exps_cpu);
        d.up_exps = mk(s.up_exps, exps_cpu);
        d.down_shexp = mk(s.down_shexp, false);
        d.up_shexp = mk(s.up_shexp, false);
    }
    W.buf_gpu = ggml_backend_alloc_ctx_tensors_from_buft(W.ctx_gpu, buft_gpu);
    if (!W.buf_gpu) throw std::runtime_error("nh_engine: VRAM allocation of the weights failed");
    ggml_backend_buffer_set_usage(W.buf_gpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (auto& p : gpu) {
        ggml_backend_tensor_set(p.first, p.second->data, 0, ggml_nbytes(p.first));
        W.gpu_bytes += ggml_nbytes(p.first);
    }
    for (auto& p : cpu) {
        void* ptr = (void*) p.second->data;
        ggml_backend_buffer_t b = ggml_backend_cpu_buffer_from_ptr(ptr, ggml_nbytes(p.first));
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (ggml_backend_tensor_alloc(b, p.first, ptr) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("nh_engine: cannot map " + std::string(p.first->name));
        W.bufs_cpu.push_back(b);
        W.cpu_bytes += ggml_nbytes(p.first);
        ++W.n_cpu_tensors;
    }
}

// ---- the copy callback of llama-context.cpp:2778-2858 (sched_copy_experts, without the MoE cache): when the
// scheduler offloads a mul_mat_id whose experts live in host memory, upload only the experts the ids select ----
struct CopyExperts {
    const ggml_tensor* ids = nullptr;
    std::vector<int32_t> ids_data;
    std::vector<bool> used;
    void reset() {
        ids = nullptr;
        used.clear();
    }
};

bool copy_experts_cb(ggml_backend_t backend, const ggml_tensor* src, ggml_tensor* dst, ggml_cgraph* graph, void* ud) {
    CopyExperts& st = *static_cast<CopyExperts*>(ud);
    if (ggml_graph_n_nodes(graph) == 0) return false;
    const ggml_tensor* node = ggml_graph_node(graph, 0);
    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != dst) return false;
    const ggml_tensor* ids = node->src[2];
    if (ggml_nelements(ids) == 0) return true;
    const int64_t n_expert = src->ne[2];
    const size_t expert_size = src->nb[2];
    if (ids != st.ids || (int64_t) st.used.size() != n_expert) {
        st.ids_data.resize(ggml_nbytes(ids) / sizeof(int32_t));
        ggml_backend_tensor_get_async(backend, ids, st.ids_data.data(), 0, ggml_nbytes(ids));
        ggml_backend_synchronize(backend);
        st.used.assign((size_t) n_expert, false);
        for (int64_t i1 = 0; i1 < ids->ne[1]; i1++)
            for (int64_t i0 = 0; i0 < ids->ne[0]; i0++) {
                const int32_t id = st.ids_data[i1 * ids->nb[1] / sizeof(int32_t) + i0 * ids->nb[0] / sizeof(int32_t)];
                GGML_ASSERT(id >= 0 && id < n_expert);
                st.used[(size_t) id] = true;
            }
        st.ids = ids;
    }
    for (int64_t first = 0; first < n_expert;) {
        if (!st.used[(size_t) first]) {
            first++;
            continue;
        }
        int64_t last = first;
        while (last + 1 < n_expert && st.used[(size_t) last + 1]) last++;
        // a bit extra so the padding of the last expert holds no NaNs (MMQ reads it)
        const size_t offset = (size_t) first * expert_size;
        const size_t padding = last < n_expert - 1 ? std::min<size_t>(expert_size, 512) : 0;
        const size_t size = (size_t) (last + 1 - first) * expert_size + padding;
        ggml_backend_tensor_set_async(backend, dst, (const uint8_t*) src->data + offset, offset, size);
        first = last + 1;
    }
    return true;
}

// ---- per-slot state: llama_memory_recurrent (R conv, S ssm) + llama_kv_cache (f16 K/V, one stream per slot) ----
struct Memory {
    int n_slots = 1, n_ctx = 0;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor*> r, s, k, v;   // by layer
    std::vector<int> n_past;                 // tokens stored per slot
};

struct GraphIn {
    ggml_tensor *tokens = nullptr, *s_copy = nullptr, *s_copy_main = nullptr, *s_copy_tail = nullptr;
    ggml_tensor *k_idxs = nullptr, *v_idxs = nullptr, *kq_mask = nullptr, *out_ids = nullptr, *logits = nullptr;
};

struct DumpCtx {
    FILE* f = nullptr;
    std::vector<float> tmp;
};

bool dump_cb(ggml_tensor* t, bool ask, void* ud) {
    DumpCtx& d = *static_cast<DumpCtx*>(ud);
    if (ask) return true;
    if (!d.f || t->type != GGML_TYPE_F32 || ggml_is_empty(t)) return true;
    if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE)
        return true;
    if (!ggml_is_contiguous(t)) return true;
    const size_t n = (size_t) ggml_nelements(t);
    d.tmp.resize(n);
    ggml_backend_tensor_get(t, d.tmp.data(), 0, n * 4);
    double s = 0, sa = 0;
    for (size_t i = 0; i < n; ++i) {
        s += d.tmp[i];
        sa += std::fabs((double) d.tmp[i]);
    }
    std::fprintf(d.f, "%s\t%s\t%lld,%lld,%lld,%lld\t%.9e\t%.9e\t%.9e\t%.9e\n", t->name, ggml_op_desc(t),
                 (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], s, sa,
                 n ? (double) d.tmp[0] : 0.0, n ? (double) d.tmp[n - 1] : 0.0);
    return true;
}

struct Engine {
    Weights W;
    Memory M;
    ggml_backend_t gpu = nullptr, cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    CopyExperts cx;
    DumpCtx dump;
    int n_threads = 16;
    bool force_norm = true;   // llama-context.cpp graph_get_cb: "norm" pinned to the layer's device (full offload)
    int max_nodes = 16384;
    std::vector<uint8_t> meta;
    std::vector<float> logits;   // host, [n_out][n_vocab]
    ggml_context* ctx0 = nullptr;
    ggml_cgraph* gf = nullptr;

    int64_t n_embd_r() const { return (int64_t) (W.g.ssm_d_conv - 1) * W.g.ssm_conv_dim; }
    int64_t n_embd_s() const { return (int64_t) W.g.ssm_d_state * W.g.ssm_d_inner; }

    void init_memory(int n_slots, int n_ctx) {
        M.n_slots = n_slots;
        M.n_ctx = n_ctx;
        const int nl = W.g.n_layer;
        ggml_init_params ip{ggml_tensor_overhead() * (size_t) (4 * nl + 4), nullptr, true};
        M.ctx = ggml_init(ip);
        M.r.assign((size_t) nl, nullptr);
        M.s.assign((size_t) nl, nullptr);
        M.k.assign((size_t) nl, nullptr);
        M.v.assign((size_t) nl, nullptr);
        const int64_t n_embd_kv = (int64_t) W.g.n_head_kv * W.g.head_dim;
        for (int il = 0; il < nl; ++il) {
            const nh::Kind kd = W.L[(size_t) il].kind;
            if (kd == nh::Kind::Mamba) {
                M.r[(size_t) il] = ggml_new_tensor_2d(M.ctx, GGML_TYPE_F32, n_embd_r(), n_slots);
                M.s[(size_t) il] = ggml_new_tensor_2d(M.ctx, GGML_TYPE_F32, n_embd_s(), n_slots);
                ggml_format_name(M.r[(size_t) il], "cache_r_l%d", il);
                ggml_format_name(M.s[(size_t) il], "cache_s_l%d", il);
            } else if (kd == nh::Kind::Attention) {
                M.k[(size_t) il] = ggml_new_tensor_3d(M.ctx, GGML_TYPE_F16, n_embd_kv, n_ctx, n_slots);
                M.v[(size_t) il] = ggml_new_tensor_3d(M.ctx, GGML_TYPE_F16, n_embd_kv, n_ctx, n_slots);
                ggml_format_name(M.k[(size_t) il], "cache_k_l%d", il);
                ggml_format_name(M.v[(size_t) il], "cache_v_l%d", il);
            }
        }
        M.buf = ggml_backend_alloc_ctx_tensors(M.ctx, gpu);
        if (!M.buf) throw std::runtime_error("nh_engine: VRAM allocation of the state failed");
        ggml_backend_buffer_clear(M.buf, 0);
        M.n_past.assign((size_t) n_slots, 0);
    }

    // build_norm (llama-graph.cpp:1759) with the graph callback of llama-context.cpp graph_get_cb
    ggml_tensor* build_norm(ggml_tensor* x, ggml_tensor* w, int il) {
        ggml_tensor* cur = ggml_rms_norm(ctx0, x, W.g.rms_eps);
        if (il >= 0) {
            ggml_format_name(cur, "norm-%d", il);
            if (force_norm) ggml_backend_sched_set_tensor_backend(sched, cur, gpu);
        } else {
            ggml_set_name(cur, "norm");
        }
        return ggml_mul(ctx0, cur, w);
    }

    // build_rs (llama-graph.cpp:3882-3946)
    ggml_tensor* build_rs(ggml_tensor* s, ggml_tensor* state_copy, ggml_tensor* state_copy_main, int64_t state_size,
                          int64_t n_seqs, int64_t n_rs, int64_t rs_head, int64_t rs_zero,
                          const std::function<ggml_tensor*(ggml_context*, ggml_tensor*, ggml_tensor*)>& get_rows_fn) {
        ggml_tensor* states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);
        ggml_tensor* state_zero = ggml_view_1d(ctx0, states, state_size * (rs_zero >= 0),
                                               (size_t) (rs_zero >= 0 ? rs_zero : 0) * states->nb[1]);
        ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));
        const int64_t i0 = n_rs - state_copy->ne[0];
        ggml_tensor* states_all = ggml_get_rows(ctx0, states, state_copy);
        ggml_tensor* output_states = get_rows_fn ? get_rows_fn(ctx0, states, state_copy_main)
                                                 : ggml_view_2d(ctx0, states_all, state_size, n_seqs, states_all->nb[1], 0);
        ggml_build_forward_expand(gf, output_states);
        ggml_tensor* states_extra = ggml_view_2d(ctx0, states_all, state_size, n_rs - n_seqs, states_all->nb[1],
                                                 (size_t) (n_seqs - i0) * states_all->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, states_extra,
                                               ggml_view_2d(ctx0, s, state_size, n_rs - n_seqs, s->nb[1],
                                                            (size_t) (rs_head + n_seqs) * s->nb[1])));
        return output_states;
    }

    // build_mamba2_layer (mamba-base.cpp:153-305), one sequence per ubatch, no MTP (K = 1)
    ggml_tensor* build_mamba2(ggml_tensor* cur, int il, int64_t T, int slot, int64_t rs_zero, GraphIn& gi) {
        const LayerW& l = W.L[(size_t) il];
        const nh::Geometry& g = W.g;
        const int64_t d_conv = g.ssm_d_conv, d_inner = g.ssm_d_inner, d_state = g.ssm_d_state, n_head = g.ssm_n_head;
        const int64_t head_dim = d_inner / n_head, n_group = g.ssm_n_group;
        const int64_t n_seqs = 1, n_seq_tokens = T, K = 1;
        const int64_t mem_size = M.n_slots, kv_head = slot, n_rs = 1;
        ggml_tensor* conv_states_all = M.r[(size_t) il];
        ggml_tensor* ssm_states_all = M.s[(size_t) il];
        const int64_t state_slots = ssm_states_all->ne[1];
        ggml_tensor* conv = build_rs(conv_states_all, gi.s_copy, gi.s_copy_main, n_embd_r(), n_seqs, n_rs, kv_head,
                                     rs_zero, nullptr);
        conv = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner + 2 * n_group * d_state, n_seqs);
        ggml_tensor* zxBCdt = ggml_mul_mat(ctx0, l.ssm_in, cur);
        zxBCdt = ggml_reshape_3d(ctx0, zxBCdt, zxBCdt->ne[0], n_seq_tokens, n_seqs);
        ggml_tensor* z = ggml_view_4d(ctx0, zxBCdt, head_dim, n_head, n_seq_tokens, n_seqs, head_dim * zxBCdt->nb[0],
                                      zxBCdt->nb[1], zxBCdt->nb[2], 0);
        ggml_tensor* xBC = ggml_view_3d(ctx0, zxBCdt, d_inner + 2 * n_group * d_state, n_seq_tokens, n_seqs,
                                        zxBCdt->nb[1], zxBCdt->nb[2], d_inner * ggml_element_size(zxBCdt));
        ggml_tensor* dt = ggml_view_3d(ctx0, zxBCdt, n_head, n_seq_tokens, n_seqs, zxBCdt->nb[1], zxBCdt->nb[2],
                                       (2 * d_inner + 2 * n_group * d_state) * ggml_element_size(zxBCdt));
        {
            ggml_tensor* conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, xBC), 0);
            const int64_t row_count = (d_conv - 1) * (d_inner + 2 * n_group * d_state);
            const size_t row_size = ggml_row_size(conv_states_all->type, row_count);
            const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);
            for (int64_t sl = 0; sl < n_written; ++sl) {
                ggml_tensor* last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner + 2 * n_group * d_state, n_seqs,
                                                      conv_x->nb[1], conv_x->nb[2], (n_seq_tokens - sl) * conv_x->nb[0]);
                ggml_build_forward_expand(
                    gf, ggml_cpy(ctx0, last_conv,
                                 ggml_view_2d(ctx0, conv_states_all, row_count, n_seqs, conv_states_all->nb[1],
                                              ((size_t) sl * mem_size + kv_head) * row_size)));
            }
            xBC = ggml_ssm_conv(ctx0, conv_x, l.ssm_conv1d);
            xBC = ggml_add(ctx0, xBC, l.ssm_conv1d_b);
            xBC = ggml_silu(ctx0, xBC);
        }
        ggml_tensor* x = ggml_view_4d(ctx0, xBC, head_dim, n_head, n_seq_tokens, n_seqs, head_dim * xBC->nb[0],
                                      xBC->nb[1], xBC->nb[2], 0);
        ggml_tensor* B = ggml_view_4d(ctx0, xBC, d_state, n_group, n_seq_tokens, n_seqs, d_state * xBC->nb[0],
                                      xBC->nb[1], xBC->nb[2], d_inner * ggml_element_size(xBC));
        ggml_tensor* C = ggml_view_4d(ctx0, xBC, d_state, n_group, n_seq_tokens, n_seqs, d_state * xBC->nb[0],
                                      xBC->nb[1], xBC->nb[2], (d_inner + n_group * d_state) * ggml_element_size(xBC));
        dt = ggml_add(ctx0, ggml_cont(ctx0, dt), l.ssm_dt_b);
        ggml_tensor* A = l.ssm_a;
        auto get_ssm_rows = [&](ggml_context* ctx, ggml_tensor* states, ggml_tensor* ids) {
            ggml_tensor* ssm = ggml_reshape_4d(ctx, states, d_state, head_dim, n_head, state_slots);
            return ggml_ssm_scan(ctx, ssm, x, dt, A, B, C, ids, K);
        };
        ggml_tensor* y_ssm = build_rs(ssm_states_all, gi.s_copy_tail, gi.s_copy_main, n_embd_s(), n_seqs, n_rs, kv_head,
                                      rs_zero, get_ssm_rows);
        {
            const int64_t D = d_state * d_inner;
            const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);
            const size_t row_size = ggml_row_size(ssm_states_all->type, D);
            const size_t y_row_size = ggml_row_size(y_ssm->type, D);
            const size_t state_offset = ggml_nelements(x) * ggml_element_size(x);
            ggml_build_forward_expand(
                gf, ggml_cpy(ctx0, ggml_view_3d(ctx0, y_ssm, D, n_seqs, n_written, y_row_size, y_row_size * n_seqs, state_offset),
                             ggml_view_3d(ctx0, ssm_states_all, D, n_seqs, n_written, ssm_states_all->nb[1],
                                          (size_t) mem_size * row_size, kv_head * row_size)));
        }
        ggml_tensor* y = ggml_view_4d(ctx0, y_ssm, head_dim, n_head, n_seq_tokens, n_seqs, x->nb[1], n_head * x->nb[1],
                                      n_seq_tokens * n_head * x->nb[1], 0);
        y = ggml_add(ctx0, y, ggml_mul(ctx0, x, l.ssm_d));
        ggml_format_name(y, "mamba2_y_add_d-%d", il);
        y = ggml_swiglu_split(ctx0, ggml_cont(ctx0, z), y);
        y = ggml_reshape_4d(ctx0, y, d_inner / n_group, n_group, n_seq_tokens, n_seqs);
        y = build_norm(y, l.ssm_norm, il);
        y = ggml_reshape_2d(ctx0, y, d_inner, n_seq_tokens * n_seqs);
        cur = ggml_mul_mat(ctx0, l.ssm_out, y);
        ggml_format_name(cur, "mamba_out-%d", il);
        return cur;
    }

    // build_attention_layer (nemotron-h.cpp:277-292) -> build_qkv (separate Q/K/V), build_attn (:3212), mha (:2948)
    ggml_tensor* build_attention(ggml_tensor* cur, int il, int64_t T, int slot, int64_t n_kv, GraphIn& gi) {
        const LayerW& l = W.L[(size_t) il];
        const int64_t hd = W.g.head_dim, n_head = W.g.n_head, n_head_kv = W.g.n_head_kv;
        ggml_tensor* Qcur = ggml_mul_mat(ctx0, l.wq, cur);
        ggml_format_name(Qcur, "Qcur-%d", il);
        ggml_tensor* Kcur = ggml_mul_mat(ctx0, l.wk, cur);
        ggml_format_name(Kcur, "Kcur-%d", il);
        ggml_tensor* Vcur = ggml_mul_mat(ctx0, l.wv, cur);
        ggml_format_name(Vcur, "Vcur-%d", il);
        Qcur = ggml_reshape_3d(ctx0, Qcur, hd, n_head, T);
        Kcur = ggml_reshape_3d(ctx0, Kcur, hd, n_head_kv, T);
        Vcur = ggml_reshape_3d(ctx0, Vcur, hd, n_head_kv, T);
        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Vcur);
        ggml_build_forward_expand(gf, Kcur);
        ggml_tensor* kc = M.k[(size_t) il];
        ggml_tensor* vc = M.v[(size_t) il];
        const int64_t n_embd_gqa = hd * n_head_kv, kv_size = M.n_ctx, n_stream = kc->ne[2];
        {   // cpy_k / cpy_v (llama-kv-cache.cpp:1353-1442), the V cache not transposed (flash attention)
            ggml_tensor* k_cur = ggml_view_2d(ctx0, Kcur, n_embd_gqa, T, Kcur->nb[2], 0);
            ggml_tensor* kd = n_stream > 1 ? ggml_reshape_2d(ctx0, kc, n_embd_gqa, kv_size * n_stream) : kc;
            ggml_build_forward_expand(gf, ggml_set_rows(ctx0, kd, k_cur, gi.k_idxs));
            ggml_tensor* v_cur = ggml_view_2d(ctx0, Vcur, n_embd_gqa, T, Vcur->nb[2], 0);
            ggml_tensor* vd = n_stream > 1 ? ggml_reshape_2d(ctx0, vc, n_embd_gqa, kv_size * n_stream) : vc;
            ggml_build_forward_expand(gf, ggml_set_rows(ctx0, vd, v_cur, gi.v_idxs));
        }
        // get_k / get_v (llama-kv-cache.cpp:1301-1351): one stream (the slot), n_kv cells
        ggml_tensor* k = ggml_view_4d(ctx0, kc, hd, n_head_kv, n_kv, 1, ggml_row_size(kc->type, hd),
                                      ggml_row_size(kc->type, n_embd_gqa), ggml_row_size(kc->type, n_embd_gqa * kv_size),
                                      ggml_row_size(kc->type, n_embd_gqa * kv_size) * (size_t) slot);
        ggml_tensor* v = ggml_view_4d(ctx0, vc, hd, n_head_kv, n_kv, 1, ggml_row_size(vc->type, hd),
                                      ggml_row_size(vc->type, n_embd_gqa), ggml_row_size(vc->type, n_embd_gqa * kv_size),
                                      ggml_row_size(vc->type, n_embd_gqa * kv_size) * (size_t) slot);
        ggml_tensor* q = Qcur;
        const int64_t ns = k->ne[3];
        q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2] / ns, ns, q->nb[1], q->nb[2], q->nb[2] * (q->ne[2] / ns), 0);
        q = ggml_permute(ctx0, q, 0, 2, 1, 3);
        k = ggml_permute(ctx0, k, 0, 2, 1, 3);
        v = ggml_permute(ctx0, v, 0, 2, 1, 3);
        const float kq_scale = 1.0f / std::sqrt((float) hd);
        ggml_tensor* o = ggml_flash_attn_ext(ctx0, q, k, v, gi.kq_mask, kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_add_sinks(o, nullptr);
        ggml_flash_attn_ext_set_n_kv_max(o, 0);
        ggml_prec_set_acc(o, GGML_PREC_F32);
        o = ggml_reshape_2d(ctx0, o, o->ne[0] * o->ne[1], o->ne[2] * o->ne[3]);
        ggml_build_forward_expand(gf, o);
        ggml_format_name(o, "kqv_out-%d", il);
        o = ggml_mul_mat(ctx0, l.wo, o);
        ggml_format_name(o, "attn_out-%d", il);
        return o;
    }

    // build_ffn_layer (nemotron-h.cpp:293-345) with build_moe_ffn (llama-graph.cpp:2191-2582) and build_ffn (:1924)
    ggml_tensor* build_moe(ggml_tensor* cur, int il) {
        const LayerW& l = W.L[(size_t) il];
        const nh::Geometry& g = W.g;
        ggml_tensor* inp_emb = cur;
        ggml_tensor* inp_latent = ggml_mul_mat(ctx0, l.latent_down, cur);
        ggml_tensor* router_logits = ggml_mul_mat(ctx0, l.gate_inp, cur);
        ggml_format_name(router_logits, "ffn_moe_logits-%d", il);
        const int64_t n_expert = l.up_exps->ne[2], n_used = g.n_expert_used;
        const int64_t n_embd = inp_latent->ne[0], n_tokens = inp_latent->ne[1];
        ggml_tensor* probs = ggml_sigmoid(ctx0, router_logits);
        ggml_format_name(probs, "ffn_moe_probs-%d", il);
        ggml_tensor* selection_probs = ggml_add(ctx0, probs, l.exp_probs_b);
        ggml_format_name(selection_probs, "ffn_moe_probs_biased-%d", il);
        ggml_tensor* selected = ggml_argsort_top_k(ctx0, selection_probs, (int) n_used);
        ggml_format_name(selected->src[0], "ffn_moe_argsort-%d", il);
        ggml_format_name(selected, "ffn_moe_topk-%d", il);
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
        ggml_tensor* weights = ggml_get_rows(ctx0, probs, selected);
        ggml_format_name(weights, "ffn_moe_weights-%d", il);
        if (g.expert_weights_norm) {
            weights = ggml_reshape_2d(ctx0, weights, n_used, n_tokens);
            ggml_tensor* weights_sum = ggml_sum_rows(ctx0, weights);
            ggml_format_name(weights_sum, "ffn_moe_weights_sum-%d", il);
            weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
            ggml_format_name(weights_sum, "ffn_moe_weights_sum_clamped-%d", il);
            weights = ggml_div(ctx0, weights, weights_sum);
            ggml_format_name(weights, "ffn_moe_weights_norm-%d", il);
            weights = ggml_reshape_3d(ctx0, weights, 1, n_used, n_tokens);
        }
        if (g.expert_weights_scale != 0.0f && g.expert_weights_scale != 1.0f) {
            weights = ggml_scale(ctx0, weights, g.expert_weights_scale);
            ggml_format_name(weights, "ffn_moe_weights_scaled-%d", il);
        }
        ggml_build_forward_expand(gf, weights);
        cur = ggml_reshape_3d(ctx0, inp_latent, n_embd, 1, n_tokens);
        ggml_tensor* up = ggml_mul_mat_id(ctx0, l.up_exps, cur, selected);
        ggml_format_name(up, "ffn_moe_up-%d", il);
        cur = ggml_relu(ctx0, up);
        cur = ggml_sqr(ctx0, cur);
        ggml_format_name(cur, "ffn_moe_relu_sqr-%d", il);
        ggml_tensor* experts = ggml_mul_mat_id(ctx0, l.down_exps, cur, selected);
        ggml_format_name(experts, "ffn_moe_down-%d", il);
        experts = ggml_mul(ctx0, experts, weights);
        ggml_format_name(experts, "ffn_moe_weighted-%d", il);
        ggml_build_forward_expand(gf, experts);
        std::vector<ggml_tensor*> cur_experts((size_t) n_used);
        for (int64_t i = 0; i < n_used; ++i) {
            cur_experts[(size_t) i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i * experts->nb[1]);
            ggml_build_forward_expand(gf, cur_experts[(size_t) i]);
        }
        ggml_tensor* moe_out = cur_experts[0];
        for (int64_t i = 1; i < n_used; ++i) {
            moe_out = ggml_add(ctx0, moe_out, cur_experts[(size_t) i]);
            ggml_build_forward_expand(gf, moe_out);
        }
        if (n_used == 1) moe_out = ggml_cont(ctx0, moe_out);
        ggml_format_name(moe_out, "ffn_moe_out-%d", il);
        moe_out = ggml_mul_mat(ctx0, l.latent_up, moe_out);
        ggml_tensor* tmp = ggml_mul_mat(ctx0, l.up_shexp, inp_emb);
        ggml_format_name(tmp, "ffn_up-%d", il);
        ggml_tensor* sh = ggml_relu(ctx0, tmp);
        ggml_format_name(sh, "ffn_relu-%d", il);
        sh = ggml_sqr(ctx0, sh);
        ggml_format_name(sh, "ffn_sqr(relu)-%d", il);
        sh = ggml_mul_mat(ctx0, l.down_shexp, sh);
        ggml_format_name(sh, "ffn_shexp-%d", il);
        cur = ggml_add(ctx0, moe_out, sh);
        ggml_format_name(cur, "l_out-%d", il);
        return cur;
    }

    // the graph of one ubatch of one sequence (slot): T tokens at positions pos0.., n_out outputs (all or the last)
    void build_graph(int64_t T, int slot, int64_t pos0, int64_t n_kv, int64_t n_out, int64_t rs_zero, GraphIn& gi) {
        ggml_init_params ip{meta.size(), meta.data(), true};
        ctx0 = ggml_init(ip);
        gf = ggml_new_graph_custom(ctx0, max_nodes, false);
        const nh::Geometry& g = W.g;
        (void) pos0;
        gi.tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
        ggml_set_name(gi.tokens, "inp_tokens");
        ggml_set_input(gi.tokens);
        ggml_tensor* inpL = ggml_get_rows(ctx0, W.tok_embd, gi.tokens);
        ggml_set_name(inpL, "embd");
        ggml_build_forward_expand(gf, inpL);
        // build_inp_mem_hybrid: the attention inputs and the recurrent-state inputs
        gi.k_idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, T);
        ggml_set_name(gi.k_idxs, "attn_inp_k_idxs");
        ggml_set_input(gi.k_idxs);
        gi.v_idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, T);
        ggml_set_name(gi.v_idxs, "attn_inp_v_idxs");
        ggml_set_input(gi.v_idxs);
        gi.kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, n_kv, T, 1, 1);
        ggml_set_name(gi.kq_mask, "attn_inp_kq_mask");
        ggml_set_input(gi.kq_mask);
        const int64_t n_rs = 1, n_seqs = 1;
        gi.s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
        ggml_set_name(gi.s_copy, "rs_s_copy");
        ggml_set_input(gi.s_copy);
        gi.s_copy_main = ggml_view_1d(ctx0, gi.s_copy, n_seqs, 0);
        gi.s_copy_tail = ggml_view_1d(ctx0, gi.s_copy, n_rs - 1, gi.s_copy->nb[0]);
        gi.out_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_out);
        ggml_set_name(gi.out_ids, "out_ids");
        ggml_set_input(gi.out_ids);
        for (int il = 0; il < g.n_layer; ++il) {
            ggml_tensor* inpSA = inpL;
            ggml_tensor* cur = build_norm(inpL, W.L[(size_t) il].attn_norm, il);
            ggml_format_name(cur, "attn_norm-%d", il);
            switch (W.L[(size_t) il].kind) {
            case nh::Kind::Mamba: cur = build_mamba2(cur, il, T, slot, rs_zero, gi); break;
            case nh::Kind::Attention: cur = build_attention(cur, il, T, slot, n_kv, gi); break;
            case nh::Kind::Moe: cur = build_moe(cur, il); break;
            default: throw std::runtime_error("nh_engine: unexpected block kind");
            }
            if (il == g.n_layer - 1) {
                cur = ggml_get_rows(ctx0, cur, gi.out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, gi.out_ids);
            }
            cur = ggml_add(ctx0, cur, inpSA);
            ggml_format_name(cur, "nemotron_h_block_out-%d", il);
            inpL = cur;
        }
        ggml_tensor* cur = build_norm(inpL, W.output_norm, -1);
        ggml_set_name(cur, "result_norm");
        cur = ggml_mul_mat(ctx0, W.output, cur);
        ggml_set_name(cur, "result_output");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
        gi.logits = cur;
    }

    void init(const nh::Model& m, int n_cpu_moe, int n_slots, int n_ctx, int threads, int ubatch) {
        gpu = ggml_backend_cuda_init(0);
        cpu = ggml_backend_cpu_init();
        if (!gpu || !cpu) throw std::runtime_error("nh_engine: backend init failed");
        n_threads = threads;
        ggml_backend_cpu_set_n_threads(cpu, n_threads);
        load_weights(m, n_cpu_moe, ggml_backend_get_default_buffer_type(gpu), W);
        init_memory(n_slots, n_ctx);
        ggml_backend_t backs[2] = {gpu, cpu};
        // the CPU backend computes into pinned host memory (llama-context.cpp:505-518: host buffer of the first device)
        ggml_backend_buffer_type_t bufts[2] = {ggml_backend_get_default_buffer_type(gpu), ggml_backend_cuda_host_buffer_type()};
        sched = ggml_backend_sched_new(backs, bufts, 2, (size_t) max_nodes, false, true);
        ggml_backend_sched_set_copy_callback(sched, copy_experts_cb, &cx);
        meta.resize(ggml_tensor_overhead() * (size_t) max_nodes + ggml_graph_overhead_custom((size_t) max_nodes, false));
        // reserve the worst case: a full ubatch, all outputs, every cell
        GraphIn gi;
        ggml_backend_sched_reset(sched);
        build_graph(ubatch, 0, 0, n_ctx, ubatch, 0, gi);
        if (!ggml_backend_sched_reserve(sched, gf)) throw std::runtime_error("nh_engine: sched reserve failed");
        ggml_free(ctx0);
        ctx0 = nullptr;
    }

    void clear_slot(int slot) { M.n_past[(size_t) slot] = 0; }

    // one ubatch of tokens for `slot`; all_out: logits for every token, else for the last one only
    void decode(const int32_t* tok, int64_t T, int slot, bool all_out) {
        const int64_t pos0 = M.n_past[(size_t) slot];
        if (pos0 + T > M.n_ctx) throw std::runtime_error("nh_engine: context full");
        const int64_t n_kv = std::min<int64_t>(M.n_ctx, std::max<int64_t>(256, ((pos0 + T + 255) / 256) * 256));
        const int64_t n_out = all_out ? T : 1;
        const int64_t rs_zero = pos0 == 0 ? slot : -1;
        GraphIn gi;
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_set_eval_callback(sched, dump.f ? dump_cb : nullptr, &dump);
        build_graph(T, slot, pos0, n_kv, n_out, rs_zero, gi);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) throw std::runtime_error("nh_engine: graph alloc failed");
        ggml_backend_tensor_set(gi.tokens, tok, 0, (size_t) T * 4);
        {
            std::vector<int64_t> idx((size_t) T);
            const int64_t base = M.k.empty() ? 0 : (int64_t) slot * M.n_ctx * (M.n_slots > 1 ? 1 : 0);
            for (int64_t i = 0; i < T; ++i) idx[(size_t) i] = base + pos0 + i;
            ggml_backend_tensor_set(gi.k_idxs, idx.data(), 0, idx.size() * 8);
            ggml_backend_tensor_set(gi.v_idxs, idx.data(), 0, idx.size() * 8);
        }
        {
            std::vector<ggml_fp16_t> mask((size_t) (n_kv * T));
            const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f), drop = ggml_fp32_to_fp16(-INFINITY);
            for (int64_t i = 0; i < T; ++i)
                for (int64_t j = 0; j < n_kv; ++j) mask[(size_t) (i * n_kv + j)] = j <= pos0 + i ? keep : drop;
            ggml_backend_tensor_set(gi.kq_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }
        {
            const int32_t sc = slot;
            ggml_backend_tensor_set(gi.s_copy, &sc, 0, 4);
            std::vector<int32_t> oi((size_t) n_out);
            for (int64_t i = 0; i < n_out; ++i) oi[(size_t) i] = (int32_t) (all_out ? i : T - 1);
            ggml_backend_tensor_set(gi.out_ids, oi.data(), 0, oi.size() * 4);
        }
        cx.reset();
        ggml_backend_cpu_set_n_threads(cpu, n_threads);
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("nh_engine: graph compute failed");
        const size_t nv = (size_t) gi.logits->ne[0];
        logits.resize(nv * (size_t) n_out);
        ggml_backend_tensor_get(gi.logits, logits.data(), 0, logits.size() * 4);
        ggml_free(ctx0);
        ctx0 = nullptr;
        M.n_past[(size_t) slot] = (int) (pos0 + T);
    }
};

std::set<std::string> split_ids(const std::string& s) {
    std::set<std::string> r;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ','))
        if (!x.empty()) r.insert(x);
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, list, tokdir, outdir, force, only, dump_sums, dump_prompt;
    int K = 32, chunk = 512, ctx = 32768, threads = 16, n_cpu_moe = 68;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--model") model = nx();
        else if (a == "--list") list = nx();
        else if (a == "--tokdir") tokdir = nx();
        else if (a == "--out") outdir = nx();
        else if (a == "--k") K = std::stoi(nx());
        else if (a == "--chunk") chunk = std::stoi(nx());
        else if (a == "--ctx") ctx = std::stoi(nx());
        else if (a == "--threads") threads = std::stoi(nx());
        else if (a == "--n-cpu-moe") n_cpu_moe = std::stoi(nx());
        else if (a == "--force") force = nx();
        else if (a == "--only") only = nx();
        else if (a == "--dump-sums") dump_sums = nx();
        else if (a == "--dump-prompt") dump_prompt = nx();
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (model.empty() || list.empty() || tokdir.empty() || outdir.empty()) {
        std::fprintf(stderr, "usage: strata-nh-engine --model SHARD --list LIST --tokdir DIR --out RAWDIR [--k 32] "
                             "[--chunk 512] [--ctx 32768] [--threads 16] [--n-cpu-moe 68] [--force RAWDIR] [--only ids] "
                             "[--dump-sums FILE --dump-prompt ID]\n");
        return 2;
    }
    const std::set<std::string> only_ids = split_ids(only);
    const double t_load0 = now_s();
    const strata::GgufModel gm = strata::GgufModel::open(model);
    const nh::Model m = nh::load_model(gm);
    Engine E;
    E.init(m, n_cpu_moe, 1, ctx, threads, chunk);
    {
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(E.gpu), &fr, &tot);
        std::fprintf(stderr, "[nh_engine] loaded in %.1f s; weights VRAM %.1f MiB, host-mapped %.1f MiB (%d tensors, "
                             "experts of blocks < %d); state %.1f MiB; GPU free %.1f MiB of %.1f\n",
                     now_s() - t_load0, E.W.gpu_bytes / 1048576.0, E.W.cpu_bytes / 1048576.0, E.W.n_cpu_tensors, n_cpu_moe,
                     ggml_backend_buffer_get_size(E.M.buf) / 1048576.0, fr / 1048576.0, tot / 1048576.0);
    }
    const int nv = m.g.n_vocab;
    const std::set<int> eog = {11, 2};   // llama-vocab.cpp:3172-3202 (<|im_end|>, </s>), :3233 (eos)
    FILE* sumf = std::fopen((outdir + "/capture_log.txt").c_str(), "w");
    if (!sumf) throw std::runtime_error("cannot write " + outdir + "/capture_log.txt");
    std::fprintf(sumf, "engine=strata-nh-engine n_vocab=%d n_ctx=%d K=%d chunk=%d force=%s n_cpu_moe=%d threads=%d\n", nv,
                 ctx, K, chunk, force.c_str(), n_cpu_moe, threads);
    std::ifstream lf(list);
    std::string id;
    int ngen;
    const unsigned nthr = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    while (lf >> id >> ngen) {
        if (!only_ids.empty() && !only_ids.count(id)) continue;
        const std::vector<int32_t> pt = read_i32(tokdir + "/" + id + ".i32");
        std::vector<int32_t> forced;
        if (!force.empty()) forced = read_i32(force + "/" + id + ".out.i32");
        const int np = (int) pt.size();
        int maxgen = std::min(ngen, ctx - np);
        if (!force.empty()) maxgen = (int) forced.size();
        E.clear_slot(0);
        E.dump.f = (!dump_sums.empty() && id == dump_prompt) ? std::fopen(dump_sums.c_str(), "w") : nullptr;
        std::vector<Row> rows;
        rows.reserve((size_t) np + (size_t) maxgen);
        std::vector<float> last((size_t) nv);
        const double t0 = now_s();
        for (int off = 0; off < np; off += chunk) {
            const int n = std::min(chunk, np - off);
            E.decode(pt.data() + off, n, 0, true);
            const size_t base = rows.size();
            const int nrow = std::min(n, np - 1 - off);
            if (nrow > 0) rows.resize(base + (size_t) nrow);
            std::vector<std::thread> th;
            for (unsigned w = 0; w < nthr; ++w)
                th.emplace_back([&, w] {
                    for (int i = (int) w; i < nrow; i += (int) nthr)
                        make_row(E.logits.data() + (size_t) i * nv, nv, K, pt[(size_t) off + i + 1], rows[base + (size_t) i]);
                });
            for (auto& t : th) t.join();
            if (off + n == np) std::memcpy(last.data(), E.logits.data() + (size_t) (n - 1) * nv, sizeof(float) * nv);
        }
        const double t1 = now_s();
        if (E.dump.f) {
            std::fclose(E.dump.f);
            E.dump.f = nullptr;
        }
        std::vector<int32_t> out;
        bool is_eog = false;
        for (int j = 0; j < maxgen; ++j) {
            const int am = argmax(last.data(), nv);
            const int t = force.empty() ? am : forced[(size_t) j];
            rows.emplace_back();
            make_row(last.data(), nv, K, t, rows.back());
            out.push_back(t);
            if (force.empty() && eog.count(t)) {
                is_eog = true;
                break;
            }
            if (j == maxgen - 1) break;
            E.decode(&out.back(), 1, 0, false);
            std::memcpy(last.data(), E.logits.data(), sizeof(float) * nv);
        }
        const double t2 = now_s();
        auto op = [&](const char* s) { return std::fopen((outdir + "/" + id + s).c_str(), "wb"); };
        FILE *fi = op(".topk_ids.i32"), *fl = op(".topk_logits.f32"), *fs = op(".lse.f32"), *ft = op(".target_lp.f32");
        for (const Row& r : rows) {
            std::fwrite(r.ids.data(), 4, (size_t) K, fi);
            std::fwrite(r.lg.data(), 4, (size_t) K, fl);
            std::fwrite(&r.lse, 4, 1, fs);
            std::fwrite(&r.tlp, 4, 1, ft);
        }
        std::fclose(fi);
        std::fclose(fl);
        std::fclose(fs);
        std::fclose(ft);
        FILE* fo = op(".out.i32");
        std::fwrite(out.data(), 4, out.size(), fo);
        std::fclose(fo);
        std::fprintf(sumf, "%s n_prompt=%d n_out=%zu rows=%zu eog=%d prompt_s=%.3f gen_s=%.3f\n", id.c_str(), np,
                     out.size(), rows.size(), is_eog ? 1 : 0, t1 - t0, t2 - t1);
        std::fflush(sumf);
        std::fprintf(stderr, "[nh_engine] %s n_prompt=%d n_out=%zu eog=%d prompt_s=%.1f gen_s=%.1f\n", id.c_str(), np,
                     out.size(), is_eog ? 1 : 0, t1 - t0, t2 - t1);
    }
    std::fclose(sumf);
    return 0;
}
