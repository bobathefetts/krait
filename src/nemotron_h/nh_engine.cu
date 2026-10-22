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
//   strata-nh-engine --serve --model SHARD [--batch N] [--ctx 32768] ...   (serve/server.py's engine protocol, see Serve)
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

#include "nh_cpuexact.cuh"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <random>
#include <cctype>
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
    if (b.size() % 4) throw std::runtime_error(p + ": size is not a multiple of 4 bytes");
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

// the MTP / NextN head (blk.88; nemotron-h-moe.cpp graph_mtp).  Dense tensors in VRAM, routed experts host-mapped
// (computed by ggml-cpu at draft batch sizes: drafts need not be exact, the verify step is)
struct MtpW {
    ggml_tensor *enorm = nullptr, *hnorm = nullptr, *eh_proj = nullptr, *attn_norm = nullptr, *wq = nullptr, *wk = nullptr,
                *wv = nullptr, *wo = nullptr, *post_attn_norm = nullptr, *gate_inp = nullptr, *exp_probs_b = nullptr,
                *latent_down = nullptr, *latent_up = nullptr, *up_exps = nullptr, *down_exps = nullptr, *up_shexp = nullptr,
                *down_shexp = nullptr, *shared_head_norm = nullptr;
};

struct Weights {
    MtpW mtp;
    bool has_mtp = false;
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
void load_weights(const nh::Model& m, int n_cpu_moe, ggml_backend_buffer_type_t buft_gpu, Weights& W, bool with_mtp = false) {
    W.g = m.g;
    const int nl = m.g.n_layer;
    const size_t n_t = 8 + (size_t) (nl + 1) * 24;
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
    if (with_mtp) {
        if ((int) m.layers.size() <= nl || m.layers[(size_t) nl].kind != nh::Kind::Mtp)
            throw std::runtime_error("nh_engine: --spec mtp needs the MTP block (blk." + std::to_string(nl) + ")");
        const nh::Layer& s = m.layers[(size_t) nl];
        MtpW& d = W.mtp;
        d.enorm = mk(s.enorm, false);
        d.hnorm = mk(s.hnorm, false);
        d.eh_proj = mk(s.eh_proj, false);
        d.attn_norm = mk(s.attn_norm, false);
        d.wq = mk(s.wq, false);
        d.wk = mk(s.wk, false);
        d.wv = mk(s.wv, false);
        d.wo = mk(s.wo, false);
        d.post_attn_norm = mk(s.post_attn_norm, false);
        d.gate_inp = mk(s.gate_inp, false);
        d.exp_probs_b = mk(s.exp_probs_b, false);
        d.latent_down = mk(s.latent_down, false);
        d.latent_up = mk(s.latent_up, false);
        d.up_exps = mk(s.up_exps, true);
        d.down_exps = mk(s.down_exps, true);
        d.up_shexp = mk(s.up_shexp, false);
        d.down_shexp = mk(s.down_shexp, false);
        d.shared_head_norm = mk(s.shared_head_norm, false);
        for (ggml_tensor* t : {d.enorm, d.hnorm, d.eh_proj, d.attn_norm, d.wq, d.wk, d.wv, d.wo, d.post_attn_norm, d.gate_inp,
                               d.exp_probs_b, d.latent_down, d.latent_up, d.up_exps, d.down_exps, d.up_shexp, d.down_shexp,
                               d.shared_head_norm})
            if (!t) throw std::runtime_error("nh_engine: MTP block tensor missing");
        W.has_mtp = true;
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
    ggml_tensor* h = nullptr;   // result_norm rows (the MTP head's input), when requested
    std::vector<ggml_tensor*> s_copy_r;   // rows mode: the recurrent-state index (slot) of each row
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

// ---- Phase 4 debugging: per-node, per-token hashes of every named F32 node, plain T == 1 steps vs one verify batch ----
struct DiffCtx {
    int mode = 0;   // 1 plain step (record one row per node), 2 verify batch (split each node into T token rows)
    int T = 1;
    std::map<std::string, std::vector<uint64_t>> P, V;
    std::map<std::string, std::string> shape_p, shape_v;
    std::vector<std::string> order;
    std::set<std::string> seen;   // names already met in the current graph
    std::vector<uint8_t> tmp;
};

static uint64_t fnv(const uint8_t* b, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

bool diff_cb(ggml_tensor* t, bool ask, void* ud) {
    DiffCtx& d = *static_cast<DiffCtx*>(ud);
    if (ask) return t->name[0] != 0;
    if (t->type != GGML_TYPE_F32 || ggml_is_empty(t) || !ggml_is_contiguous(t) || t->name[0] == 0) return true;
    std::string nm = t->name;
    while (d.seen.count(nm)) nm += "#";
    d.seen.insert(nm);
    const size_t nb = ggml_nbytes(t);
    d.tmp.resize(nb);
    ggml_backend_tensor_get(t, d.tmp.data(), 0, nb);
    char shp[96];
    std::snprintf(shp, sizeof shp, "%lld,%lld,%lld,%lld", (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
                  (long long) t->ne[3]);
    if (d.mode == 1) {
        if (!d.P.count(nm)) d.order.push_back(nm);
        d.P[nm].push_back(fnv(d.tmp.data(), nb));
        d.shape_p[nm] = shp;
    } else if (d.mode == 2) {
        d.shape_v[nm] = shp;
        auto& v = d.V[nm];
        v.clear();
        if (nb % (size_t) d.T) return true;
        const size_t n = nb / (size_t) d.T;
        for (int i = 0; i < d.T; ++i) v.push_back(fnv(d.tmp.data() + (size_t) i * n, n));
    }
    return true;
}

// ---- Phase 3: the routed-expert cache (decode, T == 1) ----
// Every MoE layer gets a VRAM slot pool with an expert -> slot map and LFU-with-decay scores.
//  * blocks < n_cpu_moe ("lo"): llama.cpp computes their experts on the CPU (ggml-cpu vec_dot) at T < 32.  Here three
//    CPU-side custom ops replace mul_mat_id/relu/sqr/mul_mat_id: PLAN (1 thread: hit/miss split, q8_K of the latent with
//    ggml's own quantize_row_q8_K, launch of the CPU-exact CUDA kernels of nh_cpuexact.cuh for the hits on a separate
//    stream), UP (all threads: miss rows with ggml-cpu's vec_dot, relu^2, ggml's quantize_row_q8_0) and DOWN (all
//    threads: miss rows; thread 0 then waits for the hit kernels and fills their columns).  Same values as the CPU path
//    bit for bit; afterwards a few misses are admitted into the pool by async H2D copies from the registered mmap.
//  * blocks >= n_cpu_moe ("hi"): llama.cpp keeps them resident and runs MMVQ.  Here a PLAN op (1 thread) uploads the
//    missing experts synchronously into the pool and returns slot ids, and the same mul_mat_id runs on the pool tensor,
//    so ggml-cuda computes exactly what it computes on the resident tensor.  2 <= T < 32 runs mul_mat_id on the host
//    tensor forced to the GPU (the scheduler copies the used experts, as op offload does for T >= 32).
struct SlotPool {
    int il = -1;
    bool lo = true;
    ggml_type tu = GGML_TYPE_COUNT, td = GGML_TYPE_COUNT;
    size_t up_row = 0, dn_row = 0, up_exp = 0, dn_exp = 0;
    const uint8_t* host_up = nullptr;   // expert 0 in the mmap
    const uint8_t* host_dn = nullptr;
    int cap = 0;
    ggml_tensor* t_up = nullptr;   // [n_in, n_out, cap] pool tensors (hi: used by mul_mat_id; lo: raw slots)
    ggml_tensor* t_dn = nullptr;
    std::vector<int> slot_of, exp_of;
    std::vector<cudaEvent_t> ready;    // lo: async upload done
    std::vector<uint8_t> pending;
    std::vector<int> pend_list;   // slots with an admission upload issued and not yet seen complete (review H2)
    std::vector<double> score;         // LFU with decay: score at step last[e]
    std::vector<int64_t> last;
    long hits = 0, misses = 0, uploads = 0, upload_skips = 0;
    double t_plan = 0, tm_up = 0, tm_dn = 0, t_wait = 0, t_hi_upload = 0;
    std::atomic<int> ctr_up{0}, ctr_dn{0};
};

struct CacheCtx {
    bool on = false, hi_on = false;
    int n_used = 22, n_in = 1024, n_ff = 2688;
    double decay = 0.97;               // per decode step
    int max_admit = 2;                 // per lo layer per step
    int64_t step = 0;
    cudaStream_t s_hit = nullptr, s_copy = nullptr, s_hi = nullptr;
    cudaEvent_t ev_hit = nullptr;
    // device scratch for the hit kernels (one layer at a time)
    uint8_t* d_args = nullptr;         // [up ptrs | dn ptrs | xi | col] for <= NU_MAX items
    uint8_t* d_q8k = nullptr;
    float* d_upo = nullptr;
    uint8_t* d_hq = nullptr;
    float* d_out = nullptr;
    uint8_t* h_args = nullptr;         // pinned
    uint8_t* h_q8k = nullptr;
    float* h_out = nullptr;
    std::vector<std::unique_ptr<SlotPool>> pools;   // by layer (nullptr for non-MoE)
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    size_t pool_bytes = 0;
    const ggml_type_traits_cpu* q8k_tr = nullptr;
    const ggml_type_traits_cpu* q80_tr = nullptr;
    // review H4/M1: custom ops run inside ggml's worker threads and must not throw; the first CUDA failure is recorded
    // here and the engine fails the request after graph compute (Engine::check_cache_error)
    std::atomic<int> err{0};
    char err_msg[256] = {0};
};

// record the first cache-path CUDA failure; returns true when the call succeeded
inline bool cache_ok(CacheCtx& C, cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    int z = 0;
    if (C.err.compare_exchange_strong(z, 1)) std::snprintf(C.err_msg, sizeof C.err_msg, "%s: %s", what, cudaGetErrorString(e));
    return false;
}

double eff_score(const SlotPool& p, int e, int64_t step, double decay) {
    return p.score[(size_t) e] * std::pow(decay, (double) (step - p.last[(size_t) e]));
}
void touch(SlotPool& p, int e, int64_t step, double decay) {
    p.score[(size_t) e] = eff_score(p, e, step, decay) + 1.0;
    p.last[(size_t) e] = step;
}

// Up to NH_TMAX tokens per cached step: T == 1 (decode) and the speculative verify batches (T <= NH_TMAX), where every
// token's experts are computed exactly as a T == 1 step computes them (per token q8_K, per token vec_dot).
constexpr int NH_TMAX = 4;
constexpr int NU_MAX = 22 * NH_TMAX;
// plan layout (I8 tensor, bytes): int32 n_miss, n_hit, n_tok, miss_k[NU_MAX], hit_k[NU_MAX], hit_slot[NU_MAX], then the
// q8_K rows of the T latent columns (item k = t * 22 + j: token t, selection slot j)
constexpr size_t PLAN_Q8K_OFF = 4 * (3 + 3 * NU_MAX);
constexpr size_t PLAN_Q8K_ROW = 4 * nhx::Q8K_BYTES;   // n_in = 1024: 4 q8_K blocks per token
constexpr size_t PLAN_BYTES = PLAN_Q8K_OFF + NH_TMAX * PLAN_Q8K_ROW;
constexpr size_t ARGS_BYTES = 2 * NU_MAX * sizeof(void*) + 2 * NU_MAX * sizeof(int);

CacheCtx* g_cache = nullptr;

inline int sel_at(const ggml_tensor* sel, int k, int n_used) {
    const int t = k / n_used, j = k % n_used;
    return *(const int32_t*) ((const char*) sel->data + (size_t) j * sel->nb[0] + (size_t) t * sel->nb[1]);
}

void launch_hits(CacheCtx& C, SlotPool& p, int n_hit) {
    const nhx::UpT ut = p.tu == GGML_TYPE_Q4_K ? nhx::UpT::Q4K : p.tu == GGML_TYPE_Q5_K ? nhx::UpT::Q5K : nhx::UpT::Q6K;
    const nhx::DownT dt = p.td == GGML_TYPE_Q5_0 ? nhx::DownT::Q50 : nhx::DownT::IQ4NL;
    const uint8_t* const* wup = (const uint8_t* const*) C.d_args;
    const uint8_t* const* wdn = wup + NU_MAX;
    const int* xi = (const int*) (C.d_args + 2 * NU_MAX * sizeof(void*));
    const int* col = xi + NU_MAX;
    const int F = C.n_ff, D = C.n_in;
    dim3 gu((unsigned) (F / 32), (unsigned) n_hit), gq((unsigned) ((F / 32 + 7) / 8), (unsigned) n_hit), gd((unsigned) (D / 32), (unsigned) n_hit);
    switch (ut) {
    case nhx::UpT::Q4K: nhx::k_up<nhx::UpT::Q4K><<<gu, 256, 0, C.s_hit>>>(wup, C.d_q8k, F, D / 256, p.up_row, C.d_upo, xi); break;
    case nhx::UpT::Q5K: nhx::k_up<nhx::UpT::Q5K><<<gu, 256, 0, C.s_hit>>>(wup, C.d_q8k, F, D / 256, p.up_row, C.d_upo, xi); break;
    default: nhx::k_up<nhx::UpT::Q6K><<<gu, 256, 0, C.s_hit>>>(wup, C.d_q8k, F, D / 256, p.up_row, C.d_upo, xi); break;
    }
    nhx::k_relu2_q8_0<<<gq, 256, 0, C.s_hit>>>(C.d_upo, F, C.d_hq);
    if (dt == nhx::DownT::Q50) nhx::k_down<nhx::DownT::Q50><<<gd, 256, 0, C.s_hit>>>(wdn, C.d_hq, D, F / 32, p.dn_row, col, C.d_out);
    else nhx::k_down<nhx::DownT::IQ4NL><<<gd, 256, 0, C.s_hit>>>(wdn, C.d_hq, D, F / 32, p.dn_row, col, C.d_out);
    cache_ok(C, cudaGetLastError(), "hit kernel launch");
}

// lo PLAN: src0 = latent [n_in, 1, T] f32, src1 = selected [22, T] i32 (a view); dst = plan bytes
void op_plan_lo(ggml_tensor* dst, int ith, int nth, void* ud) {
    (void) nth;
    if (ith != 0) return;
    const double t0 = now_s();
    SlotPool& p = *static_cast<SlotPool*>(ud);
    CacheCtx& C = *g_cache;
    const ggml_tensor* lat = dst->src[0];
    const ggml_tensor* sel = dst->src[1];
    const int T = (int) lat->ne[2];
    uint8_t* plan = (uint8_t*) dst->data;
    int32_t* pi = (int32_t*) plan;
    int32_t *miss_k = pi + 3, *hit_k = pi + 3 + NU_MAX, *hit_slot = pi + 3 + 2 * NU_MAX;
    for (int t = 0; t < T; ++t)
        C.q8k_tr->from_float((const float*) ((const char*) lat->data + (size_t) t * lat->nb[2]),
                             plan + PLAN_Q8K_OFF + (size_t) t * PLAN_Q8K_ROW, C.n_in);
    int n_miss = 0, n_hit = 0;
    for (int k = 0; k < C.n_used * T; ++k) {
        const int e = sel_at(sel, k, C.n_used);
        const int s = p.slot_of[(size_t) e];
        bool ok = s >= 0;
        if (ok && p.pending[(size_t) s]) {
            if (cudaEventQuery(p.ready[(size_t) s]) == cudaSuccess) p.pending[(size_t) s] = 0;
            else ok = false;
        }
        if (ok) {
            hit_k[n_hit] = k;
            hit_slot[n_hit++] = s;
        } else {
            miss_k[n_miss++] = k;
        }
    }
    pi[0] = n_miss;
    pi[1] = n_hit;
    pi[2] = T;
    p.hits += n_hit;
    p.misses += n_miss;
    p.ctr_up.store(0);
    p.ctr_dn.store(0);
    if (n_hit) {
        const uint8_t** hu = (const uint8_t**) C.h_args;
        const uint8_t** hd = hu + NU_MAX;
        int* xi = (int*) (C.h_args + 2 * NU_MAX * sizeof(void*));
        int* col = xi + NU_MAX;
        for (int h = 0; h < n_hit; ++h) {
            hu[h] = (const uint8_t*) p.t_up->data + (size_t) hit_slot[h] * p.up_exp;
            hd[h] = (const uint8_t*) p.t_dn->data + (size_t) hit_slot[h] * p.dn_exp;
            xi[h] = hit_k[h] / C.n_used;
            col[h] = h;
        }
        std::memcpy(C.h_q8k, plan + PLAN_Q8K_OFF, (size_t) T * PLAN_Q8K_ROW);
        cache_ok(C, cudaMemcpyAsync(C.d_args, C.h_args, ARGS_BYTES, cudaMemcpyHostToDevice, C.s_hit), "hit args upload");
        cache_ok(C, cudaMemcpyAsync(C.d_q8k, C.h_q8k, (size_t) T * PLAN_Q8K_ROW, cudaMemcpyHostToDevice, C.s_hit), "hit q8_K upload");
        launch_hits(C, p, n_hit);
        cache_ok(C, cudaMemcpyAsync(C.h_out, C.d_out, (size_t) n_hit * C.n_in * sizeof(float), cudaMemcpyDeviceToHost, C.s_hit),
                 "hit result download");
        cache_ok(C, cudaEventRecord(C.ev_hit, C.s_hit), "hit event");
    }
    p.t_plan += now_s() - t0;
}

// lo UP: src0 = plan, src1 = selected; dst = [34*n_ff/32, NU_MAX] I8: q8_0(relu^2(up)) of each miss (by miss index)
void op_up_lo(ggml_tensor* dst, int ith, int nth, void* ud) {
    (void) nth;
    const double t0 = now_s();
    SlotPool& p = *static_cast<SlotPool*>(ud);
    CacheCtx& C = *g_cache;
    const uint8_t* plan = (const uint8_t*) dst->src[0]->data;
    const int32_t* pi = (const int32_t*) plan;
    const ggml_tensor* sel = dst->src[1];
    const int n_miss = pi[0];
    const int nblk = C.n_ff / 32;
    const auto* tr = ggml_get_type_traits_cpu(p.tu);
    float h[32];
    for (;;) {
        const int u = p.ctr_up.fetch_add(1);
        if (u >= n_miss * nblk) break;
        const int m = u / nblk, b = u % nblk;
        const int k = pi[3 + m];
        const int e = sel_at(sel, k, C.n_used);
        const uint8_t* q8k = plan + PLAN_Q8K_OFF + (size_t) (k / C.n_used) * PLAN_Q8K_ROW;
        const uint8_t* base = p.host_up + (size_t) e * p.up_exp + (size_t) b * 32 * p.up_row;
        for (int r = 0; r < 32; ++r) {
            float v;
            tr->vec_dot(C.n_in, &v, 0, base + (size_t) r * p.up_row, 0, q8k, 0, 1);
            v = v > 0.f ? v : 0.f;
            h[r] = v * v;
        }
        C.q80_tr->from_float(h, (uint8_t*) dst->data + (size_t) m * nblk * nhx::Q80_BYTES + (size_t) b * nhx::Q80_BYTES, 32);
    }
    if (ith == 0) p.tm_up += now_s() - t0;
}

void admit_lo(CacheCtx& C, SlotPool& p, const ggml_tensor* sel, const int32_t* pi) {
    const int n_miss = pi[0], n_hit = pi[1], T = pi[2];
    // review H2: clear finished uploads here too, not only when the expert is selected again (else such slots can
    // never be victims and the pool freezes).  Only the few slots uploaded recently are queried.
    for (size_t i = 0; i < p.pend_list.size();) {
        const int s = p.pend_list[i];
        if (!p.pending[(size_t) s] || cudaEventQuery(p.ready[(size_t) s]) == cudaSuccess) {
            p.pending[(size_t) s] = 0;
            p.pend_list[i] = p.pend_list.back();
            p.pend_list.pop_back();
        } else {
            ++i;
        }
    }
    std::vector<uint8_t> in_use((size_t) p.cap, 0);
    for (int h = 0; h < n_hit; ++h) in_use[(size_t) pi[3 + 2 * NU_MAX + h]] = 1;
    for (int k = 0; k < C.n_used * T; ++k) touch(p, sel_at(sel, k, C.n_used), C.step, C.decay);
    // candidates: misses by score, best first
    std::vector<std::pair<double, int>> cand;
    for (int m = 0; m < n_miss; ++m) {
        const int e = sel_at(sel, pi[3 + m], C.n_used);
        if (p.slot_of[(size_t) e] >= 0) continue;   // pending upload
        bool dup = false;
        for (auto& c : cand) dup = dup || c.second == e;
        if (!dup) cand.push_back({eff_score(p, e, C.step, C.decay), e});
    }
    std::sort(cand.rbegin(), cand.rend());
    int admitted = 0;
    for (auto& ce : cand) {
        if (admitted >= C.max_admit) {
            p.upload_skips++;
            continue;
        }
        int victim = -1;
        double vs = 1e300;
        for (int s = 0; s < p.cap; ++s) {
            if (in_use[(size_t) s] || p.pending[(size_t) s]) continue;
            const int ve = p.exp_of[(size_t) s];
            const double sc = ve < 0 ? -1.0 : eff_score(p, ve, C.step, C.decay);
            if (sc < vs) {
                vs = sc;
                victim = s;
            }
        }
        if (victim < 0 || vs >= ce.first) break;
        const int e = ce.second;
        if (p.exp_of[(size_t) victim] >= 0) p.slot_of[(size_t) p.exp_of[(size_t) victim]] = -1;
        p.exp_of[(size_t) victim] = e;
        p.slot_of[(size_t) e] = victim;
        const bool ok =
            cache_ok(C, cudaMemcpyAsync((uint8_t*) p.t_up->data + (size_t) victim * p.up_exp, p.host_up + (size_t) e * p.up_exp, p.up_exp,
                                        cudaMemcpyHostToDevice, C.s_copy), "admission upload") &&
            cache_ok(C, cudaMemcpyAsync((uint8_t*) p.t_dn->data + (size_t) victim * p.dn_exp, p.host_dn + (size_t) e * p.dn_exp, p.dn_exp,
                                        cudaMemcpyHostToDevice, C.s_copy), "admission upload") &&
            cache_ok(C, cudaEventRecord(p.ready[(size_t) victim], C.s_copy), "admission event");
        if (!ok) {
            // the slot holds nothing usable: forget the mapping (the request fails through the error flag)
            p.slot_of[(size_t) e] = -1;
            p.exp_of[(size_t) victim] = -1;
            break;
        }
        p.pending[(size_t) victim] = 1;
        p.pend_list.push_back(victim);
        in_use[(size_t) victim] = 1;
        p.uploads++;
        admitted++;
    }
}

// lo DOWN: src0 = UP's q8_0 bytes, src1 = plan, src2 = selected; dst = [n_in, 22, T] f32 (llama's ffn_moe_down)
void op_down_lo(ggml_tensor* dst, int ith, int nth, void* ud) {
    (void) nth;
    const double t0 = now_s();
    SlotPool& p = *static_cast<SlotPool*>(ud);
    CacheCtx& C = *g_cache;
    const uint8_t* hq = (const uint8_t*) dst->src[0]->data;
    const int32_t* pi = (const int32_t*) dst->src[1]->data;
    const ggml_tensor* sel = dst->src[2];
    const int n_miss = pi[0], n_hit = pi[1];
    const int nblk = C.n_ff / 32, rows = 64, nch = C.n_in / rows;
    const auto* tr = ggml_get_type_traits_cpu(p.td);
    float* out = (float*) dst->data;
    for (;;) {
        const int u = p.ctr_dn.fetch_add(1);
        if (u >= n_miss * nch) break;
        const int m = u / nch, c = u % nch;
        const int k = pi[3 + m];
        const int e = sel_at(sel, k, C.n_used);
        const uint8_t* base = p.host_dn + (size_t) e * p.dn_exp;
        const uint8_t* y = hq + (size_t) m * nblk * nhx::Q80_BYTES;
        for (int r = c * rows; r < (c + 1) * rows; ++r)
            tr->vec_dot(C.n_ff, &out[(size_t) k * C.n_in + r], 0, base + (size_t) r * p.dn_row, 0, y, 0, 1);
    }
    if (ith == 0) {
        const double t1 = now_s();
        if (n_hit && cache_ok(C, cudaEventSynchronize(C.ev_hit), "hit wait")) {
            for (int h = 0; h < n_hit; ++h)
                std::memcpy(out + (size_t) pi[3 + NU_MAX + h] * C.n_in, C.h_out + (size_t) h * C.n_in, sizeof(float) * C.n_in);
        }
        const double t2 = now_s();
        admit_lo(C, p, sel, pi);
        p.tm_dn += t1 - t0;
        p.t_wait += t2 - t1;
    }
}

// hi PLAN (T <= NH_TMAX): src0 = selected [22, T]; dst = slot ids [22, T] i32 for mul_mat_id on the pool tensors.  Every
// expert of the T tokens is resident when it returns (missing ones uploaded synchronously from pinned memory).
void op_plan_hi(ggml_tensor* dst, int ith, int nth, void* ud) {
    (void) nth;
    if (ith != 0) return;
    const double t0 = now_s();
    SlotPool& p = *static_cast<SlotPool*>(ud);
    CacheCtx& C = *g_cache;
    const ggml_tensor* sel = dst->src[0];
    const int T = (int) sel->ne[1], NK = C.n_used * T;
    int32_t* ids = (int32_t*) dst->data;
    std::vector<int> es((size_t) NK);
    std::vector<uint8_t> in_use((size_t) p.cap, 0);
    for (int k = 0; k < NK; ++k) {
        es[(size_t) k] = sel_at(sel, k, C.n_used);
        touch(p, es[(size_t) k], C.step, C.decay);
        const int s = p.slot_of[(size_t) es[(size_t) k]];
        if (s >= 0) in_use[(size_t) s] = 1;
    }
    bool up = false;
    for (int k = 0; k < NK; ++k) {
        const int e = es[(size_t) k];
        int s = p.slot_of[(size_t) e];
        if (s >= 0) {
            p.hits++;
        } else {
            p.misses++;
            double vs = 1e300;
            for (int j = 0; j < p.cap; ++j) {
                if (in_use[(size_t) j]) continue;
                const int ve = p.exp_of[(size_t) j];
                const double sc = ve < 0 ? -1.0 : eff_score(p, ve, C.step, C.decay);
                if (sc < vs) {
                    vs = sc;
                    s = j;
                }
            }
            if (s < 0) {
                // unreachable with init's cap >= n_used * NH_TMAX check; never throw inside a ggml op (review M1)
                int z = 0;
                if (C.err.compare_exchange_strong(z, 1)) std::snprintf(C.err_msg, sizeof C.err_msg, "hi pool smaller than one step's experts");
                for (int kk = 0; kk < NK; ++kk)
                    *(int32_t*) ((char*) ids + (size_t) (kk % C.n_used) * dst->nb[0] + (size_t) (kk / C.n_used) * dst->nb[1]) = 0;
                return;
            }
            if (p.exp_of[(size_t) s] >= 0) p.slot_of[(size_t) p.exp_of[(size_t) s]] = -1;
            p.exp_of[(size_t) s] = e;
            p.slot_of[(size_t) e] = s;
            in_use[(size_t) s] = 1;
            cache_ok(C, cudaMemcpyAsync((uint8_t*) p.t_up->data + (size_t) s * p.up_exp, p.host_up + (size_t) e * p.up_exp, p.up_exp,
                                        cudaMemcpyHostToDevice, C.s_hi), "hi upload");
            cache_ok(C, cudaMemcpyAsync((uint8_t*) p.t_dn->data + (size_t) s * p.dn_exp, p.host_dn + (size_t) e * p.dn_exp, p.dn_exp,
                                        cudaMemcpyHostToDevice, C.s_hi), "hi upload");
            p.uploads++;
            up = true;
        }
        *(int32_t*) ((char*) ids + (size_t) (k % C.n_used) * dst->nb[0] + (size_t) (k / C.n_used) * dst->nb[1]) = s;
    }
    const double t1 = now_s();
    if (up) cache_ok(C, cudaStreamSynchronize(C.s_hi), "hi upload wait");
    p.t_hi_upload += now_s() - t1;
    p.t_plan += t1 - t0;
}

// ---- --profile: every node computed on its own (sched eval callback asks for all), wall time per node aggregated by
// (backend, op, role); used to see where a decode step goes, never for timing runs ----
struct ProfCtx {
    bool on = false;
    ggml_backend_sched_t sched = nullptr;
    double t_last = 0;
    std::map<std::string, std::pair<double, long>> agg;
};

std::string prof_role(const char* name) {
    std::string n(name);
    const size_t d = n.rfind('-');
    if (d != std::string::npos && d + 1 < n.size() && std::isdigit((unsigned char) n[d + 1])) n = n.substr(0, d);
    return n;
}

bool prof_cb(ggml_tensor* t, bool ask, void* ud) {
    ProfCtx& p = *static_cast<ProfCtx*>(ud);
    if (ask) return true;
    const double t1 = now_s();
    ggml_backend_t b = ggml_backend_sched_get_tensor_backend(p.sched, t);
    const std::string key = std::string(b ? ggml_backend_name(b) : "?") + "\t" + ggml_op_desc(t) + "\t" + prof_role(t->name);
    auto& e = p.agg[key];
    e.first += t1 - p.t_last;
    e.second += 1;
    p.t_last = now_s();
    return true;
}

struct Engine {
    Weights W;
    Memory M;
    ggml_backend_t gpu = nullptr, cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    CopyExperts cx;
    DumpCtx dump;
    DiffCtx diff;
    ProfCtx prof;
    CacheCtx C;
    // speculative decoding: verify batches (T <= NH_TMAX tokens computed with T == 1 numerics), rollback snapshots
    bool verify = false, want_h = false;
    int64_t vpos0 = 0;
    // rows mode (--batch decode windows): T rows, row r = the next token of session row_slot[r] at position row_pos[r];
    // every row is computed as that session's own T == 1 step computes it (per-row Mamba-2 state and attention, the
    // verify paths of the MoE), so a batched answer is token-for-token the answer of the request alone
    bool rows_mode = false;
    int rows_max = 1;               // most rows a window can have (the batch slots)
    std::vector<int> row_slot;
    std::vector<int64_t> row_pos;
    int k_snap = 1;                 // recurrent-state rows per slot: 1 + rollback snapshots
    std::vector<float> hrows;       // result_norm rows of the last decode (want_h)
    // MTP draft head
    bool mtp_on = false;
    int mtp_topk = 0;               // experts per draft token (0: the model's n_expert_used)
    ggml_tensor *mtp_k = nullptr, *mtp_v = nullptr;
    ggml_context* mtp_ctx = nullptr;
    ggml_backend_buffer_t mtp_buf = nullptr;
    std::vector<float> mtp_logits, mtp_h;   // last row of the last MTP call
    int64_t kv_len(int64_t n) const { return std::min<int64_t>(M.n_ctx, std::max<int64_t>(256, ((n + 255) / 256) * 256)); }
    int n_cpu_moe = 68;      // numerics split: experts of blocks < n_cpu_moe follow llama.cpp's CPU path
    bool pin = false;
    long cache_mib = -1;     // -1: auto (budget from free VRAM, see init_cache)
    long vram_reserve_mib = 2048;   // review H3: VRAM left free after the expert pools
    long vram_limit_mib = 0;        // optional cap on the process's total VRAM use (0: none)
    std::vector<std::pair<uintptr_t, size_t>> pinned_ranges;   // cudaHostRegister'ed ranges (unregistered at teardown)
    int cap_lo = 0, cap_hi = 0;
    size_t pinned_bytes = 0;
    int n_threads = 16;
    bool force_norm = true;   // llama-context.cpp graph_get_cb: "norm" pinned to the layer's device (full offload)
    int max_nodes = 32768;
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
                M.r[(size_t) il] = ggml_new_tensor_2d(M.ctx, GGML_TYPE_F32, n_embd_r(), (int64_t) n_slots * k_snap);
                M.s[(size_t) il] = ggml_new_tensor_2d(M.ctx, GGML_TYPE_F32, n_embd_s(), (int64_t) n_slots * k_snap);
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
        if (rows_mode) return build_mamba2_rows(cur, il, T, gi);
        const int64_t n_seqs = 1, n_seq_tokens = T, K = verify ? T : 1;   // verify: T rollback snapshots
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

    // rows mode: ssm_in / ssm_out / the group norm over all rows at once (row-wise ops: MMVQ for ncols <= 4 and the
    // norms reduce each row on its own), the recurrent part (conv state, ssm_conv, ssm_scan, D, gate) per row on its own
    // slot's state, built exactly as build_mamba2 builds a T == 1 step
    ggml_tensor* build_mamba2_rows(ggml_tensor* cur, int il, int64_t T, GraphIn& gi) {
        const LayerW& l = W.L[(size_t) il];
        const nh::Geometry& g = W.g;
        const int64_t d_conv = g.ssm_d_conv, d_inner = g.ssm_d_inner, d_state = g.ssm_d_state, n_head = g.ssm_n_head;
        const int64_t head_dim = d_inner / n_head, n_group = g.ssm_n_group;
        const int64_t n_seqs = 1, n_seq_tokens = 1, K = 1;
        const int64_t mem_size = M.n_slots, n_rs = 1;
        ggml_tensor* conv_states_all = M.r[(size_t) il];
        ggml_tensor* ssm_states_all = M.s[(size_t) il];
        const int64_t state_slots = ssm_states_all->ne[1];
        ggml_tensor* zx_all = ggml_mul_mat(ctx0, l.ssm_in, cur);
        ggml_tensor* yall = nullptr;
        for (int64_t r = 0; r < T; ++r) {
            const int64_t kv_head = row_slot[(size_t) r];
            ggml_tensor* sc = gi.s_copy_r[(size_t) r];
            ggml_tensor* sc_main = ggml_view_1d(ctx0, sc, n_seqs, 0);
            ggml_tensor* sc_tail = ggml_view_1d(ctx0, sc, n_rs - 1, sc->nb[0]);
            ggml_tensor* conv = build_rs(conv_states_all, sc, sc_main, n_embd_r(), n_seqs, n_rs, kv_head, -1, nullptr);
            conv = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner + 2 * n_group * d_state, n_seqs);
            ggml_tensor* zxBCdt = ggml_view_3d(ctx0, zx_all, zx_all->ne[0], n_seq_tokens, n_seqs, zx_all->nb[1], zx_all->nb[1],
                                               (size_t) r * zx_all->nb[1]);
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
                ggml_tensor* last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner + 2 * n_group * d_state, n_seqs,
                                                      conv_x->nb[1], conv_x->nb[2], n_seq_tokens * conv_x->nb[0]);
                ggml_build_forward_expand(
                    gf, ggml_cpy(ctx0, last_conv,
                                 ggml_view_2d(ctx0, conv_states_all, row_count, n_seqs, conv_states_all->nb[1],
                                              (size_t) kv_head * row_size)));
                xBC = ggml_ssm_conv(ctx0, conv_x, l.ssm_conv1d);
                xBC = ggml_add(ctx0, xBC, l.ssm_conv1d_b);
                xBC = ggml_silu(ctx0, xBC);
            }
            ggml_tensor* x = ggml_view_4d(ctx0, xBC, head_dim, n_head, n_seq_tokens, n_seqs, head_dim * xBC->nb[0],
                                          xBC->nb[1], xBC->nb[2], 0);
            ggml_tensor* B = ggml_view_4d(ctx0, xBC, d_state, n_group, n_seq_tokens, n_seqs, d_state * xBC->nb[0],
                                          xBC->nb[1], xBC->nb[2], d_inner * ggml_element_size(xBC));
            ggml_tensor* C_ = ggml_view_4d(ctx0, xBC, d_state, n_group, n_seq_tokens, n_seqs, d_state * xBC->nb[0],
                                           xBC->nb[1], xBC->nb[2], (d_inner + n_group * d_state) * ggml_element_size(xBC));
            dt = ggml_add(ctx0, ggml_cont(ctx0, dt), l.ssm_dt_b);
            ggml_tensor* A = l.ssm_a;
            auto get_ssm_rows = [&](ggml_context* ctx, ggml_tensor* states, ggml_tensor* ids) {
                ggml_tensor* ssm = ggml_reshape_4d(ctx, states, d_state, head_dim, n_head, state_slots);
                return ggml_ssm_scan(ctx, ssm, x, dt, A, B, C_, ids, K);
            };
            ggml_tensor* y_ssm = build_rs(ssm_states_all, sc_tail, sc_main, n_embd_s(), n_seqs, n_rs, kv_head, -1, get_ssm_rows);
            {
                const int64_t D = d_state * d_inner;
                const size_t row_size = ggml_row_size(ssm_states_all->type, D);
                const size_t y_row_size = ggml_row_size(y_ssm->type, D);
                const size_t state_offset = ggml_nelements(x) * ggml_element_size(x);
                ggml_build_forward_expand(
                    gf, ggml_cpy(ctx0, ggml_view_3d(ctx0, y_ssm, D, n_seqs, 1, y_row_size, y_row_size * n_seqs, state_offset),
                                 ggml_view_3d(ctx0, ssm_states_all, D, n_seqs, 1, ssm_states_all->nb[1],
                                              (size_t) mem_size * row_size, kv_head * row_size)));
            }
            ggml_tensor* y = ggml_view_4d(ctx0, y_ssm, head_dim, n_head, n_seq_tokens, n_seqs, x->nb[1], n_head * x->nb[1],
                                          n_seq_tokens * n_head * x->nb[1], 0);
            y = ggml_add(ctx0, y, ggml_mul(ctx0, x, l.ssm_d));
            y = ggml_swiglu_split(ctx0, ggml_cont(ctx0, z), y);
            y = ggml_reshape_2d(ctx0, y, d_inner, 1);
            yall = r == 0 ? y : ggml_concat(ctx0, yall, y, 1);
        }
        ggml_format_name(yall, "mamba2_rows_y-%d", il);
        ggml_tensor* y = ggml_reshape_4d(ctx0, yall, d_inner / n_group, n_group, T, 1);
        y = build_norm(y, l.ssm_norm, il);
        y = ggml_reshape_2d(ctx0, y, d_inner, T);
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
        ggml_tensor* o = nullptr;
        if (rows_mode) {
            // one flash attention per row over its own slot's cells, with the KV length and mask row of that session's
            // T == 1 step
            for (int64_t t = 0; t < T; ++t) {
                const int64_t sl = row_slot[(size_t) t];
                const int64_t nkv_t = kv_len(row_pos[(size_t) t] + 1);
                ggml_tensor* kt = ggml_view_4d(ctx0, kc, hd, n_head_kv, nkv_t, 1, ggml_row_size(kc->type, hd),
                                               ggml_row_size(kc->type, n_embd_gqa), ggml_row_size(kc->type, n_embd_gqa * kv_size),
                                               ggml_row_size(kc->type, n_embd_gqa * kv_size) * (size_t) sl);
                ggml_tensor* vt = ggml_view_4d(ctx0, vc, hd, n_head_kv, nkv_t, 1, ggml_row_size(vc->type, hd),
                                               ggml_row_size(vc->type, n_embd_gqa), ggml_row_size(vc->type, n_embd_gqa * kv_size),
                                               ggml_row_size(vc->type, n_embd_gqa * kv_size) * (size_t) sl);
                kt = ggml_permute(ctx0, kt, 0, 2, 1, 3);
                vt = ggml_permute(ctx0, vt, 0, 2, 1, 3);
                ggml_tensor* qt = ggml_view_4d(ctx0, q, q->ne[0], 1, q->ne[2], q->ne[3], q->nb[1], q->nb[2], q->nb[3], (size_t) t * q->nb[1]);
                ggml_tensor* mt = ggml_view_4d(ctx0, gi.kq_mask, nkv_t, 1, 1, 1, gi.kq_mask->nb[1], gi.kq_mask->nb[2], gi.kq_mask->nb[3],
                                               (size_t) t * gi.kq_mask->nb[1]);
                ggml_tensor* ot = ggml_flash_attn_ext(ctx0, qt, kt, vt, mt, kq_scale, 0.0f, 0.0f);
                ggml_flash_attn_ext_add_sinks(ot, nullptr);
                ggml_flash_attn_ext_set_n_kv_max(ot, 0);
                ggml_prec_set_acc(ot, GGML_PREC_F32);
                ot = ggml_reshape_2d(ctx0, ot, ot->ne[0] * ot->ne[1], ot->ne[2] * ot->ne[3]);
                o = t == 0 ? ot : ggml_concat(ctx0, o, ot, 1);
            }
        } else if (!verify || T == 1) {
            o = ggml_flash_attn_ext(ctx0, q, k, v, gi.kq_mask, kq_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_add_sinks(o, nullptr);
            ggml_flash_attn_ext_set_n_kv_max(o, 0);
            ggml_prec_set_acc(o, GGML_PREC_F32);
            o = ggml_reshape_2d(ctx0, o, o->ne[0] * o->ne[1], o->ne[2] * o->ne[3]);
        } else {
            // verify batch: one flash attention per token with the KV length and mask row a T == 1 step at that position
            // uses (the kernel choice and the KV split depend on the query count and on n_kv)
            for (int64_t t = 0; t < T; ++t) {
                const int64_t nkv_t = kv_len(vpos0 + t + 1);
                ggml_tensor* qt = ggml_view_4d(ctx0, q, q->ne[0], 1, q->ne[2], q->ne[3], q->nb[1], q->nb[2], q->nb[3], (size_t) t * q->nb[1]);
                ggml_tensor* kt = ggml_view_4d(ctx0, k, k->ne[0], nkv_t, k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);
                ggml_tensor* vt = ggml_view_4d(ctx0, v, v->ne[0], nkv_t, v->ne[2], v->ne[3], v->nb[1], v->nb[2], v->nb[3], 0);
                ggml_tensor* mt = ggml_view_4d(ctx0, gi.kq_mask, nkv_t, 1, 1, 1, gi.kq_mask->nb[1], gi.kq_mask->nb[2], gi.kq_mask->nb[3],
                                               (size_t) t * gi.kq_mask->nb[1]);
                ggml_tensor* ot = ggml_flash_attn_ext(ctx0, qt, kt, vt, mt, kq_scale, 0.0f, 0.0f);
                ggml_flash_attn_ext_add_sinks(ot, nullptr);
                ggml_flash_attn_ext_set_n_kv_max(ot, 0);
                ggml_prec_set_acc(ot, GGML_PREC_F32);
                ot = ggml_reshape_2d(ctx0, ot, ot->ne[0] * ot->ne[1], ot->ne[2] * ot->ne[3]);
                o = t == 0 ? ot : ggml_concat(ctx0, o, ot, 1);
            }
        }
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
        ggml_tensor* router_logits = nullptr;
        if ((verify || rows_mode) && cur->ne[1] > 3) {
            // F32 router: MMVF serves ne11 <= 3 on sm_120 (mmvf.cu ggml_cuda_should_use_mmvf); keep T == 1 numerics per token
            for (int64_t t = 0; t < cur->ne[1]; ++t) {
                ggml_tensor* ct = ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], (size_t) t * cur->nb[1]);
                ggml_tensor* lt = ggml_mul_mat(ctx0, l.gate_inp, ct);
                router_logits = router_logits ? ggml_concat(ctx0, router_logits, lt, 1) : lt;
            }
        } else {
            router_logits = ggml_mul_mat(ctx0, l.gate_inp, cur);
        }
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
        ggml_tensor* experts = nullptr;
        SlotPool* pool = C.on ? C.pools[(size_t) il].get() : nullptr;
        const bool lo = il < n_cpu_moe;
        const bool cached_step = n_tokens == 1 || ((verify || rows_mode) && n_tokens <= NH_TMAX);
        if (pool && cached_step && lo) {
            // the CPU path of llama.cpp (ffn_moe_up / relu / sqr / ffn_moe_down on ggml-cpu), split into cached hits
            // (CPU-exact CUDA kernels) and misses (ggml-cpu vec_dot), see SlotPool
            ggml_tensor* a0[2] = {cur, selected};
            ggml_tensor* plan = ggml_custom_4d(ctx0, GGML_TYPE_I8, (int64_t) PLAN_BYTES, 1, 1, 1, a0, 2, op_plan_lo, 1, pool);
            ggml_format_name(plan, "nhc_plan-%d", il);
            ggml_backend_sched_set_tensor_backend(sched, plan, cpu);
            ggml_tensor* a1[2] = {plan, selected};
            const int64_t n_ff = l.up_exps->ne[1];
            ggml_tensor* hq = ggml_custom_4d(ctx0, GGML_TYPE_I8, n_ff / 32 * (int64_t) nhx::Q80_BYTES, n_used * n_tokens, 1, 1, a1, 2,
                                             op_up_lo, GGML_N_TASKS_MAX, pool);
            ggml_format_name(hq, "nhc_up-%d", il);
            ggml_backend_sched_set_tensor_backend(sched, hq, cpu);
            ggml_tensor* a2[3] = {hq, plan, selected};
            experts = ggml_custom_4d(ctx0, GGML_TYPE_F32, n_embd, n_used, n_tokens, 1, a2, 3, op_down_lo, GGML_N_TASKS_MAX, pool);
            ggml_backend_sched_set_tensor_backend(sched, experts, cpu);
            ggml_format_name(experts, "ffn_moe_down-%d", il);
        } else if (pool && cached_step) {
            // resident-numerics block: the same mul_mat_id on the slot pool, ids remapped by the plan op.  A verify batch
            // runs it once per token (the T == 1 kernel: ggml-cuda's multi-token MoE kernel reduces differently)
            ggml_tensor* a0[1] = {selected};
            ggml_tensor* ids = ggml_custom_4d(ctx0, GGML_TYPE_I32, n_used, n_tokens, 1, 1, a0, 1, op_plan_hi, 1, pool);
            ggml_format_name(ids, "nhc_ids-%d", il);
            ggml_backend_sched_set_tensor_backend(sched, ids, cpu);
            for (int64_t t = 0; t < n_tokens; ++t) {
                ggml_tensor* ct = n_tokens == 1 ? cur : ggml_view_3d(ctx0, cur, n_embd, 1, 1, cur->nb[1], cur->nb[2], (size_t) t * cur->nb[2]);
                ggml_tensor* it = n_tokens == 1 ? ids : ggml_view_2d(ctx0, ids, n_used, 1, ids->nb[1], (size_t) t * ids->nb[1]);
                ggml_tensor* up = ggml_mul_mat_id(ctx0, pool->t_up, ct, it);
                ggml_format_name(up, "ffn_moe_up-%d", il);
                ggml_tensor* h = ggml_relu(ctx0, up);
                h = ggml_sqr(ctx0, h);
                ggml_format_name(h, "ffn_moe_relu_sqr-%d", il);
                ggml_tensor* dn = ggml_mul_mat_id(ctx0, pool->t_dn, h, it);
                ggml_format_name(dn, "ffn_moe_down-%d", il);
                experts = t == 0 ? dn : ggml_concat(ctx0, experts, dn, 2);
            }
        } else {
            ggml_tensor* up = ggml_mul_mat_id(ctx0, l.up_exps, cur, selected);
            ggml_format_name(up, "ffn_moe_up-%d", il);
            cur = ggml_relu(ctx0, up);
            cur = ggml_sqr(ctx0, cur);
            ggml_format_name(cur, "ffn_moe_relu_sqr-%d", il);
            experts = ggml_mul_mat_id(ctx0, l.down_exps, cur, selected);
            ggml_format_name(experts, "ffn_moe_down-%d", il);
            if (C.on && C.hi_on && !lo && n_tokens < 32) {
                // llama.cpp keeps these experts resident; with them in host memory the GPU must still compute them
                // (the scheduler uploads the used experts, as op offload does for >= 32 tokens)
                ggml_backend_sched_set_tensor_backend(sched, up, gpu);
                ggml_backend_sched_set_tensor_backend(sched, experts, gpu);
            }
        }
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
        gi.s_copy_r.clear();
        if (rows_mode)
            for (int64_t r = 0; r < T; ++r) {
                ggml_tensor* sc = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
                ggml_format_name(sc, "rs_s_copy_r%d", (int) r);
                ggml_set_input(sc);
                gi.s_copy_r.push_back(sc);
            }
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
        if (want_h) {
            ggml_set_output(cur);
            gi.h = cur;
        }
        cur = ggml_mul_mat(ctx0, W.output, cur);
        ggml_set_name(cur, "result_output");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
        gi.logits = cur;
    }

    // cudaHostRegister (read-only) of the host-mapped routed experts, merged into page-aligned ranges, so expert
    // uploads (op-offload copies of prefill, cache admissions) are DMA from page-locked memory (~50+ GB/s measured vs
    // ~20 GB/s pageable: reports/phase3/pin_test).  Stops at the first failure and keeps what is registered.
    void pin_experts() {
        std::vector<std::pair<uintptr_t, uintptr_t>> r;
        const uintptr_t pg = 4096;
        for (const LayerW& l : W.L)
            for (ggml_tensor* t : {l.up_exps, l.down_exps})
                if (t && t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
                    const uintptr_t a = (uintptr_t) t->data & ~(pg - 1);
                    const uintptr_t b = ((uintptr_t) t->data + ggml_nbytes(t) + pg - 1) & ~(pg - 1);
                    r.push_back({a, b});
                }
        std::sort(r.begin(), r.end());
        std::vector<std::pair<uintptr_t, uintptr_t>> mr;
        for (auto& x : r) {
            if (!mr.empty() && x.first <= mr.back().second) mr.back().second = std::max(mr.back().second, x.second);
            else mr.push_back(x);
        }
        // the GGUF loader maps the shards MAP_PRIVATE; cudaHostRegister on a private file mapping does not return
        // under WSL2 (reports/phase3/pin_priv: 1 GiB > 100 s, vs 0.035 s for MAP_SHARED).  The expert ranges are
        // read-only and never written, so they are re-mapped MAP_SHARED|MAP_FIXED from the same file offsets (same
        // bytes, same addresses: the ggml tensors keep their data pointers).
        struct MapEnt { uintptr_t a, b; uint64_t off; std::string path; };
        std::vector<MapEnt> maps;
        {
            std::ifstream mf("/proc/self/maps");
            std::string line;
            while (std::getline(mf, line)) {
                std::istringstream ls(line);
                std::string range, perms, offs, dev, inode, path;
                ls >> range >> perms >> offs >> dev >> inode;
                std::getline(ls, path);
                path.erase(0, path.find_first_not_of(' '));
                if (path.empty() || path[0] != '/') continue;
                const size_t dash = range.find('-');
                maps.push_back({std::stoull(range.substr(0, dash), nullptr, 16), std::stoull(range.substr(dash + 1), nullptr, 16),
                                std::stoull(offs, nullptr, 16), path});
            }
        }
        for (auto& x : mr) {
            const MapEnt* me = nullptr;
            for (const MapEnt& m : maps)
                if (x.first >= m.a && x.second <= m.b) me = &m;
            if (!me) {
                std::fprintf(stderr, "[nh_engine] pin: range not inside one file mapping; stopping\n");
                break;
            }
            const int fd = open(me->path.c_str(), O_RDONLY);
            void* q = fd < 0 ? MAP_FAILED
                             : mmap((void*) x.first, x.second - x.first, PROT_READ, MAP_SHARED | MAP_FIXED, fd,
                                    (off_t) (me->off + (x.first - me->a)));
            if (fd >= 0) close(fd);
            if (q != (void*) x.first) {
                // MAP_FIXED failure leaves the old mapping in place only if mmap failed before unmapping; abort loudly
                throw std::runtime_error("nh_engine: pin remap failed for " + me->path);
            }
            const cudaError_t e = cudaHostRegister((void*) x.first, x.second - x.first, cudaHostRegisterReadOnly);
            if (e != cudaSuccess) {
                cudaGetLastError();
                std::fprintf(stderr, "[nh_engine] cudaHostRegister failed (%s) after %.1f MiB; the rest stays pageable\n",
                             cudaGetErrorString(e), pinned_bytes / 1048576.0);
                break;
            }
            pinned_bytes += x.second - x.first;
            pinned_ranges.push_back({x.first, x.second - x.first});
        }
        std::fprintf(stderr, "[nh_engine] page-locked %.1f MiB of host experts in %zu ranges\n", pinned_bytes / 1048576.0, mr.size());
    }

    // vram_limit_used: 0, or a cap on the process's total VRAM use (bytes)
    void init_cache(size_t vram_limit_used) {
        g_cache = &C;
        C.on = true;
        C.n_used = W.g.n_expert_used;
        // review C1: the cache's plan layout, scratch and CPU-exact kernels are written for this geometry; refuse others
        if (C.n_used < 1 || C.n_used * NH_TMAX > NU_MAX)
            throw std::runtime_error("nh_engine cache: expert_used_count " + std::to_string(C.n_used) + " > 22 is not supported; run with --cache off");
        int n_expert = 0;
        for (const LayerW& l : W.L)
            if (l.kind == nh::Kind::Moe) {
                C.n_in = (int) l.up_exps->ne[0];
                C.n_ff = (int) l.up_exps->ne[1];
                n_expert = (int) l.up_exps->ne[2];
                break;
            }
        C.q8k_tr = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
        C.q80_tr = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        auto ck = [](cudaError_t e, const char* w) {
            if (e != cudaSuccess) throw std::runtime_error(std::string("nh_engine cache: ") + w + ": " + cudaGetErrorString(e));
        };
        ck(cudaStreamCreateWithFlags(&C.s_hit, cudaStreamNonBlocking), "stream");
        ck(cudaStreamCreateWithFlags(&C.s_copy, cudaStreamNonBlocking), "stream");
        ck(cudaStreamCreateWithFlags(&C.s_hi, cudaStreamNonBlocking), "stream");
        ck(cudaEventCreateWithFlags(&C.ev_hit, cudaEventDisableTiming), "event");
        const size_t NU = NU_MAX, F = (size_t) C.n_ff, D = (size_t) C.n_in;
        if (C.n_in != 1024 || C.n_ff <= 0 || C.n_ff % 32 != 0)
            throw std::runtime_error("nh_engine cache: expert shape " + std::to_string(C.n_in) + "x" + std::to_string(C.n_ff) +
                                     " is not supported (needs a 1024-wide latent and n_ff % 32 == 0); run with --cache off");
        ck(cudaMalloc(&C.d_args, ARGS_BYTES), "scratch");
        ck(cudaMalloc(&C.d_q8k, NH_TMAX * PLAN_Q8K_ROW), "scratch");
        ck(cudaMalloc(&C.d_upo, NU * F * 4), "scratch");
        ck(cudaMalloc(&C.d_hq, NU * (F / 32) * nhx::Q80_BYTES), "scratch");
        ck(cudaMalloc(&C.d_out, NU * D * 4), "scratch");
        ck(cudaHostAlloc(&C.h_args, ARGS_BYTES, cudaHostAllocDefault), "pinned");
        ck(cudaHostAlloc(&C.h_q8k, NH_TMAX * PLAN_Q8K_ROW, cudaHostAllocDefault), "pinned");
        ck(cudaHostAlloc(&C.h_out, NU * D * 4, cudaHostAllocDefault), "pinned");
        // VRAM budget: what is free now minus what keeps the process under vram_limit_used, minus a margin for
        // CUDA graphs / cuBLAS workspaces that appear at the first decode
        // review H3: budget = free VRAM (cudaMemGetInfo) - reserve; --vram-limit-mib additionally caps the process total
        // (used + pools + a 384 MiB margin for CUDA graphs / cuBLAS workspaces that appear at the first decode)
        size_t fr = 0, tot = 0;
        ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
        const size_t used = tot - fr, margin = 384ull << 20, reserve = (size_t) std::max(0L, vram_reserve_mib) << 20;
        size_t budget = fr > reserve ? fr - reserve : 0;
        if (vram_limit_used) budget = std::min(budget, used + margin < vram_limit_used ? vram_limit_used - used - margin : 0);
        if (cache_mib >= 0) budget = std::min(budget, (size_t) cache_mib << 20);
        C.pools.resize((size_t) W.g.n_layer);
        size_t per_set = 0;
        int n_lo = 0, n_hi = 0;
        for (int il = 0; il < W.g.n_layer; ++il) {
            const LayerW& l = W.L[(size_t) il];
            if (l.kind != nh::Kind::Moe) continue;
            const bool lo = il < n_cpu_moe;
            if (!lo && !C.hi_on) continue;
            if (!ggml_backend_buffer_is_host(l.up_exps->buffer)) throw std::runtime_error("nh_engine cache: experts not host-mapped");
            {
                // review C1: every layer must match the shapes the scratch was sized for, and a lo pool needs the
                // quant types the CPU-exact hit kernels implement (with ggml-cpu's matching vec_dot activation type);
                // a layer that fails gets no pool and runs the plain llama.cpp path
                std::string why;
                const ggml_type tu = l.up_exps->type, td = l.down_exps->type;
                if (l.up_exps->ne[0] != C.n_in || l.up_exps->ne[1] != C.n_ff || l.down_exps->ne[0] != C.n_ff ||
                    l.down_exps->ne[1] != C.n_in || l.up_exps->ne[2] != n_expert || l.down_exps->ne[2] != n_expert)
                    why = "expert shape differs from layer 0's";
                else if (lo && !((tu == GGML_TYPE_Q4_K || tu == GGML_TYPE_Q5_K || tu == GGML_TYPE_Q6_K) &&
                                 ggml_get_type_traits_cpu(tu)->vec_dot_type == GGML_TYPE_Q8_K))
                    why = std::string("up type ") + ggml_type_name(tu) + " has no CPU-exact hit kernel";
                else if (lo && !((td == GGML_TYPE_Q5_0 || td == GGML_TYPE_IQ4_NL) && ggml_get_type_traits_cpu(td)->vec_dot_type == GGML_TYPE_Q8_0))
                    why = std::string("down type ") + ggml_type_name(td) + " has no CPU-exact hit kernel";
                if (!why.empty()) {
                    if (!lo) throw std::runtime_error("nh_engine cache: layer " + std::to_string(il) + ": " + why + "; run with --cache lo or off");
                    std::fprintf(stderr, "[nh_engine] cache: layer %d not cached (%s)\n", il, why.c_str());
                    continue;
                }
            }
            auto p = std::make_unique<SlotPool>();
            p->il = il;
            p->lo = lo;
            p->tu = l.up_exps->type;
            p->td = l.down_exps->type;
            p->up_row = ggml_row_size(p->tu, C.n_in);
            p->dn_row = ggml_row_size(p->td, C.n_ff);
            p->up_exp = l.up_exps->nb[2];
            p->dn_exp = l.down_exps->nb[2];
            if (p->up_exp != p->up_row * F || p->dn_exp != p->dn_row * D) throw std::runtime_error("nh_engine cache: expert stride");
            p->host_up = (const uint8_t*) l.up_exps->data;
            p->host_dn = (const uint8_t*) l.down_exps->data;
            per_set += p->up_exp + p->dn_exp;
            (lo ? n_lo : n_hi)++;
            C.pools[(size_t) il] = std::move(p);
        }
        const int cap_u = per_set ? (int) std::min<size_t>(budget / per_set, (size_t) n_expert) : 0;
        ggml_init_params ip{ggml_tensor_overhead() * 2 * (size_t) W.g.n_layer + 1024, nullptr, true};
        C.ctx = ggml_init(ip);
        for (auto& pp : C.pools) {
            if (!pp) continue;
            int cap = pp->lo ? (cap_lo > 0 ? cap_lo : cap_u) : (cap_hi > 0 ? cap_hi : cap_u);
            cap = std::min(cap, n_expert);
            if (!pp->lo && cap < C.n_used * std::max(k_snap, rows_max))
                throw std::runtime_error("nh_engine cache: VRAM budget too small for the hi pools (need " + std::to_string(C.n_used * std::max(k_snap, rows_max)) +
                                         " slots per layer, have " + std::to_string(cap) + "); lower --vram-reserve-mib or use --cache lo");
            pp->cap = cap;
            pp->t_up = ggml_new_tensor_3d(C.ctx, pp->tu, C.n_in, C.n_ff, std::max(cap, 1));
            pp->t_dn = ggml_new_tensor_3d(C.ctx, pp->td, C.n_ff, C.n_in, std::max(cap, 1));
            ggml_format_name(pp->t_up, "pool_up-%d", pp->il);
            ggml_format_name(pp->t_dn, "pool_down-%d", pp->il);
            pp->slot_of.assign((size_t) n_expert, -1);
            pp->exp_of.assign((size_t) std::max(cap, 1), -1);
            pp->pending.assign((size_t) std::max(cap, 1), 0);
            pp->ready.resize((size_t) std::max(cap, 1));
            for (auto& ev : pp->ready) ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "event");
            pp->score.assign((size_t) n_expert, 0.0);
            pp->last.assign((size_t) n_expert, 0);
        }
        C.buf = ggml_backend_alloc_ctx_tensors_from_buft(C.ctx, ggml_backend_get_default_buffer_type(gpu));
        if (!C.buf) throw std::runtime_error("nh_engine cache: pool allocation failed");
        ggml_backend_buffer_set_usage(C.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        C.pool_bytes = ggml_backend_buffer_get_size(C.buf);
        ggml_backend_dev_memory(ggml_backend_get_device(gpu), &fr, &tot);
        std::fprintf(stderr, "[nh_engine] expert cache: %d lo + %d hi pools, slots/layer lo %d hi %d, %.1f MiB (budget %.1f MiB); "
                             "GPU free now %.1f MiB\n", n_lo, n_hi, cap_lo > 0 ? cap_lo : cap_u, cap_hi > 0 ? cap_hi : cap_u,
                     C.pool_bytes / 1048576.0, budget / 1048576.0, fr / 1048576.0);
    }

    void cache_stats(FILE* f) {
        if (!C.on) return;
        long h = 0, mi = 0, up = 0;
        double tp = 0, tu = 0, td = 0, tw = 0, th = 0;
        for (auto& pp : C.pools) {
            if (!pp) continue;
            std::fprintf(f, "[cache] blk %2d %s cap %3d hits %8ld misses %8ld hit %.4f uploads %6ld skips %6ld plan %.3f up %.3f "
                            "down %.3f wait %.3f hi_upload %.3f s\n", pp->il, pp->lo ? "lo" : "hi", pp->cap, pp->hits, pp->misses,
                         pp->hits + pp->misses ? (double) pp->hits / (double) (pp->hits + pp->misses) : 0.0, pp->uploads,
                         pp->upload_skips, pp->t_plan, pp->tm_up, pp->tm_dn, pp->t_wait, pp->t_hi_upload);
            h += pp->hits;
            mi += pp->misses;
            up += pp->uploads;
            tp += pp->t_plan;
            tu += pp->tm_up;
            td += pp->tm_dn;
            tw += pp->t_wait;
            th += pp->t_hi_upload;
        }
        std::fprintf(f, "[cache] total hits %ld misses %ld hit %.4f uploads %ld steps %lld; time plan %.2f up %.2f down %.2f wait %.2f "
                        "hi_upload %.2f s\n", h, mi, h + mi ? (double) h / (double) (h + mi) : 0.0, up, (long long) C.step, tp, tu, td, tw, th);
    }

    void init(const nh::Model& m, int n_cpu_moe_, int n_slots, int n_ctx, int threads, int ubatch, int cache_mode = 0) {
        n_cpu_moe = n_cpu_moe_;
        gpu = ggml_backend_cuda_init(0);
        cpu = ggml_backend_cpu_init();
        if (!gpu || !cpu) throw std::runtime_error("nh_engine: backend init failed");
        n_threads = threads;
        ggml_backend_cpu_set_n_threads(cpu, n_threads);
        C.hi_on = cache_mode == 2;
        // cache mode 2 keeps every routed expert in host memory (the VRAM of the resident blocks becomes slot pools)
        load_weights(m, cache_mode == 2 ? m.g.n_layer : n_cpu_moe, ggml_backend_get_default_buffer_type(gpu), W, mtp_on);
        if (mtp_on) {
            // the MTP head's own KV cache (dense NoPE attention), one sequence
            ggml_init_params mp{ggml_tensor_overhead() * 4, nullptr, true};
            mtp_ctx = ggml_init(mp);
            const int64_t n_embd_kv_m = W.mtp.wk->ne[1];
            mtp_k = ggml_new_tensor_2d(mtp_ctx, GGML_TYPE_F16, n_embd_kv_m, n_ctx);
            mtp_v = ggml_new_tensor_2d(mtp_ctx, GGML_TYPE_F16, n_embd_kv_m, n_ctx);
            mtp_buf = ggml_backend_alloc_ctx_tensors(mtp_ctx, gpu);
            if (!mtp_buf) throw std::runtime_error("nh_engine: VRAM allocation of the MTP KV cache failed");
            ggml_backend_buffer_clear(mtp_buf, 0);
        }
        if (pin) pin_experts();
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
        if (cache_mode) init_cache(vram_limit_mib > 0 ? (size_t) vram_limit_mib << 20 : 0);
    }

    void clear_slot(int slot) { M.n_past[(size_t) slot] = 0; }

    // one ubatch of tokens for `slot`; all_out: logits for every token, else for the last one only
    // verify_: a speculative verify batch (all outputs; every token computed as a T == 1 step would compute it; the
    // recurrent states keep T rollback snapshots, see spec_commit).  want_h_: also return the result_norm rows (hrows).
    void check_tokens(const int32_t* tok, int64_t T) const {
        for (int64_t i = 0; i < T; ++i)
            if (tok[i] < 0 || tok[i] >= W.g.n_vocab) throw std::runtime_error("nh_engine: token id " + std::to_string(tok[i]) + " out of range");
    }

    // review H4: fail the request if a cache op recorded a CUDA error during the last graph
    void check_cache_error() {
        if (C.err.load()) {
            const std::string m = C.err_msg;
            throw std::runtime_error("nh_engine: CUDA error in the expert cache: " + m);
        }
    }

    Engine() = default;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // review M2: teardown in dependency order (streams idle -> cache scratch/pools -> unpin -> scheduler/buffers ->
    // backends); the GGUF mapping is unmapped after this (main declares the model before the engine)
    ~Engine() {
        for (cudaStream_t st : {C.s_hit, C.s_copy, C.s_hi})
            if (st) cudaStreamSynchronize(st);
        if (sched) ggml_backend_sched_free(sched);
        for (auto& pp : C.pools)
            if (pp)
                for (cudaEvent_t ev : pp->ready)
                    if (ev) cudaEventDestroy(ev);
        C.pools.clear();
        if (C.buf) ggml_backend_buffer_free(C.buf);
        if (C.ctx) ggml_free(C.ctx);
        if (C.ev_hit) cudaEventDestroy(C.ev_hit);
        for (cudaStream_t st : {C.s_hit, C.s_copy, C.s_hi})
            if (st) cudaStreamDestroy(st);
        for (void* d : {(void*) C.d_args, (void*) C.d_q8k, (void*) C.d_upo, (void*) C.d_hq, (void*) C.d_out})
            if (d) cudaFree(d);
        for (void* h : {(void*) C.h_args, (void*) C.h_q8k, (void*) C.h_out})
            if (h) cudaFreeHost(h);
        if (g_cache == &C) g_cache = nullptr;
        for (auto& r : pinned_ranges) cudaHostUnregister((void*) r.first);
        if (mtp_buf) ggml_backend_buffer_free(mtp_buf);
        if (mtp_ctx) ggml_free(mtp_ctx);
        if (M.buf) ggml_backend_buffer_free(M.buf);
        if (M.ctx) ggml_free(M.ctx);
        if (W.buf_gpu) ggml_backend_buffer_free(W.buf_gpu);
        for (auto b : W.bufs_cpu) ggml_backend_buffer_free(b);
        if (W.ctx_gpu) ggml_free(W.ctx_gpu);
        if (W.ctx_cpu) ggml_free(W.ctx_cpu);
        if (gpu) ggml_backend_free(gpu);
        if (cpu) ggml_backend_free(cpu);
    }

    void decode(const int32_t* tok, int64_t T, int slot, bool all_out, bool verify_ = false, bool want_h_ = false) {
        check_tokens(tok, T);
        const int64_t pos0 = M.n_past[(size_t) slot];
        if (pos0 + T > M.n_ctx) throw std::runtime_error("nh_engine: context full");
        if (verify_ && (T > NH_TMAX || T > k_snap)) throw std::runtime_error("nh_engine: verify batch too large");
        verify = verify_ && T > 1;
        want_h = want_h_ || mtp_on;
        vpos0 = pos0;
        if (verify_) all_out = true;
        if (C.on && (T == 1 || verify)) C.step++;
        const int64_t n_kv = kv_len(pos0 + T);
        const int64_t n_out = all_out ? T : 1;
        const int64_t rs_zero = pos0 == 0 ? slot : -1;
        GraphIn gi;
        ggml_backend_sched_reset(sched);
        if (prof.on) {
            prof.sched = sched;
            ggml_backend_sched_set_eval_callback(sched, prof_cb, &prof);
        } else if (diff.mode) {
            diff.seen.clear();
            ggml_backend_sched_set_eval_callback(sched, diff_cb, &diff);
        } else {
            ggml_backend_sched_set_eval_callback(sched, dump.f ? dump_cb : nullptr, &dump);
        }
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
        if (prof.on) prof.t_last = now_s();
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("nh_engine: graph compute failed");
        check_cache_error();
        const size_t nv = (size_t) gi.logits->ne[0];
        logits.resize(nv * (size_t) n_out);
        ggml_backend_tensor_get(gi.logits, logits.data(), 0, logits.size() * 4);
        if (want_h && gi.h) {
            hrows.resize((size_t) ggml_nelements(gi.h));
            ggml_backend_tensor_get(gi.h, hrows.data(), 0, hrows.size() * 4);
        }
        ggml_free(ctx0);
        ctx0 = nullptr;
        verify = false;
        want_h = false;
        M.n_past[(size_t) slot] = (int) (pos0 + T);
    }

    // MTP head over T (h, token) pairs written to MTP KV cells pos0..pos0+T-1 (graph_mtp of nemotron-h-moe.cpp): h_j is
    // the main model's result_norm row of position j and tok the token at j + 1.  Leaves the last row's logits (the draft
    // for position j + 2) and its h_nextn (the h of a chained step) in mtp_logits / mtp_h.
    void mtp_decode(const int32_t* tok, const float* h, int64_t T, int64_t pos0) {
        const MtpW& w = W.mtp;
        const nh::Geometry& g = W.g;
        const int64_t E = g.n_embd, hd = g.head_dim;
        const int64_t n_head_q = w.wq->ne[1] / hd, n_head_kv = w.wk->ne[1] / hd;
        if (pos0 + T > M.n_ctx) throw std::runtime_error("nh_engine: MTP context full");
        check_tokens(tok, T);
        const int64_t n_kv = kv_len(pos0 + T);
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
        ggml_init_params ip{meta.size(), meta.data(), true};
        ctx0 = ggml_init(ip);
        gf = ggml_new_graph_custom(ctx0, max_nodes, false);
        ggml_tensor* tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
        ggml_set_input(tokens);
        ggml_tensor* hin = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, E, T);
        ggml_set_input(hin);
        ggml_tensor* kidx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, T);
        ggml_set_input(kidx);
        ggml_tensor* mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, n_kv, T, 1, 1);
        ggml_set_input(mask);
        ggml_tensor* emb = ggml_get_rows(ctx0, W.tok_embd, tokens);
        auto rms = [&](ggml_tensor* x, ggml_tensor* wt) { return ggml_mul(ctx0, ggml_rms_norm(ctx0, x, g.rms_eps), wt); };
        ggml_tensor* e_norm = rms(emb, w.enorm);
        ggml_tensor* h_norm = rms(hin, w.hnorm);
        ggml_tensor* cur = ggml_mul_mat(ctx0, w.eh_proj, ggml_concat(ctx0, e_norm, h_norm, 0));
        ggml_tensor* inpSA = cur;
        cur = rms(cur, w.attn_norm);
        {
            ggml_tensor* Q = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, w.wq, cur), hd, n_head_q, T);
            ggml_tensor* K = ggml_mul_mat(ctx0, w.wk, cur);
            ggml_tensor* V = ggml_mul_mat(ctx0, w.wv, cur);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx0, mtp_k, K, kidx));
            ggml_build_forward_expand(gf, ggml_set_rows(ctx0, mtp_v, V, kidx));
            ggml_tensor* k = ggml_view_3d(ctx0, mtp_k, hd, n_head_kv, n_kv, ggml_row_size(mtp_k->type, hd), mtp_k->nb[1], 0);
            ggml_tensor* v = ggml_view_3d(ctx0, mtp_v, hd, n_head_kv, n_kv, ggml_row_size(mtp_v->type, hd), mtp_v->nb[1], 0);
            ggml_tensor* q = ggml_permute(ctx0, Q, 0, 2, 1, 3);
            k = ggml_permute(ctx0, k, 0, 2, 1, 3);
            v = ggml_permute(ctx0, v, 0, 2, 1, 3);
            ggml_tensor* o = ggml_flash_attn_ext(ctx0, q, k, v, mask, 1.0f / std::sqrt((float) hd), 0.0f, 0.0f);
            ggml_prec_set_acc(o, GGML_PREC_F32);
            o = ggml_reshape_2d(ctx0, o, o->ne[0] * o->ne[1], o->ne[2] * o->ne[3]);
            cur = ggml_mul_mat(ctx0, w.wo, o);
        }
        cur = ggml_add(ctx0, cur, inpSA);
        // only the last row goes on (crop_before_nextn)
        cur = ggml_view_2d(ctx0, cur, E, 1, cur->nb[1], (size_t) (T - 1) * cur->nb[1]);
        ggml_tensor* ffn_res = cur;
        cur = rms(cur, w.post_attn_norm);
        {
            ggml_tensor* logits_r = ggml_mul_mat(ctx0, w.gate_inp, cur);
            ggml_tensor* tmp = ggml_mul_mat(ctx0, w.up_shexp, cur);
            ggml_tensor* sh = ggml_mul_mat(ctx0, w.down_shexp, ggml_sqr(ctx0, ggml_relu(ctx0, tmp)));
            ggml_tensor* lat = ggml_mul_mat(ctx0, w.latent_down, cur);
            const int64_t n_expert = w.up_exps->ne[2];
            const int64_t n_used = mtp_topk > 0 ? std::min<int64_t>(mtp_topk, g.n_expert_used) : g.n_expert_used;
            ggml_tensor* probs = ggml_sigmoid(ctx0, logits_r);
            ggml_tensor* selected = ggml_argsort_top_k(ctx0, ggml_add(ctx0, probs, w.exp_probs_b), (int) n_used);
            ggml_tensor* wts = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, probs, 1, n_expert, 1), selected);
            if (g.expert_weights_norm) {
                wts = ggml_reshape_2d(ctx0, wts, n_used, 1);
                ggml_tensor* sum = ggml_clamp(ctx0, ggml_sum_rows(ctx0, wts), 6.103515625e-5, INFINITY);
                wts = ggml_reshape_3d(ctx0, ggml_div(ctx0, wts, sum), 1, n_used, 1);
            }
            if (g.expert_weights_scale != 0.0f && g.expert_weights_scale != 1.0f) wts = ggml_scale(ctx0, wts, g.expert_weights_scale);
            const int64_t n_lat = lat->ne[0];
            ggml_tensor* x3 = ggml_reshape_3d(ctx0, lat, n_lat, 1, 1);
            ggml_tensor* up = ggml_mul_mat_id(ctx0, w.up_exps, x3, selected);
            ggml_tensor* ex = ggml_mul_mat_id(ctx0, w.down_exps, ggml_sqr(ctx0, ggml_relu(ctx0, up)), selected);
            ggml_backend_sched_set_tensor_backend(sched, up, cpu);
            ggml_backend_sched_set_tensor_backend(sched, ex, cpu);
            ex = ggml_mul(ctx0, ex, wts);
            ggml_tensor* moe = ggml_view_2d(ctx0, ex, n_lat, 1, ex->nb[2], 0);
            for (int64_t i = 1; i < n_used; ++i) moe = ggml_add(ctx0, moe, ggml_view_2d(ctx0, ex, n_lat, 1, ex->nb[2], (size_t) i * ex->nb[1]));
            if (n_used == 1) moe = ggml_cont(ctx0, moe);
            moe = ggml_mul_mat(ctx0, w.latent_up, moe);
            cur = ggml_add(ctx0, moe, sh);
        }
        cur = ggml_add(ctx0, cur, ffn_res);
        ggml_tensor* hn = ggml_mul(ctx0, ggml_norm(ctx0, cur, g.norm_eps), w.shared_head_norm);
        ggml_set_output(hn);
        ggml_tensor* lg = ggml_mul_mat(ctx0, W.output, hn);
        ggml_set_output(lg);
        ggml_build_forward_expand(gf, hn);
        ggml_build_forward_expand(gf, lg);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) throw std::runtime_error("nh_engine: MTP graph alloc failed");
        ggml_backend_tensor_set(tokens, tok, 0, (size_t) T * 4);
        ggml_backend_tensor_set(hin, h, 0, (size_t) (T * E) * 4);
        {
            std::vector<int64_t> idx((size_t) T);
            for (int64_t i = 0; i < T; ++i) idx[(size_t) i] = pos0 + i;
            ggml_backend_tensor_set(kidx, idx.data(), 0, idx.size() * 8);
            std::vector<ggml_fp16_t> mk((size_t) (n_kv * T));
            const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f), drop = ggml_fp32_to_fp16(-INFINITY);
            for (int64_t i = 0; i < T; ++i)
                for (int64_t j = 0; j < n_kv; ++j) mk[(size_t) (i * n_kv + j)] = j <= pos0 + i ? keep : drop;
            ggml_backend_tensor_set(mask, mk.data(), 0, mk.size() * sizeof(ggml_fp16_t));
        }
        cx.reset();
        ggml_backend_cpu_set_n_threads(cpu, n_threads);
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("nh_engine: MTP compute failed");
        check_cache_error();
        mtp_logits.resize((size_t) lg->ne[0]);
        ggml_backend_tensor_get(lg, mtp_logits.data(), 0, mtp_logits.size() * 4);
        mtp_h.resize((size_t) E);
        ggml_backend_tensor_get(hn, mtp_h.data(), 0, mtp_h.size() * 4);
        ggml_free(ctx0);
        ctx0 = nullptr;
    }

    // after a verify batch of T tokens at pos0: keep the first a (1 <= a <= T).  The KV cells past pos0 + a stay but are
    // masked; the recurrent states go back to the snapshot taken after token a - 1 (snapshot row T - a, ssm-scan.cu)
    void spec_commit(int slot, int64_t pos0, int T, int a) {
        M.n_past[(size_t) slot] = (int) (pos0 + a);
        if (a >= T) return;
        const int64_t sl = T - a;
        for (int il = 0; il < W.g.n_layer; ++il) {
            for (ggml_tensor* t : {M.r[(size_t) il], M.s[(size_t) il]}) {
                if (!t) continue;
                const size_t row = t->nb[1];
                const cudaError_t e = cudaMemcpy((char*) t->data + (size_t) slot * row, (const char*) t->data + (size_t) (sl * M.n_slots + slot) * row,
                                                 row, cudaMemcpyDeviceToDevice);
                if (e != cudaSuccess) throw std::runtime_error(std::string("nh_engine: rollback copy: ") + cudaGetErrorString(e));
            }
        }
    }

    // a --batch decode window: the next token of S sessions (distinct slots) in one graph, row r computed as session
    // slots[r]'s own T == 1 step (rows mode).  Leaves S logits rows.
    void decode_rows(const int* slots, const int32_t* tok, int S) {
        check_tokens(tok, S);
        if (S < 1 || S > NH_TMAX || S > M.n_slots) throw std::runtime_error("nh_engine: decode window of " + std::to_string(S) + " rows");
        row_slot.assign(slots, slots + S);
        row_pos.assign((size_t) S, 0);
        int64_t n_kv = 0;
        for (int r = 0; r < S; ++r) {
            for (int q = 0; q < r; ++q)
                if (slots[q] == slots[r]) throw std::runtime_error("nh_engine: a decode window names a slot twice");
            const int64_t pos = M.n_past[(size_t) slots[r]];
            if (pos < 1 || pos + 1 > M.n_ctx) throw std::runtime_error("nh_engine: decode window: slot context empty or full");
            row_pos[(size_t) r] = pos;
            n_kv = std::max(n_kv, kv_len(pos + 1));
        }
        rows_mode = true;
        verify = false;
        want_h = false;
        if (C.on) C.step++;
        GraphIn gi;
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
        build_graph(S, 0, 0, n_kv, S, -1, gi);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) throw std::runtime_error("nh_engine: rows graph alloc failed");
        ggml_backend_tensor_set(gi.tokens, tok, 0, (size_t) S * 4);
        {
            std::vector<int64_t> idx((size_t) S);
            for (int r = 0; r < S; ++r) idx[(size_t) r] = (int64_t) slots[r] * M.n_ctx * (M.n_slots > 1 ? 1 : 0) + row_pos[(size_t) r];
            ggml_backend_tensor_set(gi.k_idxs, idx.data(), 0, idx.size() * 8);
            ggml_backend_tensor_set(gi.v_idxs, idx.data(), 0, idx.size() * 8);
            std::vector<ggml_fp16_t> mask((size_t) (n_kv * S));
            const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f), drop = ggml_fp32_to_fp16(-INFINITY);
            for (int r = 0; r < S; ++r)
                for (int64_t j = 0; j < n_kv; ++j) mask[(size_t) (r * n_kv + j)] = j <= row_pos[(size_t) r] ? keep : drop;
            ggml_backend_tensor_set(gi.kq_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
            for (int r = 0; r < S; ++r) {
                const int32_t sc = slots[r];
                ggml_backend_tensor_set(gi.s_copy_r[(size_t) r], &sc, 0, 4);
            }
            std::vector<int32_t> oi((size_t) S);
            for (int r = 0; r < S; ++r) oi[(size_t) r] = r;
            ggml_backend_tensor_set(gi.out_ids, oi.data(), 0, oi.size() * 4);
        }
        cx.reset();
        ggml_backend_cpu_set_n_threads(cpu, n_threads);
        const ggml_status st = ggml_backend_sched_graph_compute(sched, gf);
        rows_mode = false;
        if (st != GGML_STATUS_SUCCESS) throw std::runtime_error("nh_engine: rows graph compute failed");
        check_cache_error();
        const size_t nv = (size_t) gi.logits->ne[0];
        logits.resize(nv * (size_t) S);
        ggml_backend_tensor_get(gi.logits, logits.data(), 0, logits.size() * 4);
        ggml_free(ctx0);
        ctx0 = nullptr;
        for (int r = 0; r < S; ++r) M.n_past[(size_t) slots[r]] = (int) (row_pos[(size_t) r] + 1);
    }

    // session dst takes session src's state (its first n_past cells of K/V, the conv and SSM state rows)
    void copy_session(int src, int dst) {
        if (src == dst) return;
        const int64_t n = M.n_past[(size_t) src];
        for (int il = 0; il < W.g.n_layer; ++il) {
            for (ggml_tensor* t : {M.k[(size_t) il], M.v[(size_t) il]}) {
                if (!t || n == 0) continue;
                const cudaError_t e = cudaMemcpy((char*) t->data + (size_t) dst * t->nb[2], (const char*) t->data + (size_t) src * t->nb[2],
                                                 (size_t) n * t->nb[1], cudaMemcpyDeviceToDevice);
                if (e != cudaSuccess) throw std::runtime_error(std::string("nh_engine: session copy: ") + cudaGetErrorString(e));
            }
            for (ggml_tensor* t : {M.r[(size_t) il], M.s[(size_t) il]}) {
                if (!t) continue;
                const cudaError_t e = cudaMemcpy((char*) t->data + (size_t) dst * t->nb[1], (const char*) t->data + (size_t) src * t->nb[1],
                                                 t->nb[1], cudaMemcpyDeviceToDevice);
                if (e != cudaSuccess) throw std::runtime_error(std::string("nh_engine: session copy: ") + cudaGetErrorString(e));
            }
        }
        M.n_past[(size_t) dst] = (int) n;
    }
};

// ---- speculative decoding (Phase 4): drafts from prompt lookup (n-gram) and/or the MTP head, verified greedily in
// one batch whose every row is computed with T == 1 numerics (Engine::decode verify), so the accepted tokens and
// their logits rows equal the plain T == 1 decode's ----
struct SpecStats {
    long steps = 0, drafted = 0, accepted = 0, plain_steps = 0;
    long src_drafted[3] = {0, 0, 0}, src_accepted[3] = {0, 0, 0}, src_steps[3] = {0, 0, 0};   // 0 ngram, 1 mtp, 2 mixed
    std::map<int, long> acc_hist;   // accepted drafts per verify step
    double t_draft = 0, t_verify = 0, t_plain = 0, t_commit = 0;
    double t_by_T[NH_TMAX + 1] = {};
    long n_by_T[NH_TMAX + 1] = {};
    long mtp_calls = 0, mtp_rows = 0, mtp_first_total = 0, mtp_first_ok = 0, gated_steps = 0;
};

// most recent earlier occurrence of the last n context tokens (n = nmax .. nmin), and up to k tokens that followed it
std::vector<int32_t> ngram_propose(const std::vector<int32_t>& c, int k, int nmin, int nmax, int* n_match) {
    *n_match = 0;
    const int64_t L = (int64_t) c.size();
    for (int n = nmax; n >= nmin; --n) {
        if (L < n + 1) continue;
        const int32_t* suf = c.data() + L - n;
        for (int64_t i = L - n - 1; i >= 0; --i) {
            if (c[(size_t) i] != suf[0] || !std::equal(suf, suf + n, c.data() + i)) continue;
            std::vector<int32_t> d;
            for (int j = 0; j < k && i + n + j < L; ++j) d.push_back(c[(size_t) (i + n + j)]);
            if (!d.empty()) {
                *n_match = n;
                return d;
            }
        }
    }
    return {};
}

struct SpecCfg {
    int mode = 0;        // 0 off, 1 ngram, 2 mtp, 3 auto (both)
    int k = 3;           // max drafts per step (verify batch = k + 1 <= NH_TMAX)
    int ng_min = 2, ng_max = 4;
    int ng_strong = 3;   // auto: an n-gram match at least this long wins over the MTP draft
    double min_acc = 0.0;   // adaptive: skip drafting while the EMA of the per-draft acceptance is below this
    int probe_every = 8;    // ... but still draft every probe_every-th step to track it
};

// Generation after the prompt: `last` = logits of the last prompt token (and, with the MTP head, E.hrows = its
// result_norm row).  Emits tokens (and, through on_row, the logits row that predicted each one) exactly as the plain
// loop does: same tokens, same rows, same stop rules.
// serve: `slot` is the session, `cancel()` true ends the answer after the token just emitted, `between()` runs between
// two steps (the --batch windows of the other sessions) and true there means this step drafts nothing.
template <class OnRow>
void spec_generate(Engine& E, const SpecCfg& cfg, SpecStats& st, std::vector<int32_t> ctx_toks, std::vector<float> last,
                   int maxgen, bool stop_eog, const std::set<int>& eog, std::vector<int32_t>& out, OnRow on_row,
                   std::vector<double>* t_tok, int slot = 0, const std::function<bool()>& cancel = nullptr,
                   const std::function<bool()>& between = nullptr) {
    const int nv = (int) last.size();
    const int64_t ne = E.W.g.n_embd;
    const bool use_mtp = E.mtp_on && (cfg.mode == 2 || cfg.mode == 3);
    // MTP inputs not yet run through the head: pairs (h_j, token_{j+1}); h_last = h of the row that predicted x
    std::vector<float> pend_h, h_last;
    std::vector<int32_t> pend_t;
    int64_t mtp_pos = 0;
    double ema = 1.0;   // per-draft-token acceptance, EMA over verify steps
    int since_probe = 0;
    if (use_mtp) {
        if (E.hrows.size() < (size_t) ne) throw std::runtime_error("spec: no h rows from the prompt");
        h_last.assign(E.hrows.end() - ne, E.hrows.end());
    }
    for (;;) {
        const int x = argmax(last.data(), nv);
        on_row(last.data(), x);
        out.push_back(x);
        ctx_toks.push_back(x);
        if (t_tok) t_tok->push_back(now_s());
        if ((stop_eog && eog.count(x)) || (int) out.size() >= maxgen || (cancel && cancel())) break;
        const bool busy = between && between();
        const double td0 = now_s();
        int kmax = std::min(cfg.k, std::min(NH_TMAX - 1, maxgen - (int) out.size()));
        if (busy) {
            kmax = 0;
        } else if (cfg.min_acc > 0 && ema < cfg.min_acc && ++since_probe < cfg.probe_every) {
            kmax = 0;
            st.gated_steps++;
        } else {
            since_probe = 0;
        }
        int nm = 0;
        std::vector<int32_t> dn, dm, d;
        if (kmax > 0 && (cfg.mode == 1 || cfg.mode == 3)) dn = ngram_propose(ctx_toks, kmax, cfg.ng_min, cfg.ng_max, &nm);
        if (use_mtp) {
            pend_h.insert(pend_h.end(), h_last.begin(), h_last.end());
            pend_t.push_back(x);
            if (kmax > 0) {
                const int64_t n = (int64_t) pend_t.size();
                E.mtp_decode(pend_t.data(), pend_h.data(), n, mtp_pos);
                st.mtp_calls++;
                st.mtp_rows += n;
                mtp_pos += n;
                pend_t.clear();
                pend_h.clear();
                dm.push_back(argmax(E.mtp_logits.data(), nv));
                for (int j = 1; j < kmax; ++j) {
                    // chained draft: the head's own h_nextn with its last draft, in a scratch cell past the committed ones
                    std::vector<float> hc = E.mtp_h;
                    E.mtp_decode(&dm.back(), hc.data(), 1, mtp_pos + j - 1);
                    st.mtp_calls++;
                    st.mtp_rows++;
                    dm.push_back(argmax(E.mtp_logits.data(), nv));
                }
            }
        }
        int src = -1;
        if (cfg.mode == 1) {
            d = dn;
            src = 0;
        } else if (cfg.mode == 2) {
            d = dm;
            src = 1;
        } else if (cfg.mode == 3) {
            // prefer the n-gram continuation when it is a long match or agrees with the head's first draft
            if (!dn.empty() && (nm >= cfg.ng_strong || (!dm.empty() && dn[0] == dm[0]))) {
                d = dn;
                src = (!dm.empty() && dn[0] == dm[0]) ? 2 : 0;
            } else {
                d = dm;
                src = 1;
            }
        }
        st.t_draft += now_s() - td0;
        const int64_t pos0 = E.M.n_past[(size_t) slot];
        if (d.empty()) {
            const double t0 = now_s();
            E.decode(&out.back(), 1, slot, false);
            std::memcpy(last.data(), E.logits.data(), sizeof(float) * (size_t) nv);
            if (use_mtp) {
                // (h_last, x) was queued above but not run (no drafting this step): keep it queued, h_last = new row
                if (kmax > 0) { /* ran above */ }
                h_last.assign(E.hrows.begin(), E.hrows.begin() + ne);
            }
            st.plain_steps++;
            st.t_plain += now_s() - t0;
            st.t_by_T[1] += now_s() - t0;
            st.n_by_T[1]++;
            continue;
        }
        std::vector<int32_t> toks{x};
        toks.insert(toks.end(), d.begin(), d.end());
        const int T = (int) toks.size();
        const double t0 = now_s();
        E.decode(toks.data(), T, slot, true, true);
        st.t_verify += now_s() - t0;
        st.t_by_T[T] += now_s() - t0;
        st.n_by_T[T]++;
        st.steps++;
        st.drafted += T - 1;
        st.src_drafted[src] += T - 1;
        st.src_steps[src]++;
        // the MTP head's own first-draft accuracy (whatever source was verified)
        if (!dm.empty()) {
            st.mtp_first_total++;
            if (dm[0] == argmax(E.logits.data(), nv)) st.mtp_first_ok++;
        }
        int a = 1;
        bool stop = false;
        for (int i = 0; i < T - 1; ++i) {
            const float* Li = E.logits.data() + (size_t) i * nv;
            if (argmax(Li, nv) != d[(size_t) i]) break;
            on_row(Li, d[(size_t) i]);
            out.push_back(d[(size_t) i]);
            ctx_toks.push_back(d[(size_t) i]);
            if (t_tok) t_tok->push_back(now_s());
            a++;
            if ((stop_eog && eog.count(d[(size_t) i])) || (int) out.size() >= maxgen || (cancel && cancel())) {
                stop = true;
                break;
            }
        }
        st.accepted += a - 1;
        st.src_accepted[src] += a - 1;
        ema = 0.8 * ema + 0.2 * (double) (a - 1) / (double) (T - 1);
        st.acc_hist[a - 1]++;
        if (stop) {
            // the state keeps x and the accepted drafts (the rejected rows are rolled back), so a later request that
            // continues this session reads on from a consistent state
            E.spec_commit(slot, pos0, T, a);
            break;
        }
        std::memcpy(last.data(), E.logits.data() + (size_t) (a - 1) * nv, sizeof(float) * (size_t) nv);
        if (use_mtp) {
            // committed rows pos0 .. pos0+a-1: pairs (h_{pos0+i}, d_{i+1}) for i < a-1 wait for the next head call
            for (int i = 0; i + 1 < a; ++i) {
                pend_h.insert(pend_h.end(), E.hrows.begin() + (size_t) i * ne, E.hrows.begin() + (size_t) (i + 1) * ne);
                pend_t.push_back(d[(size_t) i]);
            }
            h_last.assign(E.hrows.begin() + (size_t) (a - 1) * ne, E.hrows.begin() + (size_t) a * ne);
        }
        const double tc = now_s();
        E.spec_commit(slot, pos0, T, a);
        st.t_commit += now_s() - tc;
    }
}

void print_spec_stats(const SpecStats& st, FILE* f) {
    std::fprintf(f, "[spec] verify steps %ld plain steps %ld drafted %ld accepted %ld (rate %.4f, mean accepted/verify %.4f); "
                    "time draft %.3f verify %.3f plain %.3f commit %.3f s\n", st.steps, st.plain_steps, st.drafted, st.accepted,
                 st.drafted ? (double) st.accepted / (double) st.drafted : 0.0, st.steps ? (double) st.accepted / (double) st.steps : 0.0,
                 st.t_draft, st.t_verify, st.t_plain, st.t_commit);
    const char* nm[3] = {"ngram", "mtp", "mixed"};
    for (int s = 0; s < 3; ++s)
        if (st.src_steps[s])
            std::fprintf(f, "[spec] source %s: steps %ld drafted %ld accepted %ld rate %.4f\n", nm[s], st.src_steps[s], st.src_drafted[s],
                         st.src_accepted[s], st.src_drafted[s] ? (double) st.src_accepted[s] / (double) st.src_drafted[s] : 0.0);
    for (auto& kv : st.acc_hist) std::fprintf(f, "[spec] accepted %d: %ld steps\n", kv.first, kv.second);
    if (st.gated_steps) std::fprintf(f, "[spec] steps without drafting (acceptance EMA below --spec-min-acc): %ld\n", st.gated_steps);
    if (st.mtp_calls)
        std::fprintf(f, "[spec] mtp: calls %ld rows %ld; first-draft agreement with the verified argmax %ld/%ld = %.4f\n", st.mtp_calls,
                     st.mtp_rows, st.mtp_first_ok, st.mtp_first_total,
                     st.mtp_first_total ? (double) st.mtp_first_ok / (double) st.mtp_first_total : 0.0);
    for (int T = 1; T <= NH_TMAX; ++T)
        if (st.n_by_T[T]) std::fprintf(f, "[spec] decode T=%d: %ld steps, mean %.3f ms\n", T, st.n_by_T[T], 1e3 * st.t_by_T[T] / (double) st.n_by_T[T]);
}

// ================================================================================================ serve
// The engine protocol of serve/server.py, as src/qwen35/q36_engine.cu speaks it (READY / INFO / GEN / BGEN / T / BT / PP /
// DONE / BDONE / BADM / STOP / BSTOP / BYIELD / ERR):
//   GEN <max_new> [key=value ...] <id,id,...>      a request alone on the control lines: PP lines while its prompt is
//                                                   read, a T line per token, then DONE.  Greedy requests decode with the
//                                                   speculative steps of --spec (verified, so the text is the greedy text)
//   BGEN <slot> <max_new> [keys] <ids>              (--batch N) the prompt is read into batch slot <slot>; its first token
//                                                   comes as T + DONE + BADM <slot> <0|1>, the rest as BT <slot> <id> lines,
//                                                   one per decode window over every decoding slot, until BDONE
//   STOP / BSTOP <slot>                             end the request on the control lines / a slot at its next window
// DONE <produced> <prompt_n> <prompt_ms> <decode_ms> <finish> <drafts_accepted> <drafts_offered> <reused> 0 0 0 0 0.0
//      <prompt_read> 0      (the 16 fields of server.py _parse_done)
struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

// (q36_engine.cu sample(): penalties over the last n tokens, then greedy, or top-k / top-p / min-p at the temperature)
int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& kv : cnt) {
            float& v = lg[(size_t) kv.first];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) kv.second * s.freq + s.present;
        }
    }
    if (s.temperature <= 0.f) return argmax(lg.data(), (int) lg.size());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[(size_t) a] > lg[(size_t) b]; });
    idx.resize((size_t) kk);
    std::vector<double> pr(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (pr[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (pr[i] / sum < s.min_p * (pr[0] / sum)) { keep = i; break; }
        cum += pr[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += pr[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i)
        if ((r -= pr[i]) <= 0) return idx[i];
    return idx[keep - 1];
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        while (*s == ',' || *s == ' ') ++s;
        if (!*s) break;
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t) x);
        s = e;
    }
    return v;
}

struct Serve {
    Engine& E;
    SpecCfg spec;
    int chunk;
    std::set<int> eog;
    int nv;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    struct Slot {
        bool active = false, stop = false;
        int32_t x = 0;
        int64_t produced = 0, max_new = 0;
        Sampling smp;
        std::mt19937_64 rng;
        std::vector<int32_t> hist;
        double t0 = 0;
    };
    struct Sess {
        std::vector<int32_t> live;   // the tokens the slot's state holds (live.size() == E.M.n_past[slot])
        bool active = false;         // a batch slot decoding
        uint64_t used = 0;
    };
    std::vector<Slot> slots;
    std::vector<Sess> sess;
    int n_active = 0;
    uint64_t clock = 0;
    std::vector<float> lg;
    long long windows = 0, window_rows = 0;
    double window_s = 0;
    int read_windows = 1;

    Serve(Engine& e, const SpecCfg& sc, int ch, const std::set<int>& eg)
        : E(e), spec(sc), chunk(ch), eog(eg), nv(e.W.g.n_vocab), slots((size_t) e.M.n_slots), sess((size_t) e.M.n_slots) {
        if (const char* v = std::getenv("STRATA_NH_READ_WINDOWS")) read_windows = std::max(0, std::min(64, std::atoi(v)));
    }
    bool is_eos(int t) const { return eog.count(t) > 0; }
    static double ms_since(double t0) { return (now_s() - t0) * 1e3; }
    int n_sess() const { return E.M.n_slots; }

    void poll_control() {
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = lines.begin(); it != lines.end();) {
            if (it->rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(it->c_str() + 6);
                if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
                it = lines.erase(it);
            } else if (it->rfind("BYIELD", 0) == 0) {
                it = lines.erase(it);
            } else {
                ++it;
            }
        }
    }

    // one decode window: the next token of every decoding batch slot (one graph, rows mode)
    void window() {
        poll_control();
        std::vector<int> sl;
        std::vector<int32_t> tk;
        for (int b = 0; b < n_sess(); ++b)
            if (slots[(size_t) b].active) {
                sl.push_back(b);
                tk.push_back(slots[(size_t) b].x);
            }
        if (sl.empty()) return;
        const double t0 = now_s();
        if (sl.size() == 1) E.decode(tk.data(), 1, sl[0], false);
        else E.decode_rows(sl.data(), tk.data(), (int) sl.size());
        window_s += now_s() - t0;
        ++windows;
        window_rows += (long long) sl.size();
        for (size_t r = 0; r < sl.size(); ++r) {
            const int b = sl[r];
            Slot& st = slots[(size_t) b];
            Sess& ss = sess[(size_t) b];
            ss.live.push_back(st.x);
            int y;
            if (st.smp.needs_logits()) {
                lg.assign(E.logits.begin() + (long) (r * (size_t) nv), E.logits.begin() + (long) ((r + 1) * (size_t) nv));
                y = sample(lg, st.smp, st.hist, st.rng);
            } else {
                y = argmax(E.logits.data() + r * (size_t) nv, nv);
            }
            std::printf("BT %d %d\n", b, y);
            ++st.produced;
            st.hist.push_back(y);
            const char* fin = is_eos(y) ? "stop" : st.stop ? "cancel" : st.produced >= st.max_new ? "length"
                              : E.M.n_past[(size_t) b] + 1 > E.M.n_ctx ? "length" : nullptr;
            if (fin) {
                std::printf("BDONE %d %lld %s %.1f\n", b, (long long) st.produced, fin, ms_since(st.t0));
                st.active = false;
                ss.active = false;
                --n_active;
            } else {
                st.x = y;
            }
        }
        std::fflush(stdout);
    }

    size_t match_len(const Sess& ss, const std::vector<int32_t>& ids) const {
        const size_t L = ss.live.size();
        if (L == 0 || L >= ids.size() || !std::equal(ss.live.begin(), ss.live.end(), ids.begin())) return 0;
        return L;
    }

    int pick_session(const std::vector<int32_t>& ids) const {
        int best = -1, free_empty = -1, lru = -1;
        size_t bl = 0;
        for (int i = 0; i < n_sess(); ++i) {
            const Sess& ss = sess[(size_t) i];
            if (ss.active) continue;
            const size_t l = match_len(ss, ids);
            if (l > bl) { bl = l; best = i; }
            if (ss.live.empty() && free_empty < 0) free_empty = i;
            if (lru < 0 || ss.used < sess[(size_t) lru].used) lru = i;
        }
        return best >= 0 ? best : free_empty >= 0 ? free_empty : lru;
    }

    void err(const std::string& m) {
        std::printf("ERR %s\n", m.c_str());
        std::fflush(stdout);
    }

    void request(const std::string& line) {
        const bool adm = line[0] == 'B';
        const bool geni = line.rfind(adm ? "BGENI " : "GENI ", 0) == 0;
        if (!adm && !geni && line.rfind("GEN ", 0) != 0) {
            err("expected: GEN <max_new> [key=value ...] <id,id,...>");
            return;
        }
        if (geni) { err("this engine was started without --vision"); return; }
        char* p = const_cast<char*>(line.c_str()) + (adm ? 5 : 4);
        int slot = -1;
        if (adm) {
            char* e1 = nullptr;
            const long b = std::strtol(p, &e1, 10);
            if (e1 == p || b < 0 || b >= (long) slots.size() || slots[(size_t) b].active || n_sess() < 2) {
                std::printf("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", n_sess() < 2 ? 0 : n_sess());
                std::fflush(stdout);
                return;
            }
            p = e1;
            slot = (int) b;
        }
        const long long max_new = std::strtoll(p, &p, 10);
        Sampling smp;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
            // other keys of server.py (ckpt, pin, cvec, img_keys, logit bias, tuning) are not used by this engine
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty()) { err("bad request: max_new or the ids"); return; }
        if ((int64_t) ids.size() + max_new > E.M.n_ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, E.M.n_ctx);
            std::fflush(stdout);
            return;
        }
        for (int32_t t : ids)
            if (t < 0 || t >= nv) { err("a token id is outside the vocabulary"); return; }
        stop.store(false);
        const int s = adm ? slot : pick_session(ids);
        Sess& ss = sess[(size_t) s];
        size_t reuse = match_len(ss, ids);
        if (adm) {   // another idle session holding more of this prompt: copy its state
            int src = -1;
            size_t best = reuse;
            for (int i = 0; i < n_sess(); ++i) {
                if (i == s || sess[(size_t) i].active) continue;
                const size_t l = match_len(sess[(size_t) i], ids);
                if (l > best) { best = l; src = i; }
            }
            if (src >= 0) {
                E.copy_session(src, s);
                ss.live = sess[(size_t) src].live;
                reuse = best;
            }
        }
        if (reuse == 0) {
            E.clear_slot(s);
            ss.live.clear();
        }
        ss.used = ++clock;
        const double t0 = now_s();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        bool cancelled = false;
        int64_t done = 0;
        // the prompt in --chunk ubatches from the first fresh token (a fresh prompt: the CLI's chunks exactly)
        for (size_t off = reuse; off < ids.size();) {
            const int n = (int) std::min<size_t>((size_t) chunk, ids.size() - off);
            E.decode(ids.data() + off, n, s, false);
            ss.live.insert(ss.live.end(), ids.begin() + (long) off, ids.begin() + (long) off + n);
            off += (size_t) n;
            done += n;
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) fresh, ms_since(t0),
                        (double) done / std::max(1e-9, ms_since(t0) / 1000.0));
            std::fflush(stdout);
            poll_control();
            if (done < fresh) {
                for (int w = 0; w < read_windows && n_active > 0; ++w) window();
                if (stop.load()) { cancelled = true; break; }
            }
        }
        const double prompt_ms = ms_since(t0);
        if (cancelled) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms, reuse, (long long) done);
            if (adm) std::printf("BADM %d 0\n", slot);
            std::fflush(stdout);
            return;
        }
        std::vector<float> last(E.logits.begin(), E.logits.begin() + nv);
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist = ids;
        const double d0 = now_s();
        if (adm) {
            int nx;
            if (smp.needs_logits()) nx = sample(last, smp, hist, rng);
            else nx = argmax(last.data(), nv);
            std::printf("T %d\n", nx);
            hist.push_back(nx);
            const char* fin = is_eos(nx) ? "stop" : stop.load() ? "cancel" : max_new <= 1 ? "length"
                              : E.M.n_past[(size_t) s] + 1 > E.M.n_ctx ? "length" : nullptr;
            const bool cont = fin == nullptr;
            std::printf("DONE 1 %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms, ms_since(d0),
                        cont ? "length" : fin, reuse, (long long) fresh);
            if (cont) {
                Slot& st = slots[(size_t) slot];
                st = Slot{};
                st.active = true;
                st.x = nx;
                st.produced = 1;
                st.max_new = max_new;
                st.smp = smp;
                st.rng = rng;
                st.hist = hist;
                st.t0 = now_s();
                ss.active = true;
                ++n_active;
            }
            std::printf("BADM %d %d\n", slot, cont ? 1 : 0);
            std::fflush(stdout);
            std::fprintf(stderr, "[nh_serve] slot %d: prompt %zu tokens = %zu reused + %lld read in %.0f ms (%.1f tok/s)%s\n", slot,
                         ids.size(), reuse, (long long) fresh, prompt_ms, prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0,
                         cont ? "" : ", done at once");
            return;
        }
        long long produced = 0, offered = 0, accepted = 0;
        const char* finish = "length";
        std::vector<int32_t> out;
        if (!smp.needs_logits() && spec.mode) {
            // greedy: the speculative steps of the CLI (spec_generate); every token is the plain greedy token
            SpecStats sst;
            auto emit = [&](const float*, int t) {
                std::printf("T %d\n", t);
                std::fflush(stdout);
                ++produced;
            };
            const std::function<bool()> cancel = [&]() { return stop.load(); };
            const std::function<bool()> between = [&]() {
                if (n_active == 0) return false;
                window();
                return true;
            };
            spec_generate(E, spec, sst, ids, last, (int) max_new, true, eog, out, emit, nullptr, s, cancel, between);
            offered = sst.drafted;
            accepted = sst.accepted;
            finish = !out.empty() && is_eos(out.back()) ? "stop" : (long long) out.size() >= max_new ? "length" : "cancel";
        } else {
            for (;;) {
                int nx;
                if (smp.needs_logits()) nx = sample(last, smp, hist, rng);
                else nx = argmax(last.data(), nv);
                std::printf("T %d\n", nx);
                std::fflush(stdout);
                ++produced;
                hist.push_back(nx);
                out.push_back(nx);
                if (is_eos(nx)) { finish = "stop"; break; }
                if (produced >= max_new) break;
                if (stop.load()) { finish = "cancel"; break; }
                if (n_active > 0) window();
                if (E.M.n_past[(size_t) s] + 1 > E.M.n_ctx) break;
                E.decode(&nx, 1, s, false);
                last.assign(E.logits.begin(), E.logits.begin() + nv);
            }
        }
        // the state holds the prompt and the generated tokens up to n_past (the last emitted one is not read yet)
        {
            std::vector<int32_t> all = ids;
            all.insert(all.end(), out.begin(), out.end());
            const size_t np = (size_t) E.M.n_past[(size_t) s];
            if (np > all.size()) throw std::runtime_error("nh_serve: session state longer than its tokens");
            all.resize(np);
            ss.live = std::move(all);
        }
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms, decode_ms,
                    finish, accepted, offered, reuse, (long long) fresh);
        std::fflush(stdout);
        std::fprintf(stderr, "[nh_serve] session %d: prompt %zu tokens = %zu reused + %lld read in %.0f ms (%.1f tok/s), %lld generated "
                             "in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld (%s)\n", s, ids.size(), reuse, (long long) fresh,
                     prompt_ms, prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, produced, decode_ms,
                     decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0, accepted, offered, finish);
    }

    int run() {
        std::printf("INFO engine=strata-nh-engine arch=nemotron_h_moe kv_slots=%d spec=%d expert_cache_mib=%.0f\n", n_sess(), spec.mode,
                    E.C.pool_bytes / 1048576.0);
        if (n_sess() > 1) std::printf("INFO batch_slots=%d slot_cache=1\n", n_sess());
        std::printf("READY %d stop\n", E.M.n_ctx);
        std::fflush(stdout);
        std::thread reader([&] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop.store(true); continue; }
                std::lock_guard<std::mutex> lk(mu);
                lines.push_back(l);
                cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        });
        reader.detach();
        for (;;) {
            std::string line;
            bool have = false;
            {
                std::unique_lock<std::mutex> lk(mu);
                if (n_active == 0) cv.wait(lk, [&] { return eof || !lines.empty(); });
                if (!lines.empty()) {
                    line = lines.front();
                    lines.pop_front();
                    have = true;
                } else if (n_active == 0) {
                    break;
                }
            }
            if (!have) { window(); continue; }
            if (line.empty()) continue;
            if (line.rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(line.c_str() + 6);
                if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
            } else if (line.rfind("BYIELD", 0) == 0) {
            } else if (line.rfind("GEN", 0) == 0 || line.rfind("BGEN", 0) == 0) {
                request(line);
            } else {
                err("expected: GEN <max_new> [key=value ...] <id,id,...>");
            }
            if (n_active > 0) window();
        }
        std::fprintf(stderr, "[nh_serve] input closed; %lld batch windows (%lld rows, %.1f s)\n", windows, window_rows, window_s);
        E.cache_stats(stderr);
        return 0;
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

int real_main(int argc, char** argv) {
    std::string model, list, tokdir, outdir, force, only, dump_sums, dump_prompt, bench, bench_out;
    int profile = 0;
    bool pin = false;
    int cache_mode = 0, cap_lo = 0, cap_hi = 0, max_admit = 2;
    long cache_mib = -1, vram_reserve_mib = 2048, vram_limit_mib = 0;
    double cache_decay = 0.97;
    SpecCfg spec;
    bool spec_check = false;
    int spec_diff = 0;
    int mtp_topk = 0;
    int K = 32, chunk = 512, ctx = 32768, threads = 16, n_cpu_moe = 68;
    bool serve_mode = false;
    int batch = 0;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--serve") serve_mode = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--model" || a == "--native") model = nx();
        else if (a == "--serve") serve_mode = true;
        else if (a == "--batch" || a == "--slots") batch = std::stoi(nx());
        else if (a == "--max-context") ctx = std::stoi(nx());
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
        else if (a == "--bench") bench = nx();
        else if (a == "--bench-out") bench_out = nx();
        else if (a == "--profile") profile = std::stoi(nx());
        else if (a == "--pin") pin = true;
        else if (a == "--cache") {
            const std::string v = nx();
            cache_mode = v == "lo" ? 1 : v == "all" ? 2 : v == "off" ? 0 : -1;
            if (cache_mode < 0) { std::fprintf(stderr, "--cache lo|all|off\n"); return 2; }
        }
        else if (a == "--cache-mib") cache_mib = std::stol(nx());
        else if (a == "--vram-reserve-mib") vram_reserve_mib = std::stol(nx());
        else if (a == "--vram-limit-mib") vram_limit_mib = std::stol(nx());
        else if (a == "--cap-lo") cap_lo = std::stoi(nx());
        else if (a == "--cap-hi") cap_hi = std::stoi(nx());
        else if (a == "--cache-decay") cache_decay = std::stod(nx());
        else if (a == "--max-admit") max_admit = std::stoi(nx());
        else if (a == "--spec") {
            const std::string v = nx();
            spec.mode = v == "off" ? 0 : v == "ngram" ? 1 : v == "mtp" ? 2 : v == "auto" ? 3 : -1;
            if (spec.mode < 0) { std::fprintf(stderr, "--spec off|ngram|mtp|auto\n"); return 2; }
        }
        else if (a == "--spec-k") spec.k = std::stoi(nx());
        else if (a == "--mtp-topk") mtp_topk = std::stoi(nx());
        else if (a == "--ng-strong") spec.ng_strong = std::stoi(nx());
        else if (a == "--spec-min-acc") spec.min_acc = std::stod(nx());
        else if (a == "--spec-check") spec_check = true;
        else if (a == "--spec-diff") spec_diff = std::stoi(nx());
        else if (serve_mode && a.rfind("--", 0) == 0) {
            // server.py passes the config's args plus a few of its own: one this engine does not have is said and
            // skipped (with its value), not fatal (as q36_engine.cu)
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "[nh_engine] ignoring %s%s%s (not an option of this engine)\n", a.c_str(), val ? " " : "", val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (model.empty() || ((list.empty() || tokdir.empty() || outdir.empty()) && bench.empty() && !serve_mode)) {
        std::fprintf(stderr, "usage: strata-nh-engine --model SHARD --list LIST --tokdir DIR --out RAWDIR [--k 32] "
                             "[--chunk 512] [--ctx 32768] [--threads 16] [--n-cpu-moe 68] [--force RAWDIR] [--only ids] "
                             "[--dump-sums FILE --dump-prompt ID]\n       strata-nh-engine --model SHARD --bench FILE --bench-out JSONL "
                             "[--profile NTOK] [--chunk 512] [--ctx 32768] [--threads 16] [--n-cpu-moe 68]\n"
                             "       either: [--pin] [--cache lo|all|off] [--cache-mib N] [--cap-lo N] [--cap-hi N] "
                             "[--cache-decay 0.97] [--max-admit 2] [--vram-reserve-mib 2048] [--vram-limit-mib N]\n"
                             "       [--spec off|ngram|mtp|auto] [--spec-k 1..3] [--spec-min-acc X] [--mtp-topk N] [--spec-check]\n"
                             "       strata-nh-engine --serve --model SHARD [--batch 2..4] [--ctx N] [the cache/spec options]\n");
        return 2;
    }
    const std::set<std::string> only_ids = split_ids(only);
    const double t_load0 = now_s();
    const strata::GgufModel gm = strata::GgufModel::open(model);
    const nh::Model m = nh::load_model(gm);
    if (!m.report.pass()) {
        // review H1: a GGUF whose tensors do not match the expected layout (missing, wrong shape/type, payload outside
        // the file) is refused before anything reads it
        for (const auto& e : m.report.errors) std::fprintf(stderr, "[nh_engine] layout: %s\n", e.c_str());
        std::fprintf(stderr, "[nh_engine] GGUF layout check failed (%d expected, %d found, %d ok, %d in file); refusing to run\n",
                     m.report.expected, m.report.found, m.report.ok, m.report.in_file);
        return 3;
    }
    Engine E;
    E.pin = pin;
    E.cache_mib = cache_mib;
    E.vram_reserve_mib = vram_reserve_mib;
    E.vram_limit_mib = vram_limit_mib;
    E.cap_lo = cap_lo;
    E.cap_hi = cap_hi;
    E.C.decay = cache_decay;
    E.C.max_admit = max_admit;
    // rollback snapshots: a verify batch has at most spec.k + 1 rows
    if (spec.mode) E.k_snap = std::min(NH_TMAX, spec.k + 1);
    if (batch < 0 || batch > NH_TMAX || batch == 1) throw std::runtime_error("--batch must be 2.." + std::to_string(NH_TMAX) + " (or 0: one request at a time)");
    const int n_slots = batch >= 2 ? batch : 1;
    E.rows_max = n_slots;
    E.mtp_on = spec.mode == 2 || spec.mode == 3;
    E.mtp_topk = mtp_topk;
    if (spec.k < 1 || spec.k > NH_TMAX - 1) throw std::runtime_error("--spec-k must be 1.." + std::to_string(NH_TMAX - 1));
    E.init(m, n_cpu_moe, n_slots, ctx, threads, chunk, cache_mode);
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
    if (serve_mode) {
        Serve sv(E, spec, chunk, eog);
        return sv.run();
    }
    if (!bench.empty()) {
        // Gate 3 bench: each line "name n_predict ignore_eos reps n_tok tok...". Timing as llama-server
        // (server-common.h:405-480): prompt time = start -> first token sampled; decode tok/s = (n_gen - 1) / (t_last_token
        // - t_first_token); prefill in ubatches of --chunk with logits for the last token only.
        std::ifstream bf(bench);
        FILE* bo = std::fopen(bench_out.c_str(), "w");
        if (!bf || !bo) throw std::runtime_error("cannot open bench files");
        std::string name;
        int npred, igeos, reps, ntok;
        SpecStats sst;
        long spec_check_bad = 0;
        while (bf >> name >> npred >> igeos >> reps >> ntok) {
            if (ntok <= 0 || ntok >= ctx || npred < 1 || reps < 1) throw std::runtime_error("bench line " + name + ": bad n_tok/n_predict/reps");
            std::vector<int32_t> pt((size_t) ntok);
            for (int i = 0; i < ntok; ++i)
                if (!(bf >> pt[(size_t) i])) throw std::runtime_error("bench line " + name + ": fewer tokens than n_tok");
            if (spec_diff > 0) {
                // T plain steps (greedy chain) vs one verify batch of the same T tokens, node by node
                const int T = spec_diff;
                E.clear_slot(0);
                for (int off = 0; off < ntok; off += chunk) E.decode(pt.data() + off, std::min(chunk, ntok - off), 0, false);
                std::vector<int32_t> xs{argmax(E.logits.data(), nv)};
                std::vector<std::vector<float>> plain_logits;
                E.diff.mode = 1;
                for (int t = 0; t < T; ++t) {
                    E.decode(&xs.back(), 1, 0, false);
                    plain_logits.emplace_back(E.logits.begin(), E.logits.begin() + nv);
                    if (t + 1 < T) xs.push_back(argmax(E.logits.data(), nv));
                }
                E.diff.mode = 0;
                E.clear_slot(0);
                for (int off = 0; off < ntok; off += chunk) E.decode(pt.data() + off, std::min(chunk, ntok - off), 0, false);
                E.diff.mode = 2;
                E.diff.T = T;
                E.decode(xs.data(), T, 0, true, true);
                E.diff.mode = 0;
                int n_cmp = 0, n_bad = 0;
                for (const auto& nm : E.diff.order) {
                    const auto& p = E.diff.P[nm];
                    auto it = E.diff.V.find(nm);
                    if (it == E.diff.V.end() || it->second.size() != (size_t) T || p.size() != (size_t) T) {
                        std::fprintf(stderr, "[spec-diff] %-34s plain %s x%zu, verify %s: not comparable\n", nm.c_str(),
                                     E.diff.shape_p[nm].c_str(), p.size(), it == E.diff.V.end() ? "-" : E.diff.shape_v[nm].c_str());
                        continue;
                    }
                    n_cmp++;
                    std::string bad;
                    for (int t = 0; t < T; ++t)
                        if (p[(size_t) t] != it->second[(size_t) t]) bad += " t" + std::to_string(t);
                    if (!bad.empty()) {
                        n_bad++;
                        if (n_bad <= 60)
                            std::fprintf(stderr, "[spec-diff] %-34s plain %s verify %s MISMATCH%s\n", nm.c_str(), E.diff.shape_p[nm].c_str(),
                                         E.diff.shape_v[nm].c_str(), bad.c_str());
                    }
                }
                int lbad = 0;
                for (int t = 0; t < T; ++t)
                    if (std::memcmp(plain_logits[(size_t) t].data(), E.logits.data() + (size_t) t * nv, sizeof(float) * (size_t) nv)) lbad++;
                std::fprintf(stderr, "[spec-diff] T=%d: %d named nodes compared, %d mismatching; logits rows differing %d/%d; "
                                     "verify-only nodes %zu\n", T, n_cmp, n_bad, lbad, T, E.diff.V.size() - (size_t) n_cmp);
                return lbad ? 1 : 0;
            }
            for (int rep = 0; rep < reps; ++rep) {
                E.clear_slot(0);
                const double t0 = now_s();
                for (int off = 0; off < ntok; off += chunk) E.decode(pt.data() + off, std::min(chunk, ntok - off), 0, false);
                std::vector<int32_t> out{argmax(E.logits.data(), nv)};
                double t1 = now_s();
                double t2 = t1, t_first_tok = t1;
                const bool prof_run = profile > 0 && rep == 0;
                if (spec.mode && !prof_run) {
                    // the first token was sampled above; spec_generate re-emits it from `last`, so start from the prompt
                    out.clear();
                    std::vector<float> last(E.logits.begin(), E.logits.begin() + nv);
                    std::vector<double> tt;
                    std::vector<uint64_t> hs;
                    auto hrow = [&](const float* lg, int) {
                        uint64_t h = 1469598103934665603ull;
                        const uint8_t* b = (const uint8_t*) lg;
                        for (size_t i = 0; i < (size_t) nv * 4; ++i) h = (h ^ b[i]) * 1099511628211ull;
                        hs.push_back(h);
                    };
                    if (spec_check && rep == 0) {
                        // reference: the plain T == 1 loop from the same prompt state, then the prompt again for the spec run
                        SpecCfg plain;
                        SpecStats pst;
                        std::vector<int32_t> pout;
                        spec_generate(E, plain, pst, pt, last, npred, !igeos, eog, pout, hrow, nullptr);
                        std::vector<uint64_t> phs = hs;
                        hs.clear();
                        E.clear_slot(0);
                        for (int off = 0; off < ntok; off += chunk) E.decode(pt.data() + off, std::min(chunk, ntok - off), 0, false);
                        spec_generate(E, spec, sst, pt, last, npred, !igeos, eog, out, hrow, &tt);
                        int bad = 0, first_bad = -1;
                        for (size_t i = 0; i < std::max(phs.size(), hs.size()); ++i)
                            if (i >= phs.size() || i >= hs.size() || phs[i] != hs[i]) {
                                bad++;
                                if (first_bad < 0) first_bad = (int) i;
                            }
                        std::fprintf(stderr, "[spec-check] %s: plain %zu tokens, spec %zu tokens, tokens %s, logits rows bit-identical "
                                             "%zu/%zu (first diff %d)\n", name.c_str(), pout.size(), out.size(),
                                     pout == out ? "identical" : "DIFFER", std::max(phs.size(), hs.size()) - (size_t) bad,
                                     std::max(phs.size(), hs.size()), first_bad);
                        spec_check_bad += bad + (pout == out ? 0 : 1);
                    } else {
                        spec_generate(E, spec, sst, pt, last, npred, !igeos, eog, out, hrow, &tt);
                    }
                    t2 = tt.empty() ? t1 : tt.back();
                    t_first_tok = tt.empty() ? t1 : tt.front();
                }
                while (!spec.mode && (int) out.size() < npred && (igeos || !eog.count(out.back()))) {
                    if (prof_run && (int) out.size() == 4) E.prof.on = true;
                    E.decode(&out.back(), 1, 0, false);
                    out.push_back(argmax(E.logits.data(), nv));
                    t2 = now_s();
                    if (prof_run && (int) out.size() == 4 + profile) break;
                }
                if (prof_run) {
                    E.prof.on = false;
                    std::vector<std::pair<double, std::string>> v;
                    double tot = 0;
                    for (auto& kv : E.prof.agg) {
                        v.push_back({kv.second.first, kv.first + "\t" + std::to_string(kv.second.second)});
                        tot += kv.second.first;
                    }
                    std::sort(v.rbegin(), v.rend());
                    std::fprintf(stderr, "[profile] %s: %d tokens, %.3f ms/token in callbacks\n", name.c_str(), profile, tot * 1e3 / profile);
                    for (auto& x : v) std::fprintf(stderr, "[profile] %9.3f ms/tok\t%s\n", x.first * 1e3 / profile, x.second.c_str());
                    E.prof.agg.clear();
                    std::fprintf(stderr, "[profile] %s rep %d: profile run, no bench JSONL row written\n", name.c_str(), rep);
                    continue;
                }
                const int ng = (int) out.size();
                if (spec.mode) {
                    // llama-server's timing: prompt time ends when the first token is sampled
                    t1 = t_first_tok;
                }
                const double pms = (t1 - t0) * 1e3, gms = (t2 - t1) * 1e3;
                std::fprintf(bo, "{\"name\": \"%s\", \"rep\": %d, \"prompt_n\": %d, \"prompt_ms\": %.3f, \"pp_tok_s\": %.4f, "
                                 "\"predicted_n\": %d, \"predicted_ms\": %.3f, \"decode_tok_s\": %.4f, \"ttft_s\": %.4f, \"tokens\": [",
                             name.c_str(), rep, ntok, pms, ntok / (t1 - t0), ng, gms, ng > 1 ? (ng - 1) / (t2 - t1) : 0.0, t1 - t0);
                for (int i = 0; i < ng; ++i) std::fprintf(bo, "%s%d", i ? ", " : "", out[(size_t) i]);
                std::fprintf(bo, "]}\n");
                std::fflush(bo);
                std::fprintf(stderr, "[bench] %s rep %d prompt_n=%d pp=%.2f tok/s ttft=%.3f s gen=%d decode=%.3f tok/s\n",
                             name.c_str(), rep, ntok, ntok / (t1 - t0), t1 - t0, ng, ng > 1 ? (ng - 1) / (t2 - t1) : 0.0);
            }
        }
        std::fclose(bo);
        E.cache_stats(stderr);
        if (spec.mode) print_spec_stats(sst, stderr);
        if (spec_check) std::fprintf(stderr, "[spec-check] total mismatches %ld -> %s\n", spec_check_bad, spec_check_bad ? "FAIL" : "PASS");
        return 0;
    }
    FILE* sumf = std::fopen((outdir + "/capture_log.txt").c_str(), "w");
    if (!sumf) throw std::runtime_error("cannot write " + outdir + "/capture_log.txt");
    std::fprintf(sumf, "engine=strata-nh-engine n_vocab=%d n_ctx=%d K=%d chunk=%d force=%s n_cpu_moe=%d threads=%d\n", nv,
                 ctx, K, chunk, force.c_str(), n_cpu_moe, threads);
    std::ifstream lf(list);
    std::string id;
    int ngen;
    const unsigned nthr = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    SpecStats cap_sst;
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
        if (!dump_sums.empty() && id == dump_prompt && !E.dump.f) throw std::runtime_error("cannot write " + dump_sums);
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
        if (spec.mode && force.empty()) {
            spec_generate(E, spec, cap_sst, pt, last, maxgen, true, eog, out, [&](const float* lg, int t) {
                rows.emplace_back();
                make_row(lg, nv, K, t, rows.back());
            }, nullptr);
            is_eog = !out.empty() && eog.count(out.back());
        }
        for (int j = 0; j < maxgen && !(spec.mode && force.empty()); ++j) {
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
        if (!fi || !fl || !fs || !ft) throw std::runtime_error("cannot write capture files for " + id + " in " + outdir);
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
        if (!fo) throw std::runtime_error("cannot write " + outdir + "/" + id + ".out.i32");
        std::fwrite(out.data(), 4, out.size(), fo);
        std::fclose(fo);
        std::fprintf(sumf, "%s n_prompt=%d n_out=%zu rows=%zu eog=%d prompt_s=%.3f gen_s=%.3f\n", id.c_str(), np,
                     out.size(), rows.size(), is_eog ? 1 : 0, t1 - t0, t2 - t1);
        std::fflush(sumf);
        std::fprintf(stderr, "[nh_engine] %s n_prompt=%d n_out=%zu eog=%d prompt_s=%.1f gen_s=%.1f\n", id.c_str(), np,
                     out.size(), is_eog ? 1 : 0, t1 - t0, t2 - t1);
    }
    std::fclose(sumf);
    E.cache_stats(stderr);
    if (spec.mode) print_spec_stats(cap_sst, stderr);
    return 0;
}

// review M1: report every error with its message and a non-zero exit instead of std::terminate
int main(int argc, char** argv) {
    try {
        return real_main(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-nh-engine: error: %s\n", e.what());
        return 1;
    }
}
