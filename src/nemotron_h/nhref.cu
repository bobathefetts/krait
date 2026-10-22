// src/nemotron_h/nhref.cu - strata-nhref: the reference path for NVIDIA Nemotron 3.5 Super (GGUF architecture
// nemotron_h_moe), correctness first.  One GPU; the dense weights resident in VRAM in their GGUF form, the routed
// experts read from the GGUF mapping (mmap) on use into a VRAM slot cache; every matrix dequantised to fp32 with
// ggml's block formulas and multiplied with cuBLAS SGEMM (pedantic math: no TF32); fp32 activations, fp32 KV cache,
// fp32 recurrent states.  Prompts run in chunks (all positions' logits kept), generation one token at a time.
//
// The forward pass, as llama.cpp b11806 (0682a576a772de436caa5d1c1e94036486d6ebd0) builds it:
//   x = token_embd[tok]                                                  (no BOS: llama-vocab.cpp:2918)
//   per block l < 88 (src/models/nemotron-h.cpp:218-248):  h = rms(x) * attn_norm (eps 1e-5);  x += mixer(h)
//     Mamba-2 (40 blocks; src/models/mamba-base.cpp:153-305)
//       zxBCdt = W_in h (18560) -> z (8192) | xBC (10240) | dt (128)                       mamba-base.cpp:192-207
//       xBC = silu(conv4(concat(conv_state, xBC)) + b); conv_state = last 3 inputs        :210-242 (ggml_ssm_conv,
//                                                                                          ggml-cpu/ops.cpp:10022)
//       x = xBC[0:8192] (128 heads x 64), B = xBC[8192 + g*128], C = xBC[9216 + g*128] (8 groups, head h -> h/16)
//       dt += dt_bias; per head: sp = softplus(dt) (x if dt > 20), dA = exp(sp * A_h);      :251-264, ops.cpp:10151-10157
//         S = S * dA + B (x * sp);  y = S . C                                               (ggml_ssm_scan, ops.cpp:10092)
//       y += x * D_h; y = silu(z) * y; y = rms(y per group of 1024) * ssm_norm; W_out y     :286-299
//     attention (8 blocks: 7 16 25 36 47 58 69 78; nemotron-h.cpp:276-292): q = W_q h (32 x 128), k, v = W_k h,
//       W_v h (2 x 128); NO RoPE (llama-model.cpp:3497-3503); o = softmax(q k^T / sqrt(128) + causal) v, GQA h -> h/16;
//       W_o o
//     MoE (40 blocks; nemotron-h.cpp:294-345, llama-graph.cpp:2236-2550)
//       p = sigmoid(W_router h) (512, fp32 weights)                                        llama-graph.cpp:2247
//       top-22 of p + exp_probs_b (selection only)                                         :2266, :2309
//       w = p[sel] / max(sum p[sel], 6.103515625e-5) * 5.0                                 :2323, :2337-2350
//       a = W_latent_down h (1024);  routed = W_latent_up sum_k w_k W_down_k relu(W_up_k a)^2
//                                                     (sum in selection order)            nemotron-h.cpp:304-334
//       shared = W_down_s relu(W_up_s h)^2 (full hidden, no gate)                         nemotron-h.cpp:336-344
//       mixer = routed + shared
//   logits = W_output (rms(x) * output_norm)                                              nemotron-h.cpp:259-275
// The MTP block (blk.88) is loaded and checked by the layout code but not run.
//
// Output: the golden-set capture format of the port's bench/harness/FORMAT.md (raw form): per prompt
// <id>.out.i32, .topk_ids.i32, .topk_logits.f32, .lse.f32, .target_lp.f32 and capture_log.txt, with the same row
// definition, greedy rule (argmax, ties to the lowest id), EOG rule ({11 <|im_end|>, 2 </s>}: llama-vocab.cpp:3172-3202,
// 3233; kept, not fed back) and teacher-forced mode as the llama.cpp golden-set capture tool.
//
// Usage: strata-nhref --model SHARD --list capture_list.txt --tokdir DIR --out RAWDIR [--k 32] [--chunk 512]
//                     [--ctx 32768] [--force RAWDIR] [--only id,id] [--expert-cache-mib 12000]
//                     [--dump-hidden FILE --dump-prompt ID]   (x after every block, every prompt position, f32)
//                     [--route-log FILE]                       (per MoE call: layer, T, selected experts)
//                     [--emulate-act-quant]   diagnostic: round matmul inputs as llama.cpp's q8_1 path and Q/K/V to f16
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/nemotron_h/layout.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
#include <string>
#include <thread>
#include <vector>

namespace {

namespace nh = strata::nemotron_h;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "nhref: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__);   \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)
#define CB(x)                                                                                                          \
    do {                                                                                                               \
        cublasStatus_t s_ = (x);                                                                                       \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                                                             \
            std::fprintf(stderr, "nhref: %s failed: %d (%s:%d)\n", #x, (int) s_, __FILE__, __LINE__);                 \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

// ------------------------------------------------------------------------------------------------ dequantisation
// ggml's block formats, 32 values at a time (chunk c of a row); the formulas are ggml/src/ggml-quants.c's
// dequantize_row_* (q8_0, q5_0 :500, q4_K, q5_K, q6_K, iq4_nl :2726), the K-quant ones as in src/qwen35/q36ref.cu.
enum : int { T_F32 = 0, T_F16 = 1, T_Q5_0 = 6, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_IQ4_NL = 20, T_BF16 = 30 };

__constant__ int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};  // ggml-common.h:1120

__device__ __forceinline__ float h2f(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) {
        const float v = ldexpf((float) m, -24);
        return s ? -v : v;
    }
    const uint32_t f = e == 31 ? (s | 0x7f800000u | (m << 13)) : (s | ((e + 112) << 23) | (m << 13));
    return __uint_as_float(f);
}
__device__ __forceinline__ uint16_t ld16(const uint8_t* p) { return (uint16_t) (p[0] | (p[1] << 8)); }

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

template <int T> __device__ void deq32(const uint8_t* row, int c, float* out) {
    if constexpr (T == T_F32) {
        const float* r = (const float*) row + 32 * c;
        for (int l = 0; l < 32; ++l) out[l] = r[l];
    } else if constexpr (T == T_F16) {
        const uint8_t* r = row + 64 * c;
        for (int l = 0; l < 32; ++l) out[l] = h2f(ld16(r + 2 * l));
    } else if constexpr (T == T_BF16) {
        const uint8_t* r = row + 64 * c;
        for (int l = 0; l < 32; ++l) out[l] = __uint_as_float((uint32_t) ld16(r + 2 * l) << 16);
    } else if constexpr (T == T_Q8_0) {
        const uint8_t* b = row + 34 * c;
        const float d = h2f(ld16(b));
        const int8_t* q = (const int8_t*) (b + 2);
        for (int l = 0; l < 32; ++l) out[l] = (float) q[l] * d;
    } else if constexpr (T == T_Q5_0) {
        const uint8_t* b = row + 22 * c;
        const float d = h2f(ld16(b));
        const uint32_t qh = (uint32_t) b[2] | ((uint32_t) b[3] << 8) | ((uint32_t) b[4] << 16) | ((uint32_t) b[5] << 24);
        const uint8_t* qs = b + 6;
        for (int j = 0; j < 16; ++j) {
            const uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh1 = ((qh >> (j + 12))) & 0x10;
            const int32_t x0 = ((qs[j] & 0x0F) | xh0) - 16;
            const int32_t x1 = ((qs[j] >> 4) | xh1) - 16;
            out[j] = (float) x0 * d;
            out[j + 16] = (float) x1 * d;
        }
    } else if constexpr (T == T_IQ4_NL) {
        const uint8_t* b = row + 18 * c;
        const float d = h2f(ld16(b));
        const uint8_t* qs = b + 2;
        for (int j = 0; j < 16; ++j) {
            out[j] = d * (float) kIq4nl[qs[j] & 0xf];
            out[j + 16] = d * (float) kIq4nl[qs[j] >> 4];
        }
    } else if constexpr (T == T_Q4_K) {
        const uint8_t* b = row + 144 * (c >> 3);
        const int j = c & 7;
        const float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
        int sc, m;
        scale_min_k4(j, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* qs = b + 16 + 32 * (j >> 1);
        for (int l = 0; l < 32; ++l) out[l] = d1 * (float) ((j & 1) ? (qs[l] >> 4) : (qs[l] & 0xF)) - m1;
    } else if constexpr (T == T_Q5_K) {
        const uint8_t* b = row + 176 * (c >> 3);
        const int j = c & 7;
        const float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
        int sc, m;
        scale_min_k4(j, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* qh = b + 16;
        const uint8_t* qs = b + 48 + 32 * (j >> 1);
        for (int l = 0; l < 32; ++l) {
            const int q = ((j & 1) ? (qs[l] >> 4) : (qs[l] & 0xF)) + (((qh[l] >> j) & 1) ? 16 : 0);
            out[l] = d1 * (float) q - m1;
        }
    } else if constexpr (T == T_Q6_K) {
        const uint8_t* b = row + 210 * (c >> 3);
        const int j = c & 7, n = j >> 2, r = j & 3;
        const uint8_t* ql = b + 64 * n;
        const uint8_t* qh = b + 128 + 32 * n;
        const int8_t* sc = (const int8_t*) (b + 192) + 8 * n;
        const float d = h2f(ld16(b + 208));
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            int q;
            if (r == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            else if (r == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (r == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            out[l] = d * (float) sc[is + 2 * r] * (float) (q - 32);
        }
    }
}

template <int T> __global__ void k_deq(const uint8_t* W, size_t row_bytes, int n_in, int64_t rows, float* out) {
    const int64_t nc = n_in / 32, total = rows * nc;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t r = i / nc;
        const int c = (int) (i % nc);
        deq32<T>(W + r * row_bytes, c, out + r * n_in + 32 * c);
    }
}

/// x[t] = row tok[t] of the embedding matrix (grid = T)
template <int T> __global__ void k_embed(const uint8_t* W, size_t row_bytes, int n, const int32_t* tok, float* x) {
    const uint8_t* row = W + (size_t) tok[blockIdx.x] * row_bytes;
    for (int c = threadIdx.x; c < n / 32; c += blockDim.x) deq32<T>(row, c, x + (size_t) blockIdx.x * n + 32 * c);
}

void launch_deq(uint32_t type, const uint8_t* W, size_t rb, int n_in, int64_t rows, float* out) {
    const dim3 g(8192), b(256);
    switch (type) {
    case T_F16: k_deq<T_F16><<<g, b>>>(W, rb, n_in, rows, out); break;
    case T_BF16: k_deq<T_BF16><<<g, b>>>(W, rb, n_in, rows, out); break;
    case T_Q8_0: k_deq<T_Q8_0><<<g, b>>>(W, rb, n_in, rows, out); break;
    case T_Q5_0: k_deq<T_Q5_0><<<g, b>>>(W, rb, n_in, rows, out); break;
    case T_IQ4_NL: k_deq<T_IQ4_NL><<<g, b>>>(W, rb, n_in, rows, out); break;
    case T_Q4_K: k_deq<T_Q4_K><<<g, b>>>(W, rb, n_in, rows, out); break;
    case T_Q5_K: k_deq<T_Q5_K><<<g, b>>>(W, rb, n_in, rows, out); break;
    case T_Q6_K: k_deq<T_Q6_K><<<g, b>>>(W, rb, n_in, rows, out); break;
    default: std::fprintf(stderr, "nhref: no dequantiser for type %u\n", type); std::exit(1);
    }
    CK(cudaGetLastError());
}

size_t row_bytes(uint32_t type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry(type, be, bb) || n % be) {
        std::fprintf(stderr, "nhref: bad row (type %u, n %lld)\n", type, (long long) n);
        std::exit(1);
    }
    return (size_t) (n / be) * (size_t) bb;
}

// ------------------------------------------------------------------------------------------------ small kernels
constexpr int kThreads = 256;

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }

/// y = x / sqrt(mean(x^2) + eps) * w, per row of n (grid = rows): ggml_rms_norm then ggml_mul (build_norm)
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = 1.0f / sqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = (xr[i] * r) * w[i];
}

__global__ void k_add(float* x, const float* y, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) x[i] += y[i];
}

/// relu(x)^2 in place (LLM_FFN_RELU_SQR: ggml_relu then ggml_sqr in build_ffn / build_moe_ffn)
__global__ void k_relu2(float* x, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float r = fmaxf(x[i], 0.f);
        x[i] = r * r;
    }
}

/// The causal depthwise conv over the 10240 xBC channels (kernel 4) with its 3-column state, + bias, SiLU, for T
/// tokens in order (ggml_ssm_conv on concat(state, xBC^T): ggml-cpu/ops.cpp:10022-10070; mamba-base.cpp:210-242).
__global__ void k_conv(const float* in, int ld_in, float* state, const float* w, const float* b, float* out, int C, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float s0 = state[(size_t) c * 3 + 0], s1 = state[(size_t) c * 3 + 1], s2 = state[(size_t) c * 3 + 2];
    const float w0 = w[(size_t) c * 4 + 0], w1 = w[(size_t) c * 4 + 1], w2 = w[(size_t) c * 4 + 2], w3 = w[(size_t) c * 4 + 3];
    for (int t = 0; t < T; ++t) {
        const float xt = in[(size_t) t * ld_in + c];
        float acc = 0.0f;
        acc += s0 * w0;
        acc += s1 * w1;
        acc += s2 * w2;
        acc += xt * w3;
        out[(size_t) t * C + c] = silu(acc + b[c]);
        s0 = s1;
        s1 = s2;
        s2 = xt;
    }
    state[(size_t) c * 3 + 0] = s0;
    state[(size_t) c * 3 + 1] = s1;
    state[(size_t) c * 3 + 2] = s2;
}

/// The Mamba-2 selective scan for T tokens (grid = heads, block = head_dim; thread i owns S[h][i][0..DS)):
/// ggml_ssm_scan's scalar-A branch, ggml-cpu/ops.cpp:10151-10200, then y += x * D (mamba-base.cpp:286).
template <int DS>
__global__ void k_scan(const float* xbc, int ld_xbc, const float* zx, int ld_zx, int dt_off, const float* dt_b,
                       const float* A, const float* Dv, float* S, float* y, int head_dim, int heads_per_group,
                       int d_inner, int n_group, int T) {
    __shared__ float Bs[DS], Cs[DS];
    const int h = blockIdx.x, i = threadIdx.x, g = h / heads_per_group;
    float s[DS];
    float* Sg = S + ((size_t) h * head_dim + i) * DS;
    for (int k = 0; k < DS; ++k) s[k] = Sg[k];
    const float a = A[h], dd = Dv[h], db = dt_b[h];
    for (int t = 0; t < T; ++t) {
        const float* row = xbc + (size_t) t * ld_xbc;
        __syncthreads();
        for (int k = i; k < DS; k += blockDim.x) {
            Bs[k] = row[d_inner + g * DS + k];
            Cs[k] = row[d_inner + n_group * DS + g * DS + k];
        }
        __syncthreads();
        const float dt = zx[(size_t) t * ld_zx + dt_off + h] + db;
        const float sp = dt <= 20.0f ? log1pf(expf(dt)) : dt;   // ggml_compute_softplus_f32
        const float dA = expf(sp * a);
        const float x = row[h * head_dim + i];
        const float x_dt = x * sp;
        float sum = 0.0f;
        for (int k = 0; k < DS; ++k) {
            const float st = s[k] * dA + Bs[k] * x_dt;
            s[k] = st;
            sum += st * Cs[k];
        }
        y[(size_t) t * d_inner + h * head_dim + i] = sum + x * dd;
    }
    for (int k = 0; k < DS; ++k) Sg[k] = s[k];
}

/// y = silu(z) * y (ggml_swiglu_split, mamba-base.cpp:288), then RMS norm per group of gsize with weight w
/// (mamba-base.cpp:291-294).  grid = (groups, T).
__global__ void k_gate_gnorm(float* y, const float* zx, int ld_zx, const float* w, int gsize, int d_inner, float eps) {
    __shared__ float sh[kThreads];
    const int g = blockIdx.x, t = blockIdx.y;
    float* yr = y + (size_t) t * d_inner + (size_t) g * gsize;
    const float* zr = zx + (size_t) t * ld_zx + (size_t) g * gsize;
    float s = 0.f;
    for (int i = threadIdx.x; i < gsize; i += blockDim.x) {
        const float v = silu(zr[i]) * yr[i];
        yr[i] = v;
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = 1.0f / sqrtf(s / (float) gsize + eps);
    for (int i = threadIdx.x; i < gsize; i += blockDim.x) yr[i] = (yr[i] * r) * w[(size_t) g * gsize + i];
}

/// softmax(S * scale) over keys 0..p0+t of row t, 0 beyond (causal; ggml_soft_max_ext).  grid = T, S row-major [T][P].
__global__ void k_softmax_causal(float* S, int P, int p0, float scale) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, n = p0 + t + 1;
    float* r = S + (size_t) t * P;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) m = fmaxf(m, r[j] * scale);
    m = block_max(m, sh);
    float s = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float e = expf(r[j] * scale - m);
        r[j] = e;
        s += e;
    }
    s = block_sum(s, sh);
    const float inv = 1.0f / s;
    for (int j = threadIdx.x; j < P; j += blockDim.x) r[j] = j < n ? r[j] * inv : 0.0f;
}

/// rows of src gathered: dst[i] = src[idx[i]] (n columns)
__global__ void k_gather(const float* src, const int32_t* idx, int n, float* dst) {
    const float* s = src + (size_t) idx[blockIdx.x] * n;
    float* d = dst + (size_t) blockIdx.x * n;
    for (int j = threadIdx.x; j < n; j += blockDim.x) d[j] = s[j];
}
/// slots[slot[i]] = src[i] * w[slot[i]]   (ggml_mul(experts, weights) after the down projection, build_moe_ffn)
__global__ void k_scatter_w(const float* src, const int32_t* slot, const float* w, int n, float* slots) {
    const int sl = slot[blockIdx.x];
    const float ww = w[sl];
    const float* s = src + (size_t) blockIdx.x * n;
    float* d = slots + (size_t) sl * n;
    for (int j = threadIdx.x; j < n; j += blockDim.x) d[j] = s[j] * ww;
}
/// out[t] = slots[t][0] + slots[t][1] + ... in selection order (the aggregation loop of build_moe_ffn)
__global__ void k_sum_slots(const float* slots, int K, int n, float* out) {
    const int t = blockIdx.x;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float acc = slots[((size_t) t * K) * n + j];
        for (int k = 1; k < K; ++k) acc += slots[((size_t) t * K + k) * n + j];
        out[(size_t) t * n + j] = acc;
    }
}

/// Diagnostic only (--emulate-act-quant): the activation rounding of llama.cpp's quantised matmul path, applied to a
/// copy of X: per 32-value block d = amax/127, q = round(x/d), x' = q * f16(d) (quantize_q8_1, ggml-cuda/quantize.cu:54-91).
__global__ void k_q8_1_round(const float* x, float* y, int64_t nblk) {
    for (int64_t b = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; b < nblk; b += (int64_t) gridDim.x * blockDim.x) {
        const float* xb = x + b * 32;
        float* yb = y + b * 32;
        float amax = 0.f;
        for (int l = 0; l < 32; ++l) amax = fmaxf(amax, fabsf(xb[l]));
        const float d = amax / 127;
        const float dh = __half2float(__float2half(d));
        for (int l = 0; l < 32; ++l) yb[l] = amax == 0.0f ? 0.0f : roundf(xb[l] / d) * dh;
    }
}
/// Diagnostic only: round to f16 in place (the f16 KV cache and the f16 Q of flash attention)
__global__ void k_f16_round(float* x, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        x[i] = __half2float(__float2half(x[i]));
}

// ------------------------------------------------------------------------------------------------ the engine
struct W {
    const uint8_t* d = nullptr;  // device
    uint32_t type = 0;
    int n_in = 0;
    int64_t n_out = 0;
    size_t rb = 0;
    const float* f() const { return (const float*) d; }
};

struct LW {
    W norm, ssm_in, conv_w, conv_b, dt_b, a, dd, ssm_norm, ssm_out, wq, wk, wv, wo, gate, bias, ldown, lup, ush, dsh;
    float* conv_state = nullptr;
    float* ssm_state = nullptr;
    float* kc = nullptr;
    float* vc = nullptr;
};

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Engine {
    const nh::Model& m;
    const nh::Geometry& g;
    cublasHandle_t cb{};
    int Tmax, ctx;
    size_t dev_bytes = 0;
    float* scratch = nullptr;
    size_t scratch_n = (size_t) 64 << 20;
    std::vector<LW> lw;
    W embd, out_norm, out;
    // activations
    float *x, *h, *zx, *xbc, *y, *t1, *t2, *q, *kk, *vv, *S, *lat, *slots, *mlat, *shu, *ea, *eu, *ed, *logits, *rl, *dw;
    int32_t *dtok, *didx;
    float *h_logits, *h_rl;
    // routed-expert slot cache (VRAM): slot = [up | down] of one expert
    size_t slot_bytes = 0, up_off_max = 0, down_max = 0;
    int n_slots = 0;
    uint8_t* pool = nullptr;
    std::vector<int> slot_of;       // (layer * n_expert + e) -> slot or -1
    std::vector<int> owner;         // slot -> key or -1
    std::vector<uint64_t> last_use; // slot -> clock
    uint64_t clock = 0, hits = 0, misses = 0, bytes_in = 0;
    double t_upload = 0;
    std::function<void(int, int, const std::vector<int32_t>&)> route_log;
    FILE* dump = nullptr;
    bool emulate = false;   // --emulate-act-quant (diagnostic)
    float* xq = nullptr;

    Engine(const nh::Model& mm, int Tmax_, int ctx_, size_t cache_mib) : m(mm), g(mm.g), Tmax(Tmax_), ctx(ctx_) {
        CB(cublasCreate(&cb));
        CB(cublasSetMathMode(cb, CUBLAS_PEDANTIC_MATH));
        lw.resize((size_t) g.n_layer);
        embd = up(m.tok_embd);
        out_norm = up(m.output_norm);
        out = up(m.output);
        for (int l = 0; l < g.n_layer; ++l) {
            const nh::Layer& L = m.layers[(size_t) l];
            LW& w = lw[(size_t) l];
            w.norm = up(L.attn_norm);
            if (L.kind == nh::Kind::Mamba) {
                w.ssm_in = up(L.ssm_in); w.conv_w = up(L.ssm_conv1d); w.conv_b = up(L.ssm_conv1d_b);
                w.dt_b = up(L.ssm_dt_b); w.a = up(L.ssm_a); w.dd = up(L.ssm_d); w.ssm_norm = up(L.ssm_norm);
                w.ssm_out = up(L.ssm_out);
                CK(cudaMalloc(&w.conv_state, sizeof(float) * (size_t) g.ssm_conv_dim * (g.ssm_d_conv - 1)));
                CK(cudaMalloc(&w.ssm_state, sizeof(float) * (size_t) g.ssm_d_inner * g.ssm_d_state));
                dev_bytes += sizeof(float) * ((size_t) g.ssm_conv_dim * 3 + (size_t) g.ssm_d_inner * g.ssm_d_state);
            } else if (L.kind == nh::Kind::Attention) {
                w.wq = up(L.wq); w.wk = up(L.wk); w.wv = up(L.wv); w.wo = up(L.wo);
                const size_t kv = (size_t) ctx * g.n_head_kv * g.head_dim;
                CK(cudaMalloc(&w.kc, sizeof(float) * kv));
                CK(cudaMalloc(&w.vc, sizeof(float) * kv));
                dev_bytes += 2 * sizeof(float) * kv;
            } else if (L.kind == nh::Kind::Moe) {
                w.gate = up(L.gate_inp); w.bias = up(L.exp_probs_b); w.ldown = up(L.latent_down);
                w.lup = up(L.latent_up); w.ush = up(L.up_shexp); w.dsh = up(L.down_shexp);
                const size_t ub = row_bytes(L.up_exps.type, g.moe_latent) * (size_t) g.n_ff_exp;
                const size_t db = row_bytes(L.down_exps.type, g.n_ff_exp) * (size_t) g.moe_latent;
                up_off_max = std::max(up_off_max, ub);
                down_max = std::max(down_max, db);
            }
        }
        // slot layout: [up at 0 .. up_off_max) [down at up_off_max .. slot_bytes)
        up_off_max = (up_off_max + 255) / 256 * 256;
        slot_bytes = up_off_max + (down_max + 255) / 256 * 256;
        n_slots = (int) ((cache_mib << 20) / slot_bytes);
        CK(cudaMalloc(&pool, slot_bytes * (size_t) n_slots));
        dev_bytes += slot_bytes * (size_t) n_slots;
        slot_of.assign((size_t) g.n_layer * g.n_expert, -1);
        owner.assign((size_t) n_slots, -1);
        last_use.assign((size_t) n_slots, 0);
        auto A = [&](float*& p, size_t n) { CK(cudaMalloc(&p, n * sizeof(float))); dev_bytes += n * sizeof(float); };
        const size_t T = (size_t) Tmax;
        A(scratch, scratch_n);
        A(x, T * g.n_embd); A(h, T * g.n_embd); A(zx, T * g.ssm_d_in_proj); A(xbc, T * g.ssm_conv_dim);
        A(y, T * g.ssm_d_inner); A(t1, T * g.n_embd); A(t2, T * g.n_embd); A(q, T * g.n_head * g.head_dim);
        A(kk, T * g.n_head_kv * g.head_dim); A(vv, T * g.n_head_kv * g.head_dim); A(S, T * (size_t) ctx);
        A(lat, T * g.moe_latent); A(slots, T * g.n_expert_used * g.moe_latent); A(mlat, T * g.moe_latent);
        A(shu, T * g.n_ff_shexp); A(ea, T * g.moe_latent); A(eu, T * g.n_ff_exp); A(ed, T * g.moe_latent);
        A(logits, T * g.n_vocab); A(rl, T * g.n_expert); A(dw, T * g.n_expert_used);
        A(xq, T * (size_t) std::max(g.ssm_d_inner, g.n_ff_shexp));
        CK(cudaMalloc(&dtok, sizeof(int32_t) * T));
        CK(cudaMalloc(&didx, sizeof(int32_t) * T * g.n_expert_used * 2));
        CK(cudaMallocHost(&h_logits, sizeof(float) * T * g.n_vocab));
        CK(cudaMallocHost(&h_rl, sizeof(float) * T * g.n_expert));
    }

    W up(const nh::Tensor& t) {
        W w;
        w.type = t.type;
        w.n_in = (int) t.ne(0);
        w.n_out = (int64_t) (t.ne(1) * t.ne(2) * t.ne(3));
        w.rb = row_bytes(t.type, w.n_in);
        void* p = nullptr;
        CK(cudaMalloc(&p, t.bytes));
        CK(cudaMemcpy(p, t.data, t.bytes, cudaMemcpyHostToDevice));
        dev_bytes += t.bytes;
        w.d = (const uint8_t*) p;
        return w;
    }

    void reset() {
        for (int l = 0; l < g.n_layer; ++l) {
            LW& w = lw[(size_t) l];
            if (w.conv_state) CK(cudaMemset(w.conv_state, 0, sizeof(float) * (size_t) g.ssm_conv_dim * (g.ssm_d_conv - 1)));
            if (w.ssm_state) CK(cudaMemset(w.ssm_state, 0, sizeof(float) * (size_t) g.ssm_d_inner * g.ssm_d_state));
        }
    }

    /// Y[T][n_out] = X[T][n_in] W^T, W in device memory: fp32 SGEMM over dequantised row slices.
    void mm(const uint8_t* wd, uint32_t type, int n_in, int64_t n_out, size_t rb, const float* X, float* Y, int T) {
        const float one = 1.0f, zero = 0.0f;
        if (type == T_F32) {
            CB(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) n_out, T, n_in, &one, (const float*) wd, n_in, X, n_in,
                           &zero, Y, (int) n_out));
            return;
        }
        if (emulate) {
            k_q8_1_round<<<1024, kThreads>>>(X, xq, (int64_t) T * n_in / 32);
            CK(cudaGetLastError());
            X = xq;
        }
        const int64_t per = std::max<int64_t>(1, (int64_t) (scratch_n / (size_t) n_in));
        for (int64_t r0 = 0; r0 < n_out; r0 += per) {
            const int64_t rs = std::min<int64_t>(per, n_out - r0);
            launch_deq(type, wd + (size_t) r0 * rb, rb, n_in, rs, scratch);
            CB(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) rs, T, n_in, &one, scratch, n_in, X, n_in, &zero,
                           Y + r0, (int) n_out));
        }
    }
    void mm(const W& w, const float* X, float* Y, int T) { mm(w.d, w.type, w.n_in, w.n_out, w.rb, X, Y, T); }

    /// The device copy of expert e of MoE layer l ([up | down] in one slot), uploaded from the mapping on a miss (LRU).
    const uint8_t* expert(int l, int e) {
        const size_t key = (size_t) l * g.n_expert + e;
        ++clock;
        int s = slot_of[key];
        if (s >= 0) {
            ++hits;
            last_use[(size_t) s] = clock;
            return pool + (size_t) s * slot_bytes;
        }
        ++misses;
        s = 0;
        for (int i = 0; i < n_slots; ++i) {
            if (owner[(size_t) i] < 0) { s = i; break; }
            if (last_use[(size_t) i] < last_use[(size_t) s]) s = i;
        }
        if (owner[(size_t) s] >= 0) slot_of[(size_t) owner[(size_t) s]] = -1;
        owner[(size_t) s] = (int) key;
        slot_of[key] = s;
        last_use[(size_t) s] = clock;
        const nh::Layer& L = m.layers[(size_t) l];
        const size_t ub = row_bytes(L.up_exps.type, g.moe_latent) * (size_t) g.n_ff_exp;
        const size_t db = row_bytes(L.down_exps.type, g.n_ff_exp) * (size_t) g.moe_latent;
        uint8_t* dst = pool + (size_t) s * slot_bytes;
        const double t0 = now_s();
        CK(cudaMemcpy(dst, L.up_exps.data + (size_t) e * ub, ub, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dst + up_off_max, L.down_exps.data + (size_t) e * db, db, cudaMemcpyHostToDevice));
        t_upload += now_s() - t0;
        bytes_in += ub + db;
        return dst;
    }

    void mamba(int l, int T) {
        LW& w = lw[(size_t) l];
        const int DI = g.ssm_d_inner, CD = g.ssm_conv_dim, NG = g.ssm_n_group, DS = g.ssm_d_state;
        mm(w.ssm_in, h, zx, T);
        k_conv<<<(CD + 127) / 128, 128>>>(zx + DI, g.ssm_d_in_proj, w.conv_state, w.conv_w.f(), w.conv_b.f(), xbc, CD, T);
        CK(cudaGetLastError());
        if (DS != 128) { std::fprintf(stderr, "nhref: d_state %d unsupported\n", DS); std::exit(1); }
        k_scan<128><<<g.ssm_n_head, g.ssm_head_dim>>>(xbc, CD, zx, g.ssm_d_in_proj, 2 * DI + 2 * NG * DS, w.dt_b.f(),
                                                       w.a.f(), w.dd.f(), w.ssm_state, y, g.ssm_head_dim,
                                                       g.ssm_n_head / NG, DI, NG, T);
        CK(cudaGetLastError());
        k_gate_gnorm<<<dim3(NG, T), kThreads>>>(y, zx, g.ssm_d_in_proj, w.ssm_norm.f(), DI / NG, DI, g.rms_eps);
        CK(cudaGetLastError());
        mm(w.ssm_out, y, t1, T);
    }

    void attention(int l, int T, int p0) {
        LW& w = lw[(size_t) l];
        const int HD = g.head_dim, H = g.n_head, HKV = g.n_head_kv, KVW = HKV * HD, P = p0 + T;
        mm(w.wq, h, q, T);
        mm(w.wk, h, kk, T);
        mm(w.wv, h, vv, T);
        if (emulate) {
            k_f16_round<<<256, kThreads>>>(q, (int64_t) T * H * HD);
            k_f16_round<<<256, kThreads>>>(kk, (int64_t) T * KVW);
            k_f16_round<<<256, kThreads>>>(vv, (int64_t) T * KVW);
            CK(cudaGetLastError());
        }
        CK(cudaMemcpy(w.kc + (size_t) p0 * KVW, kk, sizeof(float) * (size_t) T * KVW, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(w.vc + (size_t) p0 * KVW, vv, sizeof(float) * (size_t) T * KVW, cudaMemcpyDeviceToDevice));
        const float one = 1.0f, zero = 0.0f, scale = 1.0f / sqrtf((float) HD);   // nemotron-h.cpp:285-286
        for (int hq = 0; hq < H; ++hq) {
            const int gk = hq / (H / HKV);
            CB(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, P, T, HD, &one, w.kc + gk * HD, KVW, q + hq * HD, H * HD, &zero,
                           S, P));
            k_softmax_causal<<<T, kThreads>>>(S, P, p0, scale);
            CK(cudaGetLastError());
            CB(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, HD, T, P, &one, w.vc + gk * HD, KVW, S, P, &zero, y + hq * HD,
                           H * HD));
        }
        mm(w.wo, y, t1, T);
    }

    void moe(int l, int T) {
        LW& w = lw[(size_t) l];
        const nh::Layer& L = m.layers[(size_t) l];
        const int NE = g.n_expert, K = g.n_expert_used, LAT = g.moe_latent, FF = g.n_ff_exp;
        mm(w.gate, h, rl, T);   // router logits on the full hidden (nemotron-h.cpp:313)
        CK(cudaMemcpy(h_rl, rl, sizeof(float) * (size_t) T * NE, cudaMemcpyDeviceToHost));
        std::vector<float> bias((size_t) NE);
        CK(cudaMemcpy(bias.data(), w.bias.d, sizeof(float) * NE, cudaMemcpyDeviceToHost));
        std::vector<int32_t> sel((size_t) T * K);
        std::vector<float> wt((size_t) T * K);
        std::vector<float> p((size_t) NE), sp((size_t) NE);
        std::vector<int> ord((size_t) NE);
        for (int t = 0; t < T; ++t) {
            for (int e = 0; e < NE; ++e) {
                p[(size_t) e] = 1.0f / (1.0f + expf(-h_rl[(size_t) t * NE + e]));   // sigmoid, llama-graph.cpp:2247
                sp[(size_t) e] = p[(size_t) e] + bias[(size_t) e];                   // selection only, :2266
                ord[(size_t) e] = e;
            }
            std::partial_sort(ord.begin(), ord.begin() + K, ord.end(), [&](int a, int b) {
                return sp[(size_t) a] > sp[(size_t) b] || (sp[(size_t) a] == sp[(size_t) b] && a < b);
            });
            float sum = 0.f;
            for (int k = 0; k < K; ++k) {
                sel[(size_t) t * K + k] = ord[(size_t) k];
                wt[(size_t) t * K + k] = p[(size_t) ord[(size_t) k]];   // unbiased, :2323
                sum += wt[(size_t) t * K + k];
            }
            sum = std::max(sum, 6.103515625e-5f);                          // :2337-2341
            for (int k = 0; k < K; ++k) {
                float v = wt[(size_t) t * K + k] / sum;
                if (g.expert_weights_scale != 0.f && g.expert_weights_scale != 1.f) v *= g.expert_weights_scale;   // :2350
                wt[(size_t) t * K + k] = v;
            }
        }
        if (route_log) route_log(l, T, sel);
        CK(cudaMemcpy(dw, wt.data(), sizeof(float) * (size_t) T * K, cudaMemcpyHostToDevice));
        mm(w.ldown, h, lat, T);   // latent input (nemotron-h.cpp:308-310)
        // entries grouped by expert: rows (token) and slots (token * K + k)
        std::vector<std::vector<int>> by((size_t) NE);
        for (int i = 0; i < T * K; ++i) by[(size_t) sel[(size_t) i]].push_back(i);
        std::vector<int32_t> rows, slts;
        std::vector<int> off((size_t) NE + 1, 0);
        for (int e = 0; e < NE; ++e) {
            off[(size_t) e] = (int) rows.size();
            for (int i : by[(size_t) e]) {
                rows.push_back(i / K);
                slts.push_back(i);
            }
        }
        off[(size_t) NE] = (int) rows.size();
        CK(cudaMemcpy(didx, rows.data(), sizeof(int32_t) * rows.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(didx + (size_t) T * K, slts.data(), sizeof(int32_t) * slts.size(), cudaMemcpyHostToDevice));
        const uint32_t ut = L.up_exps.type, dt = L.down_exps.type;
        const size_t urb = row_bytes(ut, LAT), drb = row_bytes(dt, FF);
        for (int e = 0; e < NE; ++e) {
            const int n = off[(size_t) e + 1] - off[(size_t) e];
            if (!n) continue;
            const uint8_t* ex = expert(l, e);
            k_gather<<<n, kThreads>>>(lat, didx + off[(size_t) e], LAT, ea);
            mm(ex, ut, LAT, FF, urb, ea, eu, n);
            k_relu2<<<256, kThreads>>>(eu, (int64_t) n * FF);
            mm(ex + up_off_max, dt, FF, LAT, drb, eu, ed, n);
            k_scatter_w<<<n, kThreads>>>(ed, didx + (size_t) T * K + off[(size_t) e], dw, LAT, slots);
            CK(cudaGetLastError());
        }
        k_sum_slots<<<T, kThreads>>>(slots, K, LAT, mlat);
        mm(w.lup, mlat, t1, T);   // nemotron-h.cpp:330-332
        mm(w.ush, h, shu, T);     // shared expert on the full hidden (nemotron-h.cpp:336-344)
        k_relu2<<<1024, kThreads>>>(shu, (int64_t) T * g.n_ff_shexp);
        mm(w.dsh, shu, t2, T);
        k_add<<<1024, kThreads>>>(t1, t2, (int64_t) T * g.n_embd);   // moe_out + ffn_shexp
        CK(cudaGetLastError());
    }

    /// tokens tok[0..T) at positions p0.. -> logits [T][vocab] in h_logits
    void forward(const int32_t* tok, int T, int p0) {
        CK(cudaMemcpy(dtok, tok, sizeof(int32_t) * T, cudaMemcpyHostToDevice));
        const int N = g.n_embd;
        if (embd.type != T_Q8_0) { std::fprintf(stderr, "nhref: token_embd type %u\n", embd.type); std::exit(1); }
        k_embed<T_Q8_0><<<T, kThreads>>>(embd.d, embd.rb, N, dtok, x);
        CK(cudaGetLastError());
        for (int l = 0; l < g.n_layer; ++l) {
            k_rms<<<T, kThreads>>>(x, lw[(size_t) l].norm.f(), h, N, g.rms_eps);
            switch (m.layers[(size_t) l].kind) {
            case nh::Kind::Mamba: mamba(l, T); break;
            case nh::Kind::Attention: attention(l, T, p0); break;
            case nh::Kind::Moe: moe(l, T); break;
            default: break;
            }
            k_add<<<1024, kThreads>>>(x, t1, (int64_t) T * N);
            CK(cudaGetLastError());
            if (dump) {
                std::vector<float> hx((size_t) T * N);
                CK(cudaMemcpy(hx.data(), x, sizeof(float) * hx.size(), cudaMemcpyDeviceToHost));
                std::fwrite(hx.data(), sizeof(float), hx.size(), dump);
            }
        }
        k_rms<<<T, kThreads>>>(x, out_norm.f(), h, N, g.rms_eps);
        mm(out, h, logits, T);
        CK(cudaMemcpy(h_logits, logits, sizeof(float) * (size_t) T * g.n_vocab, cudaMemcpyDeviceToHost));
    }
};

// ------------------------------------------------------------------------------------------------ capture rows
// the golden-set capture tool's row writer in effect: lse in double over the full vocabulary, top-K by descending
// logit with ties to the lowest id, target logprob = logit - lse.
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

std::vector<int32_t> read_i32(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<int32_t> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, list, tokdir, outdir, force, only, dump_hidden, dump_prompt, route_file, deq_test, deq_out;
    int K = 32, chunk = 512, ctx = 32768;
    size_t cache_mib = 12000;
    bool emulate = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--emulate-act-quant") emulate = true;
    for (int i = 1; i + 1 < argc; ++i) {
        const std::string a = argv[i];
        const char* v = argv[i + 1];
        if (a == "--model") model = v;
        else if (a == "--list") list = v;
        else if (a == "--tokdir") tokdir = v;
        else if (a == "--out") outdir = v;
        else if (a == "--force") force = v;
        else if (a == "--only") only = v;
        else if (a == "--k") K = std::atoi(v);
        else if (a == "--chunk") chunk = std::atoi(v);
        else if (a == "--ctx") ctx = std::atoi(v);
        else if (a == "--expert-cache-mib") cache_mib = (size_t) std::atoll(v);
        else if (a == "--dump-hidden") dump_hidden = v;
        else if (a == "--dump-prompt") dump_prompt = v;
        else if (a == "--route-log") route_file = v;
        else if (a == "--deq-test") deq_test = v;
        else if (a == "--deq-out") deq_out = v;
        else continue;
        ++i;
    }
    if (model.empty() || ((list.empty() || tokdir.empty() || outdir.empty()) && deq_test.empty())) {
        std::fprintf(stderr, "usage: strata-nhref --model SHARD --list LIST --tokdir DIR --out RAWDIR [opts]\n");
        return 2;
    }
    std::set<std::string> only_ids;
    {
        std::stringstream ss(only);
        std::string t;
        while (std::getline(ss, t, ',')) if (!t.empty()) only_ids.insert(t);
    }
    const double t_load0 = now_s();
    const strata::GgufModel gm = strata::GgufModel::open(model);
    const nh::Model m = nh::load_model(gm);
    if (!m.report.pass()) {
        for (const auto& e : m.report.errors) std::fprintf(stderr, "layout: %s\n", e.c_str());
        std::fprintf(stderr, "nhref: the layout check failed; refusing to run\n");
        return 1;
    }
    if (!deq_test.empty()) {
        // the dequantiser check: rows NAME:ROW,... (row over all n_out rows, experts included) through the GPU path,
        // written as f32 to --deq-out for comparison with gguf-py's dequantize
        float* dbuf = nullptr;
        CK(cudaMalloc(&dbuf, sizeof(float) * 65536));
        FILE* fo = std::fopen(deq_out.c_str(), "wb");
        std::stringstream ss(deq_test);
        std::string item;
        while (std::getline(ss, item, ',')) {
            const size_t c = item.rfind(':');
            const std::string name = item.substr(0, c);
            const int64_t row = std::atoll(item.substr(c + 1).c_str());
            size_t sh = 0;
            const strata::TensorInfo* t = gm.find(name, &sh);
            if (!t) { std::fprintf(stderr, "nhref: no tensor %s\n", name.c_str()); return 1; }
            const int n_in = (int) t->shape[0];
            const size_t rb = row_bytes(t->type, n_in);
            const uint8_t* src = gm.shard(sh).tensor_data(*t) + (size_t) row * rb;
            uint8_t* dsrc = nullptr;
            CK(cudaMalloc(&dsrc, rb));
            CK(cudaMemcpy(dsrc, src, rb, cudaMemcpyHostToDevice));
            std::vector<float> hv((size_t) n_in);
            if (t->type == T_F32) std::memcpy(hv.data(), src, rb);
            else {
                launch_deq(t->type, dsrc, rb, n_in, 1, dbuf);
                CK(cudaMemcpy(hv.data(), dbuf, sizeof(float) * n_in, cudaMemcpyDeviceToHost));
            }
            CK(cudaFree(dsrc));
            std::fwrite(hv.data(), sizeof(float), hv.size(), fo);
            std::fprintf(stderr, "[deq] %s row %lld type %u n %d\n", name.c_str(), (long long) row, t->type, n_in);
        }
        std::fclose(fo);
        return 0;
    }
    Engine E(m, chunk, ctx, cache_mib);
    E.emulate = emulate;
    FILE* rlog = route_file.empty() ? nullptr : std::fopen(route_file.c_str(), "wb");
    if (rlog)
        E.route_log = [&](int l, int T, const std::vector<int32_t>& sel) {
            std::fwrite(&l, 4, 1, rlog);
            std::fwrite(&T, 4, 1, rlog);
            std::fwrite(sel.data(), 4, sel.size(), rlog);
        };
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    std::fprintf(stderr, "[nhref] loaded in %.1f s; device bytes %zu (%.1f MiB); expert slots %d x %zu B; GPU free %.1f MiB\n",
                 now_s() - t_load0, E.dev_bytes, E.dev_bytes / 1048576.0, E.n_slots, E.slot_bytes, fr / 1048576.0);
    const int nv = m.g.n_vocab;
    const std::set<int> eog = {11, 2};   // llama-vocab.cpp:3172-3202 (<|im_end|>, </s>), :3233 (eos)
    FILE* sumf = std::fopen((outdir + "/capture_log.txt").c_str(), "w");
    std::fprintf(sumf, "engine=strata-nhref n_vocab=%d n_ctx=%d K=%d chunk=%d force=%s\n", nv, ctx, K, chunk, force.c_str());
    std::ifstream lf(list);
    std::string id;
    int ngen;
    unsigned nthr = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    while (lf >> id >> ngen) {
        if (!only_ids.empty() && !only_ids.count(id)) continue;
        const std::vector<int32_t> pt = read_i32(tokdir + "/" + id + ".i32");
        std::vector<int32_t> forced;
        if (!force.empty()) forced = read_i32(force + "/" + id + ".out.i32");
        const int np = (int) pt.size();
        int maxgen = std::min(ngen, ctx - np);
        if (!force.empty()) maxgen = (int) forced.size();
        E.reset();
        E.dump = (!dump_hidden.empty() && id == dump_prompt) ? std::fopen(dump_hidden.c_str(), "wb") : nullptr;
        std::vector<Row> rows;
        rows.reserve((size_t) np + maxgen);
        std::vector<float> last((size_t) nv);
        const double t0 = now_s();
        const uint64_t miss0 = E.misses, hit0 = E.hits;
        for (int off = 0; off < np; off += chunk) {
            const int n = std::min(chunk, np - off);
            E.forward(pt.data() + off, n, off);
            const size_t base = rows.size();
            const int nrow = std::min(n, np - 1 - off);   // rows with a prompt target
            if (nrow > 0) rows.resize(base + (size_t) nrow);
            std::vector<std::thread> th;
            for (unsigned w = 0; w < nthr; ++w)
                th.emplace_back([&, w] {
                    for (int i = (int) w; i < nrow; i += (int) nthr)
                        make_row(E.h_logits + (size_t) i * nv, nv, K, pt[(size_t) off + i + 1], rows[base + (size_t) i]);
                });
            for (auto& t : th) t.join();
            if (off + n == np) std::memcpy(last.data(), E.h_logits + (size_t) (n - 1) * nv, sizeof(float) * nv);
        }
        const double t1 = now_s();
        if (E.dump) {
            std::fclose(E.dump);
            E.dump = nullptr;
        }
        std::vector<int32_t> out;
        bool is_eog = false;
        for (int j = 0; j < maxgen; ++j) {
            const int am = argmax(last.data(), nv);
            const int t = force.empty() ? am : forced[(size_t) j];
            rows.emplace_back();
            make_row(last.data(), nv, K, t, rows.back());
            out.push_back(t);
            if (force.empty() && eog.count(t)) { is_eog = true; break; }
            if (j == maxgen - 1) break;
            E.forward(&out.back(), 1, np + j);
            std::memcpy(last.data(), E.h_logits, sizeof(float) * nv);
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
        std::fclose(fi); std::fclose(fl); std::fclose(fs); std::fclose(ft);
        FILE* fo = op(".out.i32");
        std::fwrite(out.data(), 4, out.size(), fo);
        std::fclose(fo);
        std::fprintf(sumf, "%s n_prompt=%d n_out=%zu rows=%zu eog=%d prompt_s=%.3f gen_s=%.3f expert_hits=%llu misses=%llu\n",
                     id.c_str(), np, out.size(), rows.size(), is_eog ? 1 : 0, t1 - t0, t2 - t1,
                     (unsigned long long) (E.hits - hit0), (unsigned long long) (E.misses - miss0));
        std::fflush(sumf);
        std::fprintf(stderr, "[nhref] %s n_prompt=%d n_out=%zu eog=%d prompt_s=%.1f gen_s=%.1f upload %.1f GB in %.1f s\n",
                     id.c_str(), np, out.size(), is_eog ? 1 : 0, t1 - t0, t2 - t1, E.bytes_in / 1e9, E.t_upload);
    }
    std::fclose(sumf);
    if (rlog) std::fclose(rlog);
    return 0;
}
