// src/nemotron_h/nh_cpuexact.cuh - CUDA kernels that reproduce, bit for bit, what ggml-cpu (x86 AVX2/AVX512-VNNI build,
// src/llama.cpp @ 0682a576a, ggml/src/ggml-cpu/arch/x86/quants.c) computes for one routed expert at decode time:
//   up   = vec_dot_{q4_K,q5_K,q6_K}_q8_K(row, q8_K(x))         (x quantised on the host by ggml's own quantize_row_q8_K)
//   h    = sqr(relu(up))                                        (ggml_vec_relu_f32 / ggml_vec_sqr_f32: exact)
//   hq   = quantize_row_q8_0(h)   (the AVX2 path: d = amax/127 -> fp16, id = 127/amax, round-half-even(h*id))
//   down = vec_dot_{q5_0,iq4_nl}_q8_0(row, hq)
// The SIMD code keeps 8 int32 lanes (lane m = bytes 4m..4m+3 of each 32-byte vector), converts each lane to float,
// accumulates per lane with FMA over the (super)blocks in order, and reduces the 8 lanes as hsum_float_8:
// ((x0+x4)+(x2+x6)) + ((x1+x5)+(x3+x7)).  Integer parts are exact, so only the lane split, the float op order and
// fused-vs-unfused rounding matter; every float op here is an explicit _rn intrinsic so nvcc cannot contract them.
// One output row is computed by 8 consecutive threads (one per SIMD lane).  Verified against ggml-cpu by
// strata-nh-vecdot-test (bitwise, every row).
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

namespace nhx {

constexpr int QK_K = 256;
constexpr int Q4K_BYTES = 144, Q5K_BYTES = 176, Q6K_BYTES = 210, Q8K_BYTES = 292;
constexpr int Q50_BYTES = 22, IQ4NL_BYTES = 18, Q80_BYTES = 34;

__device__ __forceinline__ float h2f(const uint8_t* p) {
    __half_raw r;
    r.x = (unsigned short) (p[0] | (p[1] << 8));
    return __half2float(__half(r));
}
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}
__device__ __forceinline__ int16_t ld16s(const uint8_t* p) { return (int16_t) (p[0] | (p[1] << 8)); }

// hsum_float_8 over the 8 threads of a row group (all 8 get the result)
__device__ __forceinline__ float hsum8(float v) {
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 4, 8));
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2, 8));
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1, 8));
    return v;
}

// get_scale_min_k4 as the AVX2 code unpacks it (utmp): sc[0..7] scales, mn[0..7] mins
__device__ __forceinline__ void unpack_k4(const uint8_t* s12, uint8_t sc[8], uint8_t mn[8]) {
    const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t u0 = ld32(s12), u1 = ld32(s12 + 4), u2 = ld32(s12 + 8), u3;
    u3 = ((u2 >> 4) & kmask2) | (((u1 >> 6) & kmask3) << 4);
    const uint32_t uaux = u1 & kmask1;
    u1 = (u2 & kmask2) | (((u0 >> 6) & kmask3) << 4);
    u2 = uaux;
    u0 &= kmask1;
    for (int i = 0; i < 4; ++i) {
        sc[i] = (uint8_t) (u0 >> (8 * i));
        sc[4 + i] = (uint8_t) (u1 >> (8 * i));
        mn[i] = (uint8_t) (u2 >> (8 * i));
        mn[4 + i] = (uint8_t) (u3 >> (8 * i));
    }
}

// q8_K activations staged in shared memory
struct Q8KSh {
    float d[4];
    int8_t qs[4][QK_K];
    int16_t bsums[4][16];
};

// ---- ggml_vec_dot_q4_K_q8_K (AVX2 path) ----
__device__ __forceinline__ float dot_q4K(const uint8_t* row, int nb, const Q8KSh& y, int m) {
    float acc = 0.0f, accm = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* x = row + (size_t) i * Q4K_BYTES;
        const float d = __fmul_rn(y.d[i], h2f(x));
        const float dmin = __fmul_rn(-y.d[i], h2f(x + 2));
        uint8_t sc[8], mn[8];
        unpack_k4(x + 4, sc, mn);
        const uint8_t* qs = x + 16;
        const int8_t* q8 = y.qs[i];
        int sumi = 0;
        for (int j = 0; j < 4; ++j) {
            int sl = 0, sh = 0;
            for (int b = 4 * m; b < 4 * m + 4; ++b) {
                const int q = qs[32 * j + b];
                sl += (q & 0xF) * q8[64 * j + b];
                sh += (q >> 4) * q8[64 * j + 32 + b];
            }
            sumi += (int) sc[2 * j] * sl + (int) sc[2 * j + 1] * sh;
        }
        acc = __fmaf_rn(d, __int2float_rn(sumi), acc);
        if (m < 4) {
            const int q8s0 = y.bsums[i][4 * m] + y.bsums[i][4 * m + 1];
            const int q8s1 = y.bsums[i][4 * m + 2] + y.bsums[i][4 * m + 3];
            const int prod = (int) mn[2 * m] * q8s0 + (int) mn[2 * m + 1] * q8s1;
            accm = __fmaf_rn(dmin, __int2float_rn(prod), accm);
        }
    }
    const float h = hsum8(acc);
    // acc_m: (a0+a2) + (a1+a3)
    float t = __fadd_rn(accm, __shfl_xor_sync(0xffffffffu, accm, 2, 8));
    t = __fadd_rn(t, __shfl_xor_sync(0xffffffffu, t, 1, 8));
    return __fadd_rn(h, t);
}

// ---- ggml_vec_dot_q5_K_q8_K (AVX2 path); FMA_SUMMS: whether the compiler fused `summs += dmin * isum` ----
template <bool FMA_SUMMS>
__device__ __forceinline__ float dot_q5K(const uint8_t* row, int nb, const Q8KSh& y, int m) {
    float acc = 0.0f, summs = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* x = row + (size_t) i * Q5K_BYTES;
        const float d = __fmul_rn(y.d[i], h2f(x));
        const float dmin = __fmul_rn(-y.d[i], h2f(x + 2));
        uint8_t sc[8], mn[8];
        unpack_k4(x + 4, sc, mn);
        const uint8_t* qh = x + 16;
        const uint8_t* qs = x + 48;
        const int8_t* q8 = y.qs[i];
        int isum = 0;
        for (int k = 0; k < 8; ++k) isum += (int) mn[k] * (y.bsums[i][2 * k] + y.bsums[i][2 * k + 1]);
        summs = FMA_SUMMS ? __fmaf_rn(dmin, __int2float_rn(isum), summs)
                          : __fadd_rn(summs, __fmul_rn(dmin, __int2float_rn(isum)));
        int sumi = 0;
        for (int j = 0; j < 4; ++j) {
            int s0 = 0, s1 = 0;
            for (int b = 4 * m; b < 4 * m + 4; ++b) {
                const int q = qs[32 * j + b];
                const int h0 = (qh[b] >> (2 * j)) & 1, h1 = (qh[b] >> (2 * j + 1)) & 1;
                s0 += ((q & 0xF) + 16 * h0) * q8[64 * j + b];
                s1 += ((q >> 4) + 16 * h1) * q8[64 * j + 32 + b];
            }
            sumi += (int) sc[2 * j] * s0 + (int) sc[2 * j + 1] * s1;
        }
        acc = __fmaf_rn(d, __int2float_rn(sumi), acc);
    }
    return __fadd_rn(hsum8(acc), summs);
}

// ---- ggml_vec_dot_q6_K_q8_K (AVX2 path) ----
__device__ __forceinline__ float dot_q6K(const uint8_t* row, int nb, const Q8KSh& y, int m) {
    float acc = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* x = row + (size_t) i * Q6K_BYTES;
        const uint8_t* ql = x;
        const uint8_t* qh = x + 128;
        const int8_t* sc = (const int8_t*) (x + 192);
        const float d = __fmul_rn(y.d[i], h2f(x + 208));
        const int8_t* q8 = y.qs[i];
        int sumi = 0;
        for (int j = 0; j < 2; ++j) {
            for (int b = 4 * m; b < 4 * m + 4; ++b) {
                const int l1 = ql[64 * j + b], l2 = ql[64 * j + 32 + b], h = qh[32 * j + b];
                const int v0 = (l1 & 15) | ((h & 3) << 4);
                const int v1 = (l2 & 15) | (((h >> 2) & 3) << 4);
                const int v2 = (l1 >> 4) | (((h >> 4) & 3) << 4);
                const int v3 = (l2 >> 4) | (((h >> 6) & 3) << 4);
                const int is = 8 * j + (b >= 16 ? 1 : 0);
                sumi += (int) sc[is + 0] * (v0 * q8[128 * j + b]);
                sumi += (int) sc[is + 2] * (v1 * q8[128 * j + 32 + b]);
                sumi += (int) sc[is + 4] * (v2 * q8[128 * j + 64 + b]);
                sumi += (int) sc[is + 6] * (v3 * q8[128 * j + 96 + b]);
            }
        }
        const int sub = 32 * ((int) y.bsums[i][2 * m] * sc[2 * m] + (int) y.bsums[i][2 * m + 1] * sc[2 * m + 1]);
        acc = __fmaf_rn(d, __int2float_rn(sumi - sub), acc);
    }
    return hsum8(acc);
}

// ---- ggml_vec_dot_q5_0_q8_0 (AVX2 path, one accumulator) ----
__device__ __forceinline__ float dot_q50(const uint8_t* row, int nb, const uint8_t* yq, int m) {
    float acc = 0.0f;
    for (int ib = 0; ib < nb; ++ib) {
        const uint8_t* x = row + (size_t) ib * Q50_BYTES;
        const uint8_t* y = yq + (size_t) ib * Q80_BYTES;
        const float d = __fmul_rn(h2f(x), h2f(y));
        const uint32_t qh = ld32(x + 2);
        const uint8_t* qs = x + 6;
        const int8_t* q8 = (const int8_t*) (y + 2);
        int s = 0;
        for (int b = 4 * m; b < 4 * m + 4; ++b) {
            const int nib = b < 16 ? (qs[b] & 0xF) : (qs[b - 16] >> 4);
            const int v = nib - (((qh >> b) & 1) ? 0 : 16);
            s += v * q8[b];
        }
        acc = __fmaf_rn(d, __int2float_rn(s), acc);
    }
    return hsum8(acc);
}

// ---- ggml_vec_dot_iq4_nl_q8_0 (AVX2 path: even blocks -> accum1, odd -> accum2, hsum(accum1 + accum2)) ----
__constant__ int8_t kvalues_iq4nl_d[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ __forceinline__ float dot_iq4nl(const uint8_t* row, int nb, const uint8_t* yq, int m) {
    float a1 = 0.0f, a2 = 0.0f;
    for (int ib = 0; ib + 1 < nb; ib += 2) {
        for (int h = 0; h < 2; ++h) {
            const uint8_t* x = row + (size_t) (ib + h) * IQ4NL_BYTES;
            const uint8_t* y = yq + (size_t) (ib + h) * Q80_BYTES;
            const float d = __fmul_rn(h2f(y), h2f(x));
            const uint8_t* qs = x + 2;
            const int8_t* q8 = (const int8_t*) (y + 2);
            int s = 0;
            for (int b = 4 * m; b < 4 * m + 4; ++b) {
                const int nib = b < 16 ? (qs[b] & 0xF) : (qs[b - 16] >> 4);
                s += (int) kvalues_iq4nl_d[nib] * q8[b];
            }
            if (h == 0) a1 = __fmaf_rn(d, __int2float_rn(s), a1);
            else a2 = __fmaf_rn(d, __int2float_rn(s), a2);
        }
    }
    // nb is even for this model (2688 / 32 = 84); the scalar tail of the CPU code is not needed
    return hsum8(__fadd_rn(a1, a2));
}

// Q5K: `summs += dmin * isum` is compiled as one FMA in libggml-cpu (gcc -ffp-contract=fast in GNU C mode); measured by
// strata-nh-vecdot-test: the fused variant matches every row, the unfused one ~37% of rows (Q5K_NOFMA, kept for the test)
enum class UpT : int { Q4K = 0, Q5K = 1, Q5K_NOFMA = 2, Q6K = 3 };
enum class DownT : int { Q50 = 0, IQ4NL = 1 };

// up: one block = 32 rows x 8 lanes; grid (ceil(n_rows/32), n_exp).  w[e] = expert e's first row; q8k = 4 q8_K blocks
// (host-quantised, contiguous as ggml lays them out); out[e*n_rows + r] = raw vec_dot (before relu^2)
template <UpT T>
__global__ void k_up(const uint8_t* const* w, const uint8_t* q8k, int n_rows, int nb, size_t row_bytes, float* out) {
    __shared__ Q8KSh y;
    for (int i = threadIdx.x; i < nb * QK_K; i += blockDim.x) y.qs[i / QK_K][i % QK_K] = (int8_t) q8k[(i / QK_K) * Q8K_BYTES + 4 + i % QK_K];
    for (int i = threadIdx.x; i < nb * 16; i += blockDim.x) y.bsums[i / 16][i % 16] = ld16s(q8k + (i / 16) * Q8K_BYTES + 260 + 2 * (i % 16));
    if (threadIdx.x < nb) {
        float f;
        const uint32_t u = ld32(q8k + threadIdx.x * Q8K_BYTES);
        memcpy(&f, &u, 4);
        y.d[threadIdx.x] = f;
    }
    __syncthreads();
    const int e = blockIdx.y, m = threadIdx.x & 7;
    const int r = blockIdx.x * (blockDim.x / 8) + threadIdx.x / 8;
    if (r >= n_rows) return;   // whole 8-thread groups exit together
    const uint8_t* row = w[e] + (size_t) r * row_bytes;
    float s;
    if constexpr (T == UpT::Q4K) s = dot_q4K(row, nb, y, m);
    else if constexpr (T == UpT::Q5K) s = dot_q5K<true>(row, nb, y, m);
    else if constexpr (T == UpT::Q5K_NOFMA) s = dot_q5K<false>(row, nb, y, m);
    else s = dot_q6K(row, nb, y, m);
    if (m == 0) out[(size_t) e * n_rows + r] = s;
}

// relu^2 then quantize_row_q8_0 (AVX2 path): one warp per 32-element block; grid (n_blocks/8, n_exp), 256 threads
__global__ void k_relu2_q8_0(const float* in, int n, uint8_t* out) {
    const int e = blockIdx.y, lane = threadIdx.x & 31;
    const int blk = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int nblk = n / 32;
    if (blk >= nblk) return;
    float v = in[(size_t) e * n + 32 * blk + lane];
    v = v > 0.0f ? v : 0.0f;
    v = __fmul_rn(v, v);
    float mx = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    const float d = __fdiv_rn(mx, 127.0f);
    const float id = mx != 0.0f ? __fdiv_rn(127.0f, mx) : 0.0f;
    int q = __float2int_rn(__fmul_rn(v, id));
    q = q < -128 ? -128 : (q > 127 ? 127 : q);
    uint8_t* b = out + ((size_t) e * nblk + blk) * Q80_BYTES;
    // fp16 d from lanes 0/1 (one byte each).  Writing both d bytes from lane 0 after a __syncwarp() lost the low byte
    // with nvcc 12.9 / sm_120 (bench/phase3 dbg q80dbg3.cu: variant 0 wrong, this variant right)
    const unsigned int raw = (unsigned int) __half_as_ushort(__float2half_rn(d));
    if (lane < 2) b[lane] = (uint8_t) (raw >> (8 * lane));
    b[2 + lane] = (uint8_t) (int8_t) q;
}

// down: one block = 32 rows x 8 lanes; grid (ceil(n_rows/32), n_exp); yq[e] = expert e's q8_0 input (nb blocks);
// out[col[e] * n_rows + r]: column of the selection slot the expert fills
template <DownT T>
__global__ void k_down(const uint8_t* const* w, const uint8_t* yq, int n_rows, int nb, size_t row_bytes, const int* col,
                       float* out) {
    const int e = blockIdx.y, m = threadIdx.x & 7;
    const int r = blockIdx.x * (blockDim.x / 8) + threadIdx.x / 8;
    if (r >= n_rows) return;
    const uint8_t* row = w[e] + (size_t) r * row_bytes;
    const uint8_t* y = yq + (size_t) e * nb * Q80_BYTES;
    const float s = T == DownT::Q50 ? dot_q50(row, nb, y, m) : dot_iq4nl(row, nb, y, m);
    if (m == 0) out[(size_t) col[e] * n_rows + r] = s;
}

}  // namespace nhx
