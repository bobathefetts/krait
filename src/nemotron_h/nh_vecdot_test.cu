// src/nemotron_h/nh_vecdot_test.cu - strata-nh-vecdot-test: checks the CPU-exact expert kernels (nh_cpuexact.cuh)
// bit for bit against ggml-cpu's own functions (type_traits_cpu from_float / vec_dot of the same libggml-cpu.so the
// engine runs), on the real routed experts of every MoE layer of the model and random activations.  Exit 0 only if
// every output row and every q8_0 byte matches.  Also times the kernels for a 22-expert decode layer.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/nemotron_h/layout.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include "nh_cpuexact.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace nh = strata::nemotron_h;

#define CK(x)                                                                                                       \
    do {                                                                                                            \
        cudaError_t e_ = (x);                                                                                       \
        if (e_ != cudaSuccess) {                                                                                    \
            std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);                 \
            std::exit(3);                                                                                           \
        }                                                                                                           \
    } while (0)

static void launch_up(nhx::UpT t, dim3 g, const uint8_t* const* w, const uint8_t* q8k, int nr, int nb, size_t rb, float* o) {
    switch (t) {
    case nhx::UpT::Q4K: nhx::k_up<nhx::UpT::Q4K><<<g, 256>>>(w, q8k, nr, nb, rb, o); break;
    case nhx::UpT::Q5K: nhx::k_up<nhx::UpT::Q5K><<<g, 256>>>(w, q8k, nr, nb, rb, o); break;
    case nhx::UpT::Q5K_NOFMA: nhx::k_up<nhx::UpT::Q5K_NOFMA><<<g, 256>>>(w, q8k, nr, nb, rb, o); break;
    case nhx::UpT::Q6K: nhx::k_up<nhx::UpT::Q6K><<<g, 256>>>(w, q8k, nr, nb, rb, o); break;
    }
}
static void launch_down(nhx::DownT t, dim3 g, const uint8_t* const* w, const uint8_t* yq, int nr, int nb, size_t rb,
                        const int* col, float* o) {
    if (t == nhx::DownT::Q50) nhx::k_down<nhx::DownT::Q50><<<g, 256>>>(w, yq, nr, nb, rb, col, o);
    else nhx::k_down<nhx::DownT::IQ4NL><<<g, 256>>>(w, yq, nr, nb, rb, col, o);
}

int main(int argc, char** argv) {
    std::string model;
    int n_exp = 4, n_x = 2;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--experts" && i + 1 < argc) n_exp = std::stoi(argv[++i]);
        else if (a == "--inputs" && i + 1 < argc) n_x = std::stoi(argv[++i]);
    }
    if (model.empty()) {
        std::fprintf(stderr, "usage: strata-nh-vecdot-test --model SHARD [--experts 4] [--inputs 2]\n");
        return 2;
    }
    ggml_cpu_init();
    const strata::GgufModel gm = strata::GgufModel::open(model);
    const nh::Model m = nh::load_model(gm);
    const int D = m.g.moe_latent, F = m.g.n_ff_exp;   // 1024, 2688
    const auto* tq8k = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const auto* tq80 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    std::mt19937_64 rng(20261022);
    long rows_up = 0, rows_dn = 0, bad_up = 0, bad_up_nofma = 0, bad_q = 0, bad_dn = 0, bad_e2e = 0;
    std::map<std::string, std::pair<long, long>> by_type;   // type -> (rows, mismatches)
    uint8_t *d_w = nullptr, *d_q8k = nullptr, *d_hq = nullptr, *d_wd = nullptr;
    float *d_up = nullptr, *d_dn = nullptr;
    const uint8_t** d_wp = nullptr;
    int* d_col = nullptr;
    const size_t max_exp_bytes = 8u << 20;
    CK(cudaMalloc(&d_w, max_exp_bytes * 22));
    CK(cudaMalloc(&d_wd, max_exp_bytes * 22));
    CK(cudaMalloc(&d_q8k, 4096));
    CK(cudaMalloc(&d_hq, (size_t) 22 * (F / 32) * nhx::Q80_BYTES));
    CK(cudaMalloc(&d_up, sizeof(float) * F * 22));
    CK(cudaMalloc(&d_dn, sizeof(float) * D * 22));
    CK(cudaMalloc(&d_wp, sizeof(void*) * 44));
    CK(cudaMalloc(&d_col, sizeof(int) * 22));
    std::vector<int> col(22);
    for (int i = 0; i < 22; ++i) col[(size_t) i] = i;
    CK(cudaMemcpy(d_col, col.data(), sizeof(int) * 22, cudaMemcpyHostToDevice));
    for (int il = 0; il < m.g.n_layer; ++il) {
        const nh::Layer& L = m.layers[(size_t) il];
        if (L.kind != nh::Kind::Moe) continue;
        const ggml_type tu = (ggml_type) L.up_exps.type, td = (ggml_type) L.down_exps.type;
        const size_t rbu = ggml_row_size(tu, D), rbd = ggml_row_size(td, F);
        const size_t ebu = rbu * F, ebd = rbd * D;
        nhx::UpT ut;
        if (tu == GGML_TYPE_Q4_K) ut = nhx::UpT::Q4K;
        else if (tu == GGML_TYPE_Q5_K) ut = nhx::UpT::Q5K;
        else if (tu == GGML_TYPE_Q6_K) ut = nhx::UpT::Q6K;
        else { std::fprintf(stderr, "layer %d: up type %s not covered\n", il, ggml_type_name(tu)); return 4; }
        nhx::DownT dt;
        if (td == GGML_TYPE_Q5_0) dt = nhx::DownT::Q50;
        else if (td == GGML_TYPE_IQ4_NL) dt = nhx::DownT::IQ4NL;
        else { std::fprintf(stderr, "layer %d: down type %s not covered\n", il, ggml_type_name(td)); return 4; }
        const auto* tu_t = ggml_get_type_traits_cpu(tu);
        const auto* td_t = ggml_get_type_traits_cpu(td);
        std::vector<int> ex((size_t) n_exp);
        for (int e = 0; e < n_exp; ++e) ex[(size_t) e] = (int) (rng() % (uint64_t) m.g.n_expert);
        std::vector<const uint8_t*> wp(44);
        for (int e = 0; e < n_exp; ++e) {
            CK(cudaMemcpy(d_w + (size_t) e * max_exp_bytes, L.up_exps.data + (size_t) ex[(size_t) e] * ebu, ebu, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_wd + (size_t) e * max_exp_bytes, L.down_exps.data + (size_t) ex[(size_t) e] * ebd, ebd, cudaMemcpyHostToDevice));
            wp[(size_t) e] = d_w + (size_t) e * max_exp_bytes;
            wp[22 + (size_t) e] = d_wd + (size_t) e * max_exp_bytes;
        }
        CK(cudaMemcpy(d_wp, wp.data(), sizeof(void*) * 44, cudaMemcpyHostToDevice));
        for (int xi = 0; xi < n_x; ++xi) {
            // activations: gaussian, a scale that varies by input, a few outliers
            std::vector<float> x((size_t) D);
            std::normal_distribution<float> nd(0.0f, xi % 2 ? 0.05f : 1.3f);
            for (auto& v : x) v = nd(rng);
            for (int k = 0; k < 4; ++k) x[(size_t) (rng() % (uint64_t) D)] *= 9.0f;
            std::vector<uint8_t> q8k((size_t) (D / 256) * nhx::Q8K_BYTES);
            tq8k->from_float(x.data(), q8k.data(), D);
            CK(cudaMemcpy(d_q8k, q8k.data(), q8k.size(), cudaMemcpyHostToDevice));
            std::vector<float> up_ref((size_t) n_exp * F), dn_ref((size_t) n_exp * D);
            std::vector<uint8_t> hq_ref((size_t) n_exp * (F / 32) * nhx::Q80_BYTES);
            for (int e = 0; e < n_exp; ++e) {
                const uint8_t* base = L.up_exps.data + (size_t) ex[(size_t) e] * ebu;
                for (int r = 0; r < F; ++r) tu_t->vec_dot(D, &up_ref[(size_t) e * F + r], 0, base + (size_t) r * rbu, 0, q8k.data(), 0, 1);
                std::vector<float> h((size_t) F);
                for (int r = 0; r < F; ++r) {
                    float v = up_ref[(size_t) e * F + r];
                    v = v > 0.f ? v : 0.f;
                    h[(size_t) r] = v * v;
                }
                uint8_t* hq = hq_ref.data() + (size_t) e * (F / 32) * nhx::Q80_BYTES;
                tq80->from_float(h.data(), hq, F);
                const uint8_t* bd = L.down_exps.data + (size_t) ex[(size_t) e] * ebd;
                for (int r = 0; r < D; ++r) td_t->vec_dot(F, &dn_ref[(size_t) e * D + r], 0, bd + (size_t) r * rbd, 0, hq, 0, 1);
            }
            std::vector<float> up_gpu((size_t) n_exp * F), dn_gpu((size_t) n_exp * D);
            std::vector<uint8_t> hq_gpu(hq_ref.size());
            auto run_check_up = [&](nhx::UpT t) {
                launch_up(t, dim3((unsigned) (F / 32), (unsigned) n_exp), d_wp, d_q8k, F, D / 256, rbu, d_up);
                CK(cudaGetLastError());
                CK(cudaMemcpy(up_gpu.data(), d_up, sizeof(float) * up_gpu.size(), cudaMemcpyDeviceToHost));
                long b = 0;
                for (size_t i = 0; i < up_gpu.size(); ++i) b += std::memcmp(&up_gpu[i], &up_ref[i], 4) != 0;
                return b;
            };
            const long bu = run_check_up(ut);
            bad_up += bu;
            if (ut == nhx::UpT::Q5K) bad_up_nofma += run_check_up(nhx::UpT::Q5K_NOFMA), run_check_up(ut);
            rows_up += (long) up_gpu.size();
            auto& bt = by_type[std::string("up ") + ggml_type_name(tu)];
            bt.first += (long) up_gpu.size();
            bt.second += bu;
            nhx::k_relu2_q8_0<<<dim3((unsigned) ((F / 32 + 7) / 8), (unsigned) n_exp), 256>>>(d_up, F, d_hq);
            CK(cudaGetLastError());
            CK(cudaMemcpy(hq_gpu.data(), d_hq, hq_gpu.size(), cudaMemcpyDeviceToHost));
            long bq = 0;
            for (size_t i = 0; i < hq_gpu.size(); ++i) {
                if (hq_gpu[i] != hq_ref[i] && bad_q + bq < 6) {
                    const size_t blk = i / nhx::Q80_BYTES;
                    const float* upv = nullptr;
                    (void) upv;
                    float mx = 0;
                    const size_t e = blk / (size_t) (F / 32), b = blk % (size_t) (F / 32);
                    for (int k = 0; k < 32; ++k) { float v = up_ref[e * F + b * 32 + k]; v = v > 0 ? v * v : 0; mx = std::max(mx, v); }
                    const size_t o = blk * nhx::Q80_BYTES;
                    std::printf("q8_0 diff: layer %d blk %zu byte %zu cpu %d gpu %d  (amax %.9g, d %.9g) cpu d bytes %02x %02x gpu %02x %02x q0 %d/%d\n", il, blk,
                                i % nhx::Q80_BYTES, (int) (int8_t) hq_ref[i], (int) (int8_t) hq_gpu[i], mx, mx / 127.f,
                                hq_ref[o], hq_ref[o + 1], hq_gpu[o], hq_gpu[o + 1], (int) (int8_t) hq_ref[o + 2], (int) (int8_t) hq_gpu[o + 2]);
                }
                bq += hq_gpu[i] != hq_ref[i];
            }
            bad_q += bq;
            // end to end: down from the GPU-quantised input (the engine's path)
            launch_down(dt, dim3((unsigned) (D / 32), (unsigned) n_exp), d_wp + 22, d_hq, D, F / 32, rbd, d_col, d_dn);
            CK(cudaGetLastError());
            CK(cudaMemcpy(dn_gpu.data(), d_dn, sizeof(float) * dn_gpu.size(), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < dn_gpu.size(); ++i) bad_e2e += std::memcmp(&dn_gpu[i], &dn_ref[i], 4) != 0;
            // down from the CPU-quantised input, so a q8_0 mismatch cannot hide a down mismatch
            CK(cudaMemcpy(d_hq, hq_ref.data(), hq_ref.size(), cudaMemcpyHostToDevice));
            launch_down(dt, dim3((unsigned) (D / 32), (unsigned) n_exp), d_wp + 22, d_hq, D, F / 32, rbd, d_col, d_dn);
            CK(cudaGetLastError());
            CK(cudaMemcpy(dn_gpu.data(), d_dn, sizeof(float) * dn_gpu.size(), cudaMemcpyDeviceToHost));
            long bd = 0;
            for (size_t i = 0; i < dn_gpu.size(); ++i) bd += std::memcmp(&dn_gpu[i], &dn_ref[i], 4) != 0;
            bad_dn += bd;
            rows_dn += (long) dn_gpu.size();
            auto& bt2 = by_type[std::string("down ") + ggml_type_name(td)];
            bt2.first += (long) dn_gpu.size();
            bt2.second += bd;
            if (bu || bq || bd)
                std::printf("layer %d x%d: up %s mismatches %ld/%zu, q8_0 bytes %ld, down %s mismatches %ld/%zu "
                            "(first up ref %.9g gpu %.9g)\n", il, xi, ggml_type_name(tu), bu, up_gpu.size(), bq,
                            ggml_type_name(td), bd, dn_gpu.size(), up_ref[0], up_gpu[0]);
        }
    }
    for (auto& kv : by_type) std::printf("%-12s rows %8ld mismatches %ld\n", kv.first.c_str(), kv.second.first, kv.second.second);
    std::printf("up rows %ld mismatches %ld (q5_K unfused-summs variant mismatches %ld); q8_0 byte mismatches %ld; "
                "down rows %ld mismatches %ld; end-to-end (GPU up -> GPU q8_0 -> GPU down) mismatches %ld\n", rows_up, bad_up, bad_up_nofma, bad_q, rows_dn, bad_dn, bad_e2e);
    // timing: 22 experts of the last tested layer, up + quant + down
    {
        std::vector<const uint8_t*> wp(44);
        for (int e = 0; e < 22; ++e) {
            wp[(size_t) e] = d_w + (size_t) (e % n_exp) * max_exp_bytes;
            wp[22 + (size_t) e] = d_wd + (size_t) (e % n_exp) * max_exp_bytes;
        }
        CK(cudaMemcpy(d_wp, wp.data(), sizeof(void*) * 44, cudaMemcpyHostToDevice));
        const nh::Layer& L = m.layers[(size_t) m.g.n_layer - 1];
        const ggml_type tu = (ggml_type) L.up_exps.type, td = (ggml_type) L.down_exps.type;
        const nhx::UpT ut = tu == GGML_TYPE_Q4_K ? nhx::UpT::Q4K : tu == GGML_TYPE_Q5_K ? nhx::UpT::Q5K : nhx::UpT::Q6K;
        const nhx::DownT dt = td == GGML_TYPE_Q5_0 ? nhx::DownT::Q50 : nhx::DownT::IQ4NL;
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        float ms_up = 0, ms_q = 0, ms_dn = 0, ms;
        const int it = 50;
        for (int k = 0; k < it + 5; ++k) {
            CK(cudaEventRecord(a));
            launch_up(ut, dim3((unsigned) (F / 32), 22), d_wp, d_q8k, F, D / 256, ggml_row_size(tu, D), d_up);
            CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b));
            CK(cudaEventElapsedTime(&ms, a, b));
            if (k >= 5) ms_up += ms;
            CK(cudaEventRecord(a));
            nhx::k_relu2_q8_0<<<dim3((unsigned) ((F / 32 + 7) / 8), 22), 256>>>(d_up, F, d_hq);
            CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b));
            CK(cudaEventElapsedTime(&ms, a, b));
            if (k >= 5) ms_q += ms;
            CK(cudaEventRecord(a));
            launch_down(dt, dim3((unsigned) (D / 32), 22), d_wp + 22, d_hq, D, F / 32, ggml_row_size(td, F), d_col, d_dn);
            CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b));
            CK(cudaEventElapsedTime(&ms, a, b));
            if (k >= 5) ms_dn += ms;
        }
        std::printf("timing (22 experts, %s/%s, mean of %d): up %.1f us, relu2+q8_0 %.1f us, down %.1f us\n",
                    ggml_type_name(tu), ggml_type_name(td), it, ms_up * 1e3 / it, ms_q * 1e3 / it, ms_dn * 1e3 / it);
    }
    const bool ok = bad_up == 0 && bad_q == 0 && bad_dn == 0 && bad_e2e == 0;
    std::printf("RESULT %s\n", ok ? "PASS (bit-exact)" : "FAIL");
    return ok ? 0 : 1;
}
