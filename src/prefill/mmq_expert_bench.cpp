// src/prefill/mmq_expert_bench.cpp - the prompt path's routed-expert products (MMQ) timed at the engine's shapes.
//
//     mmq_expert_bench [rows per expert ...]
//
// One MMQ group as the prompt path launches it: 16 experts, R rows each, in one activation array in expert order;
// gate/up [1280 x 2560] and down [2560 x 640] in the formats of Lamina's Flash-Next files: UD-Q4_K_XL's Q4_K / Q5_K
// gate/up and Q5_1 / Q8_0 down, IQ3_S's IQ3_XXS / IQ3_S / IQ2_S / IQ4_XS gate/up and IQ4_NL / Q2_0 down.
// Synthetic weights of the right format (valid scales), random activations of row-varying magnitude.  Per product:
// the median of 25 runs, the rate in int8-equivalent TOPS (2 x rows x out x in) and in weight GB/s.  Default R: 160
// (an 8K chunk: 8192 x 10 / 512), 320 (16K), 626 (32K).
//
//     mmq_expert_bench --dump <file> [remote 0|1] [gate/up type] [down type]
//
// A chunk's real routing: the rows of each layer's experts as experts.cpp batches them (consecutive ids, at most 16
// experts and 8,192 rows a batch; an expert of more rows left out: the engine multiplies it in FP16), from a dump of
// "<remote> <layer> <t_begin> <experts> <rows ...>" lines (the first call of each layer and runner), against the same
// experts ordered by row count first (descending); the gate/up and down products of all layers, the median of 9 runs.
// Each order also with its batches of experts of at most 8 rows on the verify windows' grouped kernel
// (native_expert_grouped from the rows' q8_1 blocks), the others on MMQ.
// Types by ggml number (default 12 Q4_K and 7 Q5_1: UD-Q4_K_XL's).
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace mmq = strata::prefill::mmq;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// valid synthetic weights: random bytes with each block's leading f16 scale (every format here starts with one)
// overwritten by a small positive value in the model's range
std::vector<uint8_t> weights(ggml_type t, int64_t rows, int64_t cols, int experts, std::mt19937& rng) {
    const size_t per = mmq::matrix_bytes(t, rows, cols);
    std::vector<uint8_t> w((size_t) experts * per + 4096);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    const size_t bs = ggml_type_size(t), nblk = (size_t) experts * per / bs;
    for (size_t i = 0; i < nblk; ++i) {
        const uint16_t d = 0x2000 + (uint16_t) byte(rng);   // f16 ~ 0.0078..0.016
        std::memcpy(w.data() + i * bs, &d, 2);
    }
    return w;
}

struct Case {
    const char* name;
    ggml_type t;
    int64_t rows, cols;
};

double run_case(const Case& c, int R, std::mt19937& rng, float* ms_out) {
    constexpr int E = 16;
    const int T = E * R;
    const size_t per = mmq::matrix_bytes(c.t, c.rows, c.cols);
    std::vector<uint8_t> w = weights(c.t, c.rows, c.cols, E, rng);
    std::vector<float> x((size_t) T * c.cols);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> mag(0.1f, 10.f);
    for (int t = 0; t < T; ++t) { const float m = mag(rng); for (int64_t k = 0; k < c.cols; ++k) x[(size_t) t * c.cols + k] = nd(rng) * m; }
    std::vector<int32_t> bounds(E + 1);
    for (int e = 0; e <= E; ++e) bounds[e] = e * R;

    uint8_t* dw; float *dx, *dd; void* dq; int32_t *dids, *db;
    ck(cudaMalloc(&dw, w.size()), "w"); ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMalloc(&dx, x.size() * 4), "x"); ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMalloc(&dq, mmq::q8_bytes(T, c.cols)), "xq");
    ck(cudaMalloc(&dd, (size_t) T * c.rows * 4), "dst");
    ck(cudaMalloc(&dids, (size_t) T * 4), "ids");
    ck(cudaMalloc(&db, (E + 1) * 4), "bounds");
    ck(cudaMemcpy(db, bounds.data(), (E + 1) * 4, cudaMemcpyHostToDevice), "bounds");
    mmq::iota(dids, T, nullptr);
    mmq::quantize(dx, nullptr, dq, c.t, c.cols, c.cols, T, nullptr);
    mmq::Context ctx;
    mmq::Product p;
    p.w = dw; p.type = c.t; p.w_rows = c.rows; p.w_cols = c.cols; p.expert_bytes = per; p.n = E;
    p.xq = dq; p.bounds = db; p.ids = dids; p.total_rows = T; p.max_rows = R; p.dst = dd; p.ld_dst = c.rows;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> ts;
    for (int i = 0; i < 28; ++i) {
        cudaEventRecord(e0);
        ctx.run(p, nullptr);
        cudaEventRecord(e1);
        ck(cudaEventSynchronize(e1), "run");
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (i >= 3) ts.push_back(ms);
    }
    std::sort(ts.begin(), ts.end());
    const float ms = ts[ts.size() / 2];
    if (ms_out) *ms_out = ms;
    cudaFree(dw); cudaFree(dx); cudaFree(dq); cudaFree(dd); cudaFree(dids); cudaFree(db);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 2.0 * T * (double) c.rows * c.cols / (ms * 1e9);
}

// one layer's batches: [first expert, last + 1) of `rows` in the given order
std::vector<std::pair<size_t, size_t>> batches_of(const std::vector<int>& rows) {
    std::vector<std::pair<size_t, size_t>> b;
    for (size_t j0 = 0; j0 < rows.size();) {
        size_t j1 = j0 + 1;
        int64_t r = rows[j0];
        while (j1 < rows.size() && j1 - j0 < 16 && r + rows[j1] <= 8192) r += rows[j1++];
        b.emplace_back(j0, j1);
        j0 = j1;
    }
    return b;
}

int run_dump(const char* path, int remote, ggml_type gt, ggml_type dt) {
    std::ifstream f(path);
    std::vector<std::vector<int>> layers;   // the first call of each layer of this runner
    std::vector<int> seen;
    for (std::string line; std::getline(f, line);) {
        std::istringstream s(line);
        int rem = 0, layer = 0, t0 = 0;
        size_t n = 0;
        s >> rem >> layer >> t0 >> n;
        if (rem != remote || t0 != 0 || std::find(seen.begin(), seen.end(), layer) != seen.end()) continue;
        std::vector<int> rows(n);
        for (auto& r : rows) s >> r;
        rows.erase(std::remove_if(rows.begin(), rows.end(), [](int r) { return r <= 0 || r > 8192; }), rows.end());
        seen.push_back(layer);
        layers.push_back(rows);
    }
    if (layers.empty()) { std::fprintf(stderr, "no layers of runner %d in %s\n", remote, path); return 1; }
    std::mt19937 rng(7);
    constexpr int64_t N = 2560, FF2 = 1280, FF = 640, RMAX = 8192;
    const size_t gub = mmq::matrix_bytes(gt, FF2, N), db = mmq::matrix_bytes(dt, N, FF);
    std::vector<uint8_t> wg = weights(gt, FF2, N, 16, rng), wd = weights(dt, N, FF, 16, rng);
    std::vector<float> x((size_t) RMAX * N);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    uint8_t *dwg, *dwd; float *dx, *dgu, *dd; void *dq, *dqh; int32_t *dids, *db_dev;
    ck(cudaMalloc(&dwg, wg.size()), "w"); ck(cudaMemcpy(dwg, wg.data(), wg.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMalloc(&dwd, wd.size()), "w"); ck(cudaMemcpy(dwd, wd.data(), wd.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMalloc(&dx, x.size() * 4), "x"); ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMalloc(&dq, mmq::q8_bytes(RMAX, N)), "xq"); ck(cudaMalloc(&dqh, mmq::q8_bytes(RMAX, FF)), "hq");
    ck(cudaMalloc(&dgu, (size_t) RMAX * FF2 * 4), "gu"); ck(cudaMalloc(&dd, (size_t) RMAX * N * 4), "d");
    ck(cudaMalloc(&dids, (size_t) RMAX * 4), "ids");
    mmq::iota(dids, RMAX, nullptr);
    mmq::quantize(dx, nullptr, dq, gt, N, N, RMAX, nullptr);
    mmq::quantize(dx, nullptr, dqh, dt, FF, N, RMAX, nullptr);
    mmq::Context ctx;
    // the grouped kernel: 16 blobs [gate | up | down] from the same weights, the rows as q8_1 blocks, a plan per batch
    const strata::kernels::NativeExpertLayout L = strata::kernels::native_expert_layout((int) gt, (int) dt, N, FF);
    const size_t stride = (L.bytes + 255) & ~(size_t) 255;
    uint8_t* dblob; void *dx81, *dscr; int32_t* dgn;
    ck(cudaMalloc(&dblob, 16 * stride), "blobs");
    for (int e = 0; e < 16; ++e) {
        ck(cudaMemcpy(dblob + e * stride, wg.data() + e * gub, gub, cudaMemcpyHostToDevice), "blob");
        ck(cudaMemcpy(dblob + e * stride + L.down_off, wd.data() + e * db, db, cudaMemcpyHostToDevice), "blob");
    }
    ck(cudaMalloc(&dx81, (size_t) RMAX * (N / 32) * 36), "x81");
    strata::kernels::quantize_q8_1_rows(dx, RMAX, N, dx81, nullptr);
    ck(cudaMalloc(&dscr, strata::kernels::native_expert_scratch_bytes(128, FF)), "scratch");
    std::vector<unsigned long long> h_ptr(16);
    for (int e = 0; e < 16; ++e) h_ptr[e] = (unsigned long long) (dblob + e * stride);
    unsigned long long* dptr;
    ck(cudaMalloc(&dptr, 16 * 8), "ptr");
    ck(cudaMemcpy(dptr, h_ptr.data(), 16 * 8, cudaMemcpyHostToDevice), "ptr");
    ck(cudaMalloc(&dgn, 17 * 4), "n");
    std::vector<int32_t> h_n(17);
    for (int i = 0; i <= 16; ++i) h_n[i] = i;
    ck(cudaMemcpy(dgn, h_n.data(), 17 * 4, cudaMemcpyHostToDevice), "n");
    // every batch's bounds, uploaded once: [order][layer][batch] -> offset in one device array
    struct B { int n; int64_t rows, max_rows; size_t at; };
    std::vector<int32_t> h_bounds;
    std::vector<std::vector<B>> plan(2);
    for (int order = 0; order < 2; ++order)
        for (const auto& rows0 : layers) {
            std::vector<int> rows = rows0;
            if (order == 1) std::sort(rows.begin(), rows.end(), std::greater<int>());
            for (const auto& [j0, j1] : batches_of(rows)) {
                B b{(int) (j1 - j0), 0, 0, h_bounds.size()};
                h_bounds.push_back(0);
                for (size_t j = j0; j < j1; ++j) {
                    b.rows += rows[j];
                    b.max_rows = std::max<int64_t>(b.max_rows, rows[j]);
                    h_bounds.push_back((int32_t) b.rows);
                }
                plan[(size_t) order].push_back(b);
            }
        }
    ck(cudaMalloc(&db_dev, h_bounds.size() * 4), "bounds");
    ck(cudaMemcpy(db_dev, h_bounds.data(), h_bounds.size() * 4, cudaMemcpyHostToDevice), "bounds");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    int64_t total = 0, n_exp = 0;
    for (const auto& l : layers) { n_exp += (int64_t) l.size(); for (int r : l) total += r; }
    std::printf("mmq_expert_bench --dump on %s: runner %d, %zu layers, %lld experts, %lld rows (%.1f an expert), %s / %s\n",
                prop.name, remote, layers.size(), (long long) n_exp, (long long) total, (double) total / (double) n_exp,
                ggml_type_name(gt), ggml_type_name(dt));
    double ms_of[4] = {0, 0, 0, 0};
    for (int arm = 0; arm < 4; ++arm) {
        const int order = arm % 2;
        const bool grouped = arm >= 2;
        std::vector<float> ts;
        int n_small = 0;
        for (int it = 0; it < 12; ++it) {
            cudaEventRecord(e0);
            n_small = 0;
            for (const B& b : plan[(size_t) order]) {
                if (grouped && b.max_rows <= 8) {
                    ++n_small;
                    strata::kernels::native_expert_grouped(L, dptr, db_dev + b.at, dgn + b.n, dids, dids, b.n, b.rows,
                                                           dx81, (N / 32) * 36, dscr, dd, nullptr);
                    continue;
                }
                mmq::Product p;
                p.w = dwg; p.type = gt; p.w_rows = FF2; p.w_cols = N; p.expert_bytes = gub; p.n = b.n;
                p.xq = dq; p.bounds = db_dev + b.at; p.ids = dids; p.total_rows = b.rows; p.max_rows = b.max_rows;
                p.dst = dgu; p.ld_dst = FF2;
                ctx.run(p, nullptr);
                p.w = dwd; p.type = dt; p.w_rows = N; p.w_cols = FF; p.expert_bytes = db; p.xq = dqh;
                p.dst = dd; p.ld_dst = N;
                ctx.run(p, nullptr);
            }
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "run");
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (it >= 3) ts.push_back(ms);
        }
        std::sort(ts.begin(), ts.end());
        ms_of[arm] = ts[ts.size() / 2];
        std::printf("  %-22s %-13s %4zu batches (%4d grouped) %8.2f ms (%+.1f%%)\n",
                    order == 0 ? "id order (the engine)" : "by row count", grouped ? "grouped <= 8" : "MMQ",
                    plan[(size_t) order].size(), grouped ? n_small : 0, ms_of[arm],
                    100.0 * (ms_of[arm] / ms_of[0] - 1.0));
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2 && std::strcmp(argv[1], "--dump") == 0)
        return run_dump(argv[2], argc > 3 ? std::atoi(argv[3]) : 0, argc > 4 ? (ggml_type) std::atoi(argv[4]) : GGML_TYPE_Q4_K,
                        argc > 5 ? (ggml_type) std::atoi(argv[5]) : GGML_TYPE_Q5_1);
    std::vector<int> Rs;
    for (int i = 1; i < argc; ++i) Rs.push_back(std::atoi(argv[i]));
    if (Rs.empty()) Rs = {160, 320, 626};
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("mmq_expert_bench on %s: 16 experts a launch, medians of 25\n", prop.name);
    const Case cases[] = {{"q4_K gate/up", GGML_TYPE_Q4_K, 1280, 2560},      {"q5_K gate/up", GGML_TYPE_Q5_K, 1280, 2560},
                          {"q5_1 down", GGML_TYPE_Q5_1, 2560, 640},          {"q8_0 down", GGML_TYPE_Q8_0, 2560, 640},
                          {"iq3_xxs gate/up", GGML_TYPE_IQ3_XXS, 1280, 2560}, {"iq3_s gate/up", GGML_TYPE_IQ3_S, 1280, 2560},
                          {"iq2_s gate/up", GGML_TYPE_IQ2_S, 1280, 2560},    {"iq4_xs gate/up", GGML_TYPE_IQ4_XS, 1280, 2560},
                          {"iq4_nl down", GGML_TYPE_IQ4_NL, 2560, 640},      {"q2_0 down", GGML_TYPE_Q2_0, 2560, 640}};
    std::mt19937 rng(7);
    for (int R : Rs)
        for (const Case& c : cases) {
            float ms = 0;
            const double tops = run_case(c, R, rng, &ms);
            const double gbs = 16.0 * (double) mmq::matrix_bytes(c.t, c.rows, c.cols) / (ms * 1e6);
            std::printf("  R %4d  %-16s %8.3f ms  %7.1f TOPS  %7.1f GB/s of weights\n", R, c.name, ms, tops, gbs);
        }
    return 0;
}
