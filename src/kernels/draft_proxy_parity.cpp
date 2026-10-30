// src/kernels/draft_proxy_parity.cpp - draft_proxy (the drafter's head argmax through a 2-bit copy of its rows) against
// native_mmvq + argmax_rows + row_top_prob_split, on synthetic Q6_K and Q8_0 heads of rows of 2560: with one row's
// logit clearly ahead (the activations lean on its weights), the pick is the exact argmax and the probability within
// 0.05 of the exact one (the copy's terms for the rows it does not rescore); every launch repeats bitwise.  Heads of
// 40,525 rows (this branch's draft subset), 49,184 (6,148 candidates: 13 for some of the 512 rescoring warps) and
// 106,299 (upstream's CJK subset), each with the copy's top-1 and the exact head timed.
// Usage: draft_proxy_parity --selftest
#include "lamina/kernels/draft_proxy.hpp"
#include "lamina/kernels/native_mmvq.hpp"
#include "lamina/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int K = 2560;

#define CK(x)                                                         \
    do {                                                              \
        const cudaError_t e_ = (x);                                   \
        if (e_ != cudaSuccess) {                                      \
            std::printf("%s: %s\n", #x, cudaGetErrorString(e_));      \
            return 1;                                                 \
        }                                                             \
    } while (0)

float half_to_float(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else {   // subnormal
            float f = std::ldexp((float) m, -24);
            return s ? -f : f;
        }
    } else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// row r's weights, as ggml's dequantize_row_q6_K / q8_0
void dequant(int type, const uint8_t* row, float* w) {
    if (type == 8) {
        for (int b = 0; b < K / 32; ++b) {
            uint16_t d;
            std::memcpy(&d, row + b * 34, 2);
            for (int l = 0; l < 32; ++l) w[b * 32 + l] = half_to_float(d) * (float) (int8_t) row[b * 34 + 2 + l];
        }
        return;
    }
    for (int b = 0; b < K / 256; ++b) {
        const uint8_t* blk = row + b * 210;
        const uint8_t *ql = blk, *qh = blk + 128;
        const int8_t* sc = (const int8_t*) (blk + 192);
        uint16_t dh;
        std::memcpy(&dh, blk + 208, 2);
        const float d = half_to_float(dh);
        float* y = w + b * 256;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                y[l + 0] = d * sc[is + 0] * (float) ((int) ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                y[l + 32] = d * sc[is + 2] * (float) ((int) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                y[l + 64] = d * sc[is + 4] * (float) ((int) ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                y[l + 96] = d * sc[is + 6] * (float) ((int) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

int run(int type, int rows) {
    const int ROWS = rows;
    std::mt19937 rng(type * 7919 + ROWS);
    using namespace lamina::kernels;
    const size_t rb = native_mmvq_weight_bytes(type, K, 1);
    std::vector<uint8_t> hw((size_t) ROWS * rb);
    for (auto& v : hw) v = (uint8_t) rng();
    const uint16_t dbits = 0x1419;   // a half of ~0.00125
    for (int r = 0; r < ROWS; ++r) {
        uint8_t* row = hw.data() + (size_t) r * rb;
        if (type == 14)
            for (int b = 0; b < K / 256; ++b) std::memcpy(row + b * 210 + 208, &dbits, 2);
        else
            for (int b = 0; b < K / 32; ++b) std::memcpy(row + b * 34, &dbits, 2);
    }
    cudaStream_t st;   // native_mmvq takes an explicit stream
    CK(cudaStreamCreate(&st));
    void *w, *proxy, *scratch, *xq, *arg, *top;
    float *dx, *logits, *prob;
    int32_t *ids, *map;
    CK(cudaMalloc(&w, hw.size()));
    CK(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&proxy, draft_proxy_bytes(ROWS)));
    CK(cudaMalloc(&scratch, draft_proxy_scratch_bytes(ROWS)));
    CK(cudaMemset(scratch, 0, draft_proxy_scratch_bytes(ROWS)));
    CK(cudaMalloc(&xq, K / 32 * 36));
    CK(cudaMalloc(&dx, K * 4));
    CK(cudaMalloc(&logits, (size_t) ROWS * 4));
    CK(cudaMalloc(&prob, 16));
    CK(cudaMalloc(&ids, 16));
    CK(cudaMalloc(&map, (size_t) ROWS * 4));
    CK(cudaMalloc(&arg, argmax_rows_scratch_bytes(1)));
    CK(cudaMalloc(&top, row_top_prob_scratch_bytes(1)));
    CK(cudaMemset(top, 0, row_top_prob_scratch_bytes(1)));
    std::vector<int32_t> hmap(ROWS);
    for (int i = 0; i < ROWS; ++i) hmap[i] = 3 * i + 1;
    CK(cudaMemcpy(map, hmap.data(), hmap.size() * 4, cudaMemcpyHostToDevice));
    draft_proxy_pack(type, w, ROWS, proxy, st);
    CK(cudaDeviceSynchronize());
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(K), wr(K), hl(ROWS);
    // the other rows' logits spread with std ~2 (as the draft head's do): the activations' noise against a row's norm
    dequant(type, hw.data(), wr.data());
    double n0 = 0;
    for (float f : wr) n0 += (double) f * f;
    const float noise = (float) (2.0 / std::sqrt(n0));
    int bad = 0, close = 0;
    double pmax = 0;
    for (int v = 0; v < 40; ++v) {
        // the activations lean on row r's weights: its logit L = 10 .. 23.7 against the others' ~N(0, 2) (the best of
        // them ~8.5)
        const int r = (int) (rng() % ROWS);
        dequant(type, hw.data() + (size_t) r * rb, wr.data());
        double n2 = 0;
        for (float f : wr) n2 += (double) f * f;
        const double L = 10.0 + 0.35 * v;
        for (int i = 0; i < K; ++i) x[i] = nd(rng) * noise + (float) (wr[i] * L / n2);
        CK(cudaMemcpy(dx, x.data(), K * 4, cudaMemcpyHostToDevice));
        native_quantize_q8_1(dx, xq, K, 1, st);
        native_mmvq(type, w, xq, logits, K, ROWS, 1, st);
        argmax_rows(logits, 1, ROWS, arg, ids, st);
        row_top_prob_split(logits, 1, ROWS, ids, prob, top, st);
        map_ids(ids, map, 1, st);
        draft_proxy_top1(type, proxy, w, xq, ROWS, map, ids + 1, prob + 1, scratch, st);
        draft_proxy_top1(type, proxy, w, xq, ROWS, map, ids + 2, prob + 2, scratch, st);
        CK(cudaDeviceSynchronize());
        int32_t hid[3];
        float hp[3];
        CK(cudaMemcpy(hid, ids, 12, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hp, prob, 12, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hl.data(), logits, hl.size() * 4, cudaMemcpyDeviceToHost));
        float t1 = -INFINITY, t2 = -INFINITY;   // the exact top two
        for (float l : hl) {
            if (l > t1) { t2 = t1; t1 = l; }
            else if (l > t2) t2 = l;
        }
        // a clear lead (3 logits, beyond the copy's error) must give the exact pick; within it the copy may pick a
        // near-tie (counted)
        const bool pick = hid[1] == hid[0];
        const bool repeat = hid[2] == hid[1] && std::memcmp(&hp[2], &hp[1], 4) == 0;
        if (pick) pmax = std::max(pmax, (double) std::fabs(hp[1] - hp[0]));
        close += !pick && t1 - t2 < 3.0f;
        if (!repeat || (!pick && t1 - t2 >= 3.0f) || (pick && std::fabs(hp[1] - hp[0]) > 0.05f)) {
            ++bad;
            std::printf("  vector %d (lead %.2f): exact %d (p %.4f), copy %d (p %.4f), again %d (p %.4f)\n", v, t1 - t2,
                        hid[0], hp[0], hid[1], hp[1], hid[2], hp[2]);
        }
    }
    std::printf("draft_proxy %s: %s (%d rows; 40 vectors; the probability within %.2e of the exact one; %d near-ties "
                "picked otherwise)\n", type == 14 ? "Q6_K" : "Q8_0", bad ? "FAIL" : "ok", ROWS, pmax, close);
    // ---- timing: the copy's top-1 against the exact head over the same rows (the last vector's activations): what the
    // 2-bit copy buys.  The exact path is native_mmvq + argmax_rows + row_top_prob_split + map_ids, as above.
    {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        const int iters = 200;
        for (int i = 0; i < 20; ++i) draft_proxy_top1(type, proxy, w, xq, ROWS, map, ids, prob, scratch, st);
        cudaEventRecord(e0, st);
        for (int i = 0; i < iters; ++i) draft_proxy_top1(type, proxy, w, xq, ROWS, map, ids, prob, scratch, st);
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        float copy_ms;
        cudaEventElapsedTime(&copy_ms, e0, e1);
        auto exact = [&] {
            native_mmvq(type, w, xq, logits, K, ROWS, 1, st);
            argmax_rows(logits, 1, ROWS, arg, ids, st);
            row_top_prob_split(logits, 1, ROWS, ids, prob, top, st);
            map_ids(ids, map, 1, st);
        };
        for (int i = 0; i < 20; ++i) exact();
        cudaEventRecord(e0, st);
        for (int i = 0; i < iters; ++i) exact();
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        float exact_ms;
        cudaEventElapsedTime(&exact_ms, e0, e1);
        std::printf("  timing (%d rows): copy %.1f us, exact head %.1f us, %.1fx\n", ROWS, copy_ms * 1000.0f / iters,
                    exact_ms * 1000.0f / iters, exact_ms / copy_ms);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }
    cudaFree(w); cudaFree(proxy); cudaFree(scratch); cudaFree(xq); cudaFree(dx); cudaFree(logits); cudaFree(prob);
    cudaFree(ids); cudaFree(map); cudaFree(arg); cudaFree(top);
    cudaStreamDestroy(st);
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::strcmp(argv[1], "--selftest") != 0) {
        std::printf("usage: draft_proxy_parity --selftest\n");
        return 2;
    }
    const int sizes[] = {40525, 49184, 106299};
    int bad = 0;
    for (const int rows : sizes) {
        std::printf("\n== %d rows ==\n", rows);
        bad |= run(14, rows);
        bad |= run(8, rows);
    }
    return bad ? 1 : 0;
}
