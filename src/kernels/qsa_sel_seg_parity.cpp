// src/kernels/qsa_sel_seg_parity.cpp - the batch window's QSA selection (block scores, top-k, attention) as one
// launch over rows of different slots (qsa_block_scores_seg, qsa_block_topk over the rows, qsa_decode_attn_seg,
// seg_rows.hpp) against one call per row, as phase 3's verify.cpp made them. Every slot has its own indexer keys,
// page table and K/V pools (fp16, int8, q4_0 and int8-K/q4_0-V) and its own context length; scores, selected ids
// and the attention output must be bitwise identical. The fused form resolves each row's state through a resident
// segment table and a permuted row -> slot index, exactly as the engine stages them. Synthetic data, no model.
//
// What each leg's equality rests on (seg_rows.hpp's taxonomy): scores, top-k and the attention chunk are program
// identity (the same __device__ body, the same grid.x; top-k pins its chunk count to the n_q = 1 value exactly as
// the engine's batch call site does, so the fused launch cannot derive a smaller one; the attention splits are
// pinned to the per-row n_q=1 value); the merge is row-independent by construction and this suite pins that too.
// No launcher declines in this tree (the KV format is global and n <= kVerifyMaxT by the window cap), so there is
// no skip-on-decline; with no CUDA device the whole suite skips (77), but a device-enumeration error FAILS.
// Cases run on every visible device: the grids and splits are device-derived.
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/seg_rows.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace k = strata::kernels;
namespace {

void ck(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        std::exit(2);
    }
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T)));
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
// a finite fp16 of magnitude 2^-5 .. 2^2
uint16_t rnd_half(std::mt19937& r) {
    const uint32_t x = r();
    return (uint16_t) (((x & 1u) << 15) | ((10u + (x >> 1) % 7u) << 10) | ((x >> 4) & 0x3ffu));
}

constexpr int64_t kMaxCells = 32768;

struct Slot {
    float *pooled = nullptr, *dead = nullptr;
    int32_t* page_table = nullptr;
    void* bufs[6] = {};
    k::QsaAttnPools pools;
};

Slot make_slot(int fmt, const k::QsaShapes& s, int64_t max_blocks, std::mt19937& r) {
    Slot sl;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> pooled((size_t) (max_blocks * s.idx_dim)), dead((size_t) s.idx_dim);
    for (auto& v : pooled) v = nd(r);
    for (auto& v : dead) v = nd(r);
    sl.pooled = up(pooled);
    sl.dead = up(dead);
    const int64_t pages = kMaxCells / s.page_size;
    std::vector<int32_t> pt((size_t) pages);
    std::iota(pt.begin(), pt.end(), 0);
    std::shuffle(pt.begin(), pt.end(), r);
    // No -1 (non-resident) entries: nothing in this tree ever unmaps a page (no writer of -1 exists), and the
    // attention kernels index the pools by page_table[...] unguarded - upstream's suite tests its elastic
    // eviction's masked cells, a mechanism this tree does not have.
    sl.page_table = up(pt);
    sl.pools.page_table = sl.page_table;
    const size_t rows = (size_t) (pages * s.n_head_kv * s.page_size), hd = (size_t) s.head_dim;
    auto halves = [&](size_t n) { std::vector<uint16_t> h(n); for (auto& v : h) v = rnd_half(r); return h; };
    auto codes = [&](size_t n) { std::vector<int8_t> h(n); for (auto& v : h) v = (int8_t) (r() & 0xff); return h; };
    // our 4-bit pools: codes [row][head][head_dim / 2] and FP16 scales [row][head][head_dim / 32], separate arrays
    auto q4c = [&](size_t n_rows) { return codes(n_rows * (hd / 2)); };
    auto q4s = [&](size_t n_rows) { return halves(n_rows * (hd / k::KV_Q4_GROUP)); };
    if (fmt == 0) {
        sl.pools.k_pool = (const uint16_t*) (sl.bufs[0] = up(halves(rows * hd)));
        sl.pools.v_pool = (const uint16_t*) (sl.bufs[1] = up(halves(rows * hd)));
    } else if (fmt == 1 || fmt == 3) {
        sl.pools.k_q = (const int8_t*) (sl.bufs[0] = up(codes(rows * hd)));
        sl.pools.k_scale = (const uint16_t*) (sl.bufs[1] = up(halves(rows * hd / k::KV_Q8_GROUP)));
        if (fmt == 1) {
            sl.pools.v_q = (const int8_t*) (sl.bufs[2] = up(codes(rows * hd)));
            sl.pools.v_scale = (const uint16_t*) (sl.bufs[3] = up(halves(rows * hd / k::KV_Q8_GROUP)));
        } else {
            sl.pools.v_q4 = (const uint8_t*) (sl.bufs[2] = up(q4c(rows)));
            sl.pools.v_q4s = (const uint16_t*) (sl.bufs[3] = up(q4s(rows)));
        }
    } else {
        sl.pools.k_q4 = (const uint8_t*) (sl.bufs[0] = up(q4c(rows)));
        sl.pools.k_q4s = (const uint16_t*) (sl.bufs[1] = up(q4s(rows)));
        sl.pools.v_q4 = (const uint8_t*) (sl.bufs[2] = up(q4c(rows)));
        sl.pools.v_q4s = (const uint16_t*) (sl.bufs[3] = up(q4s(rows)));
    }
    return sl;
}

void free_slot(Slot& sl) {
    ck(cudaFree(sl.pooled)); ck(cudaFree(sl.dead)); ck(cudaFree(sl.page_table));
    for (void* b : sl.bufs) if (b) ck(cudaFree(b));
}

// one case: `m` rows of different slots, KV format `fmt`, contexts `ctx[i]`
bool check(int fmt, int m, const int64_t* ctx, uint32_t seed) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const int64_t max_blocks = kMaxCells / s.idx_block + 2;
    const int64_t NH = s.n_head, HD = s.head_dim, QI = s.idx_n_head * s.idx_dim;
    const int64_t sf = (int64_t) k::qsa_decode_attn_scratch_floats(cap, s);
    std::mt19937 r(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<Slot> slots;
    for (int i = 0; i < m; ++i) slots.push_back(make_slot(fmt, s, max_blocks, r));
    std::vector<float> qidx((size_t) (m * QI)), qcur((size_t) (m * NH * HD));
    for (auto& v : qidx) v = nd(r);
    for (auto& v : qcur) v = nd(r);
    std::vector<int32_t> steps((size_t) (m * k::kStepCount));
    for (int i = 0; i < m; ++i) k::qsa_step_fill(steps.data() + i * k::kStepCount, ctx[i] - 1, s);
    float* d_qidx = up(qidx);
    float* d_qcur = up(qcur);
    int32_t* d_steps = up(steps);
    // the resident table (one entry per slot) and a permuted row -> slot index: the fused kernels must resolve
    // every row through them, never through launch order
    std::vector<k::SegRow> seg_h((size_t) m);
    for (int t = 0; t < m; ++t) {
        seg_h[(size_t) t].idx_pooled = slots[t].pooled;
        seg_h[(size_t) t].idx_dead = slots[t].dead;
        seg_h[(size_t) t].kv = slots[t].pools;
    }
    const k::SegRow* seg_d = up(seg_h);
    std::vector<int32_t> row_slot((size_t) m);
    std::iota(row_slot.begin(), row_slot.end(), 0);
    std::shuffle(row_slot.begin(), row_slot.end(), r);
    const int32_t* idx_d = up(row_slot);
    float *sc[2], *scratch[2], *attn[2];
    int32_t* sel[2];
    for (int v = 0; v < 2; ++v) {
        ck(cudaMalloc(&sc[v], (size_t) (m * max_blocks) * 4));
        ck(cudaMalloc(&sel[v], (size_t) (m * cap) * 4));
        ck(cudaMalloc(&scratch[v], (size_t) (m * sf) * 4));
        ck(cudaMalloc(&attn[v], (size_t) (m * NH * HD) * 4));
        ck(cudaMemset(sc[v], 0, (size_t) (m * max_blocks) * 4));
        ck(cudaMemset(sel[v], 0, (size_t) (m * cap) * 4));
        ck(cudaMemset(attn[v], 0, (size_t) (m * NH * HD) * 4));
    }
    // 0: one call per row (phase 3's batch-window loop), row t against the slot the window's index names for it -
    // the same state variant 1 resolves through the table, so the per-row memcmp checks the indirection too
    for (int t = 0; t < m; ++t) {
        const Slot& sl = slots[row_slot[t]];
        k::qsa_block_scores(sl.pooled, sl.dead, d_qidx + t * QI, d_steps + t * k::kStepCount, 1, max_blocks, s,
                            sc[0] + t * max_blocks, nullptr, k::qsa_score_grid_blocks);
        k::qsa_block_topk(sc[0] + t * max_blocks, d_steps + t * k::kStepCount, 1, max_blocks, cap, s, sel[0] + t * cap,
                          nullptr);
        k::qsa_decode_attn_batch(d_qcur + t * NH * HD, sl.pools, sel[0] + t * cap, d_steps + t * k::kStepCount, cap, s,
                                 scratch[0] + t * sf, attn[0] + t * NH * HD, 1, nullptr);
    }
    // 1: one launch per step over the rows, state through the table
    ck(cudaDeviceSynchronize());   // attribute failures to v0 vs v1
    std::puts("  v1 fused");
    std::fflush(stdout);
    k::qsa_block_scores_seg(seg_d, idx_d, d_qidx, d_steps, m, max_blocks, s, sc[1], nullptr,
                            k::qsa_score_grid_blocks);
    std::puts("  v1 topk");
    std::fflush(stdout);
    k::qsa_block_topk(sc[1], d_steps, m, max_blocks, cap, s, sel[1], nullptr, k::qsa_block_topk_nch(1, max_blocks));
    std::puts("  v1 attn");
    std::fflush(stdout);
    k::qsa_decode_attn_seg(seg_d, idx_d, d_qcur, sel[1], d_steps, cap, s, seg_h[(size_t) row_slot[0]].kv, scratch[1],
                           attn[1], m, nullptr);
    std::puts("  v1 sync");
    std::fflush(stdout);
    ck(cudaDeviceSynchronize());
    bool ok = true;
    const auto s0 = down(sc[0], (size_t) (m * max_blocks)), s1 = down(sc[1], (size_t) (m * max_blocks));
    const auto i0 = down(sel[0], (size_t) (m * cap)), i1 = down(sel[1], (size_t) (m * cap));
    const auto a0 = down(attn[0], (size_t) (m * NH * HD)), a1 = down(attn[1], (size_t) (m * NH * HD));
    for (int t = 0; t < m && ok; ++t) {
        const int64_t nb = steps[(size_t) (t * k::kStepCount + k::kStepNBid)] + 1;
        const int64_t w = steps[(size_t) (t * k::kStepCount + k::kStepWidth)];
        const char* what = nullptr;
        if (std::memcmp(&s0[(size_t) (t * max_blocks)], &s1[(size_t) (t * max_blocks)], (size_t) nb * 4)) what = "scores";
        else if (std::memcmp(&i0[(size_t) (t * cap)], &i1[(size_t) (t * cap)], (size_t) w * 4)) what = "ids";
        else if (std::memcmp(&a0[(size_t) (t * NH * HD)], &a1[(size_t) (t * NH * HD)], (size_t) (NH * HD) * 4))
            what = "attention";
        if (what) {
            std::fprintf(stderr, "FAIL fmt=%d rows=%d row=%d ctx=%lld: %s differ\n", fmt, m, t, (long long) ctx[t], what);
            ok = false;
        }
    }
    double mag = 0;   // the attention output is not trivially zero
    for (float v : a0) mag += std::fabs(v);
    if (ok && !(mag > 0)) { std::fprintf(stderr, "FAIL fmt=%d rows=%d: attention output is zero\n", fmt, m); ok = false; }
    for (int v = 0; v < 2; ++v) { ck(cudaFree(sc[v])); ck(cudaFree(sel[v])); ck(cudaFree(scratch[v])); ck(cudaFree(attn[v])); }
    ck(cudaFree(d_qidx)); ck(cudaFree(d_qcur)); ck(cudaFree(d_steps));
    ck(cudaFree((void*) seg_d)); ck(cudaFree((void*) idx_d));
    for (auto& sl : slots) free_slot(sl);
    return ok;
}

}  // namespace

int main() {
    int devices = 0;
    const cudaError_t de = cudaGetDeviceCount(&devices);
    if (de == cudaErrorNoDevice || (de == cudaSuccess && devices == 0)) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    if (de != cudaSuccess) {   // an enumeration failure must not skip a bitwise gate
        std::fprintf(stderr, "FAIL: cudaGetDeviceCount: %s\n", cudaGetErrorString(de));
        return 1;
    }
    // contexts per row: under and over the selection width (2048 cells), partial and complete indexer blocks
    const int64_t ctx[8] = {1500, 37, 9000, 2049, 32768, 300, 20001, 4096};
    const char* only = std::getenv("STRATA_P4_ONLY");   // debug: "fmt:rows" runs one case
    int cases = 0, failed = 0;
    for (int d = 0; d < devices; ++d) {
        ck(cudaSetDevice(d));
        for (int fmt = 0; fmt < 4; ++fmt)
            for (int m : {1, 2, 4, 8}) {
                if (only) {
                    char f2[16];
                    snprintf(f2, sizeof f2, "%d:%d", fmt, m);
                    if (std::strcmp(only, f2)) continue;
                }
                std::printf("case fmt=%d rows=%d\n", fmt, m);
                std::fflush(stdout);
                if (!check(fmt, m, ctx, 1000u * (uint32_t) fmt + (uint32_t) m)) ++failed;
                ++cases;
            }
    }
    if (failed) return 1;
    if (cases == 0) {   // a STRATA_P4_ONLY filter that matched nothing is not a pass
        std::fprintf(stderr, "FAIL: no cases ran (STRATA_P4_ONLY matched nothing?)\n");
        return 1;
    }
    std::printf("PASS: %d cases on %d device(s) (4 KV formats x 1/2/4/8 rows of different slots): scores, ids and "
                "attention bitwise identical to one call per row\n", cases, devices);
    return 0;
}
