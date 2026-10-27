// src/kernels/qsa_kv_seg_parity.cpp - the batch window's KV append in one launch over rows (kv_append_q8_seg,
// kv_append_q4_seg, seg_rows.hpp) against one call per row, as phase 3's verify.cpp made them. Every slot has its
// own pools and page table and appends at its own position; after both variants the pools must be byte-identical.
// F16 pools are not covered: init_slots refuses them for batch slots (the F16 append stays a host loop). The
// variants use the same page-table permutation and the same row -> slot index, so the comparison also checks the
// table indirection. Synthetic data, no model. With no CUDA device the suite skips (77); cases run on every
// visible device.
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/seg_rows.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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

constexpr int64_t kMaxCells = 8192;   // positions live under this; pools sized to it

struct Slot {
    void* bufs[8] = {};
    k::SegWritePools wr;
};

// one slot's writable pools for format `fmt` (the per-format nulls mirror QsaState), zero-filled
Slot make_slot(int fmt, const k::QsaShapes& s, const int32_t* page_table) {
    Slot sl;
    sl.wr.page_table = const_cast<int32_t*>(page_table);
    const size_t rows = (size_t) kMaxCells * s.n_head_kv, hd = (size_t) s.head_dim;
    auto zeros = [&](size_t bytes) {
        void* d = nullptr;
        ck(cudaMalloc(&d, bytes));
        ck(cudaMemset(d, 0, bytes));
        return d;
    };
    int i = 0;
    if (fmt == 0) {   // Int8
        sl.wr.k_q = (int8_t*) (sl.bufs[i++] = zeros(rows * hd));
        sl.wr.v_q = (int8_t*) (sl.bufs[i++] = zeros(rows * hd));
        sl.wr.k_scale = (uint16_t*) (sl.bufs[i++] = zeros(rows * hd / k::KV_Q8_GROUP * 2));
        sl.wr.v_scale = (uint16_t*) (sl.bufs[i++] = zeros(rows * hd / k::KV_Q8_GROUP * 2));
    } else if (fmt == 1) {   // Q4
        sl.wr.k_q4 = (uint8_t*) (sl.bufs[i++] = zeros(rows * hd / 2));
        sl.wr.k_q4s = (uint16_t*) (sl.bufs[i++] = zeros(rows * hd / k::KV_Q4_GROUP * 2));
        sl.wr.v_q4 = (uint8_t*) (sl.bufs[i++] = zeros(rows * hd / 2));
        sl.wr.v_q4s = (uint16_t*) (sl.bufs[i++] = zeros(rows * hd / k::KV_Q4_GROUP * 2));
    } else {   // K8V4
        sl.wr.k_q = (int8_t*) (sl.bufs[i++] = zeros(rows * hd));
        sl.wr.k_scale = (uint16_t*) (sl.bufs[i++] = zeros(rows * hd / k::KV_Q8_GROUP * 2));
        sl.wr.v_q4 = (uint8_t*) (sl.bufs[i++] = zeros(rows * hd / 2));
        sl.wr.v_q4s = (uint16_t*) (sl.bufs[i++] = zeros(rows * hd / k::KV_Q4_GROUP * 2));
    }
    return sl;
}

void free_slot(Slot& sl) {
    for (void* b : sl.bufs) if (b) ck(cudaFree(b));
}

// the bytes of every non-null pool array, concatenated (the comparison payload)
std::vector<uint8_t> dump(const Slot& sl, const k::QsaShapes& s) {
    const size_t rows = (size_t) kMaxCells * s.n_head_kv, hd = (size_t) s.head_dim;
    std::vector<uint8_t> out;
    auto cat = [&](const void* p, size_t bytes) {
        if (!p) return;
        const auto part = down<uint8_t>((const uint8_t*) p, bytes);
        out.insert(out.end(), part.begin(), part.end());
    };
    cat(sl.wr.k_q, rows * hd);
    cat(sl.wr.v_q, rows * hd);
    cat(sl.wr.k_scale, rows * hd / k::KV_Q8_GROUP * 2);
    cat(sl.wr.v_scale, rows * hd / k::KV_Q8_GROUP * 2);
    cat(sl.wr.k_q4, rows * hd / 2);
    cat(sl.wr.k_q4s, rows * hd / k::KV_Q4_GROUP * 2);
    cat(sl.wr.v_q4, rows * hd / 2);
    cat(sl.wr.v_q4s, rows * hd / k::KV_Q4_GROUP * 2);
    return out;
}

// one case: `m` rows of different slots, KV format `fmt`, append positions `pos[i]`
bool check(int fmt, int m, const int64_t* pos, uint32_t seed) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t NKV = s.n_head_kv, HD = s.head_dim;
    std::mt19937 r(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int64_t pages = kMaxCells / s.page_size;
    std::vector<int32_t> pt((size_t) pages);
    std::iota(pt.begin(), pt.end(), 0);
    std::shuffle(pt.begin(), pt.end(), r);
    const int32_t* pt_d = up(pt);
    std::vector<Slot> a, b;   // variant 0's and variant 1's slots, identical start states
    for (int i = 0; i < m; ++i) a.push_back(make_slot(fmt, s, pt_d));
    for (int i = 0; i < m; ++i) b.push_back(make_slot(fmt, s, pt_d));
    std::vector<float> kcur((size_t) m * NKV * HD), vcur((size_t) m * NKV * HD);
    for (auto& v : kcur) v = nd(r);
    for (auto& v : vcur) v = nd(r);
    std::vector<int32_t> steps((size_t) m * k::kStepCount);
    for (int i = 0; i < m; ++i) k::qsa_step_fill(steps.data() + i * k::kStepCount, pos[i], s);
    const float* kc_d = up(kcur);
    const float* vc_d = up(vcur);
    const int32_t* st_d = up(steps);
    // the resident table (variant 1's rows resolve through it) and the permuted row -> slot index
    std::vector<int32_t> row_slot((size_t) m);
    std::iota(row_slot.begin(), row_slot.end(), 0);
    std::shuffle(row_slot.begin(), row_slot.end(), r);
    std::vector<k::SegRow> seg_h((size_t) m);
    for (int t = 0; t < m; ++t) seg_h[(size_t) t].wr = b[(size_t) t].wr;
    const k::SegRow* seg_d = up(seg_h);
    const int32_t* idx_d = up(row_slot);
    // 0: one call per row (phase 3's batch-window loop), row t against the slot the index names for it
    for (int t = 0; t < m; ++t) {
        const k::SegWritePools& w = a[(size_t) row_slot[t]].wr;
        const int32_t* step = st_d + (size_t) t * k::kStepCount;
        const float* krow = kc_d + (size_t) t * NKV * HD;
        const float* vrow = vc_d + (size_t) t * NKV * HD;
        if (fmt == 0)
            k::kv_append_q8_steps(w.k_q, w.v_q, w.k_scale, w.v_scale, w.page_table, step, k::kStepCount, krow, vrow,
                                  1, s, nullptr, 3);
        else if (fmt == 1)
            k::kv_append_q4_steps(w.k_q4, w.k_q4s, w.v_q4, w.v_q4s, w.page_table, step, k::kStepCount, krow, vrow, 1,
                                  s, nullptr);
        else {
            k::kv_append_q8_steps(w.k_q, nullptr, w.k_scale, nullptr, w.page_table, step, k::kStepCount, krow, vrow,
                                  1, s, nullptr, 1);
            k::kv_append_q4_steps(nullptr, nullptr, w.v_q4, w.v_q4s, w.page_table, step, k::kStepCount, krow, vrow,
                                  1, s, nullptr);
        }
    }
    // 1: one launch per side over the rows, state through the table (the qsa_kv_append_steps_seg dispatch)
    if (fmt == 0) {
        k::kv_append_q8_seg(seg_d, idx_d, st_d, k::kStepCount, kc_d, vc_d, m, s, nullptr, 3);
    } else if (fmt == 1) {
        k::kv_append_q4_seg(seg_d, idx_d, st_d, k::kStepCount, kc_d, vc_d, m, s, nullptr, 2);
    } else {
        k::kv_append_q8_seg(seg_d, idx_d, st_d, k::kStepCount, kc_d, vc_d, m, s, nullptr, 1);
        k::kv_append_q4_seg(seg_d, idx_d, st_d, k::kStepCount, kc_d, vc_d, m, s, nullptr, 1);
    }
    ck(cudaDeviceSynchronize());
    bool ok = true;
    for (int i = 0; i < m && ok; ++i) {
        const auto da = dump(a[(size_t) i], s), db = dump(b[(size_t) i], s);
        if (da != db) {
            std::fprintf(stderr, "FAIL fmt=%d rows=%d slot=%d: appended pools differ\n", fmt, m, i);
            ok = false;
        }
    }
    // not trivially equal: the append must have written something
    if (ok) {
        const auto d0 = dump(a[0], s);
        bool any = false;
        for (uint8_t v : d0) any |= v != 0;
        if (!any) { std::fprintf(stderr, "FAIL fmt=%d rows=%d: nothing appended\n", fmt, m); ok = false; }
    }
    ck(cudaFree((void*) st_d)); ck(cudaFree((void*) kc_d)); ck(cudaFree((void*) vc_d));
    ck(cudaFree((void*) seg_d)); ck(cudaFree((void*) idx_d)); ck(cudaFree((void*) pt_d));
    for (auto& sl : a) free_slot(sl);
    for (auto& sl : b) free_slot(sl);
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
    // append positions per row: page boundaries, first and last page, mid-context
    const int64_t pos[8] = {1500, 37, 64, 2049, 8191, 63, 4000, 4096};
    const char* only = std::getenv("STRATA_P4_ONLY");   // debug: "fmt:rows" runs one case
    int cases = 0, failed = 0;
    for (int d = 0; d < devices; ++d) {
        ck(cudaSetDevice(d));
        for (int fmt = 0; fmt < 3; ++fmt)
            for (int m : {1, 2, 4, 8}) {
                if (only) {
                    char f2[16];
                    snprintf(f2, sizeof f2, "%d:%d", fmt, m);
                    if (std::strcmp(only, f2)) continue;
                }
                std::printf("case fmt=%d rows=%d\n", fmt, m);
                std::fflush(stdout);
                if (!check(fmt, m, pos, 2000u * (uint32_t) fmt + (uint32_t) m)) ++failed;
                ++cases;
            }
    }
    if (failed) return 1;
    if (cases == 0) {   // a STRATA_P4_ONLY filter that matched nothing is not a pass
        std::fprintf(stderr, "FAIL: no cases ran (STRATA_P4_ONLY matched nothing?)\n");
        return 1;
    }
    std::printf("PASS: %d cases on %d device(s) (3 KV formats x 1/2/4/8 rows of different slots): appended pools "
                "byte-identical to one call per row\n", cases, devices);
    return 0;
}
