// src/kernels/gdn_seg_parity.cpp - the batch window's remaining state-advancing stages in one launch over rows
// (native_qsa_indexer_append_multi_seg, gdn_conv_l2_seg + gdn_step_norm_seg, native_ple_postops_tokens_seg,
// seg_rows.hpp) against one call per row, as phase 3's verify.cpp made them.  Every row owns its buffers and a
// permuted row -> slot index resolves them through the resident table, so the comparison checks the indirection
// too.  After both variants the state (indexer tail/dead/pooled/block_pos, the GDN state block, the PLE history)
// and the per-row outputs must be byte-identical.  Synthetic data, no model.  With no CUDA device the suite skips
// (77); cases run on every visible device.
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/seg_rows.hpp"
#include "strata/kernels/verify_kernels.hpp"

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

const k::RopeScaling kRope = [] {
    k::RopeScaling r;
    r.freq_base = 5e6;
    return r;
}();

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
float edgy(std::mt19937& r) {
    static std::uniform_int_distribution<int> pick(0, 7);
    static std::normal_distribution<float> nd(0.f, 1.f);
    switch (pick(r)) {
        case 0: return 0.0f;
        case 1: return -0.0f;
        case 2: return 1e-30f;
        case 3: return -1e-30f;
        case 4: return 1e30f;
        case 5: return -1e30f;
        default: return nd(r);
    }
}
std::vector<uint8_t> bytes(const float* d, size_t n) { return down<uint8_t>((const uint8_t*) d, n * 4); }

// ---- 1. the indexer append -----------------------------------------------------------------------------
bool check_indexer(int m, uint32_t seed, cudaStream_t s) {
    const k::QsaShapes sh = k::qsa_real_shapes();
    const int D = 128, max_cells = 4096;
    const size_t pooled_n = (size_t) (max_cells / 4 + 1) * D;
    std::mt19937 r(seed);
    std::vector<float> gamma((size_t) D);
    for (auto& g : gamma) g = 0.5f + std::uniform_real_distribution<float>(0.f, 1.f)(r);
    const float* d_gamma = up(gamma);
    std::vector<float> raw((size_t) m * D);
    for (auto& x : raw) x = edgy(r);
    std::vector<int32_t> pos((size_t) m);
    const int starts[8] = {0, 1, 3, 1022, 7, 2, 4095, 5};
    for (int t = 0; t < m; ++t) pos[(size_t) t] = starts[t];
    const float* d_raw = up(raw);
    const int32_t* d_pos = up(pos);
    std::vector<std::vector<float>> init_tail;
    std::vector<k::SegRow> seg0((size_t) m), seg1((size_t) m);
    std::vector<float*> t0, t1, p0, p1, dd0, dd1;
    std::vector<int32_t*> b0, b1;
    for (int i = 0; i < m; ++i) {
        std::vector<float> tail((size_t) 3 * D), dead((size_t) D), pooled(pooled_n);
        for (auto& x : tail) x = edgy(r);
        for (auto& x : dead) x = edgy(r);
        for (auto& x : pooled) x = edgy(r);
        init_tail.push_back(tail);
        t0.push_back(up(tail)); dd0.push_back(up(dead)); p0.push_back(up(pooled)); b0.push_back(up(std::vector<int32_t>{-7}));
        t1.push_back(up(tail)); dd1.push_back(up(dead)); p1.push_back(up(pooled)); b1.push_back(up(std::vector<int32_t>{-7}));
        seg0[(size_t) i] = k::SegRow{};
        seg1[(size_t) i] = k::SegRow{};
        seg1[(size_t) i].idx_tail = t1[(size_t) i];
        seg1[(size_t) i].idx_dead = dd1[(size_t) i];
        seg1[(size_t) i].idx_pooled = p1[(size_t) i];
        seg1[(size_t) i].idx_block_pos = b1[(size_t) i];
    }
    std::vector<int32_t> row_slot((size_t) m);
    std::iota(row_slot.begin(), row_slot.end(), 0);
    std::shuffle(row_slot.begin(), row_slot.end(), r);
    const int32_t* idx_d = up(row_slot);
    const k::SegRow* seg_d = up(seg1);
    // 0: one call per row; 1: one launch over the rows through the table
    for (int t = 0; t < m; ++t) {
        const int i = row_slot[(size_t) t];
        const k::QsaIndexerBuffers ib{t0[(size_t) i], dd0[(size_t) i], p0[(size_t) i], b0[(size_t) i]};
        k::native_qsa_indexer_append_multi(d_raw + (size_t) t * D, d_pos + t, 1, 1, 0, d_gamma, 1e-6f, ib, sh,
                                           max_cells, kRope, s);
    }
    k::native_qsa_indexer_append_multi_seg(seg_d, idx_d, d_raw, d_pos, 1, m, 0, d_gamma, 1e-6f, sh, max_cells, kRope,
                                           s);
    ck(cudaStreamSynchronize(s));
    bool ok = true;
    for (int i = 0; i < m && ok; ++i) {
        const char* what = nullptr;
        if (bytes(t0[i], 3 * D) != bytes(t1[i], 3 * D)) what = "tail";
        else if (bytes(dd0[i], D) != bytes(dd1[i], D)) what = "dead";
        else if (bytes(p0[i], pooled_n) != bytes(p1[i], pooled_n)) what = "pooled";
        else if (down(b0[i], 1) != down(b1[i], 1)) what = "block_pos";
        if (what) {
            std::fprintf(stderr, "FAIL indexer rows=%d slot=%d: %s differs\n", m, i, what);
            ok = false;
        }
    }
    ck(cudaFree((void*) d_gamma)); ck(cudaFree((void*) d_raw)); ck(cudaFree((void*) d_pos));
    ck(cudaFree((void*) idx_d)); ck(cudaFree((void*) seg_d));
    for (int i = 0; i < m; ++i) {
        ck(cudaFree(t0[i])); ck(cudaFree(dd0[i])); ck(cudaFree(p0[i])); ck(cudaFree(b0[i]));
        ck(cudaFree(t1[i])); ck(cudaFree(dd1[i])); ck(cudaFree(p1[i])); ck(cudaFree(b1[i]));
    }
    return ok;
}

// ---- 2. the GDN conv and recurrence --------------------------------------------------------------------
bool check_gdn(int m, uint32_t seed, cudaStream_t s) {
    constexpr int S = 128, HK = 8, HV = 16;             // h_v % h_k == 0; the kernels' own geometry checks apply
    const int C = 2 * S * HK + S * HV;                  // q | k | v rows, as ssm_conv_channels
    const int QK = 2 * HK, ZV = S * HV;
    const uint64_t conv_off = (uint64_t) S * HV * S;
    const size_t state_n = conv_off + (size_t) C * 3;
    std::mt19937 r(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> qkv((size_t) m * C), wc((size_t) C * 4), gate((size_t) m * HV), beta((size_t) m * HV),
        z((size_t) m * ZV), gamma((size_t) S);
    for (auto& x : qkv) x = nd(r);
    for (auto& x : wc) x = nd(r);
    for (auto& x : gate) x = nd(r);
    for (auto& x : beta) x = nd(r);
    for (auto& x : z) x = nd(r);
    for (auto& x : gamma) x = 0.5f + std::uniform_real_distribution<float>(0.f, 1.f)(r);
    std::vector<std::vector<float>> init_state((size_t) m);
    std::vector<float*> st0, st1;
    // the staged outputs are shared per variant (each row writes its own row of them), the state is per slot
    float *h0 = nullptr, *h1 = nullptr, *y0 = nullptr, *y1 = nullptr;
    for (int i = 0; i < m; ++i) {
        init_state[(size_t) i].resize(state_n);
        for (auto& x : init_state[(size_t) i]) x = nd(r);
        st0.push_back(up(init_state[(size_t) i]));
        st1.push_back(up(init_state[(size_t) i]));
    }
    h0 = up(std::vector<float>((size_t) m * C, 0.f));
    h1 = up(std::vector<float>((size_t) m * C, 0.f));
    y0 = up(std::vector<float>((size_t) m * ZV, 0.f));
    y1 = up(std::vector<float>((size_t) m * ZV, 0.f));
    const float *d_qkv = up(qkv), *d_wc = up(wc), *d_gate = up(gate), *d_beta = up(beta), *d_z = up(z),
                *d_gamma = up(gamma);
    const int32_t* d_one = up(std::vector<int32_t>{1});
    std::vector<k::SegRow> seg_h((size_t) m);
    for (int i = 0; i < m; ++i) seg_h[(size_t) i].gdn_state = st1[(size_t) i];
    std::vector<int32_t> row_slot((size_t) m);
    std::iota(row_slot.begin(), row_slot.end(), 0);
    std::shuffle(row_slot.begin(), row_slot.end(), r);
    const int32_t* idx_d = up(row_slot);
    const k::SegRow* seg_d = up(seg_h);
    // 0: one conv and one step+norm per row (the one-token window's calls); 1: one launch each over the rows
    for (int t = 0; t < m; ++t) {
        const int i = row_slot[(size_t) t];
        k::gdn_conv_l2_multi(st0[(size_t) i] + conv_off, d_qkv + (size_t) t * C, d_wc, h0 + (size_t) t * C,
                             C, QK, 1e-6f, 1, s, 0, true);
        k::gdn_step_norm_multi(st0[(size_t) i], h0 + (size_t) t * C, C, d_gate + (size_t) t * HV,
                               d_beta + (size_t) t * HV, d_z + (size_t) t * ZV, d_gamma, 1e-6f,
                               y0 + (size_t) t * ZV, HK, HV, 1, d_one, s, 0);
    }
    k::gdn_conv_l2_seg(seg_d, idx_d, d_qkv, d_wc, h1, C, QK, 1e-6f, m, conv_off, s);
    k::gdn_step_norm_seg(seg_d, idx_d, h1, C, d_gate, d_beta, d_z, d_gamma, 1e-6f, y1, HK, HV, m, d_one, s);
    ck(cudaStreamSynchronize(s));
    bool ok = true;
    for (int i = 0; i < m && ok; ++i) {
        const char* what = nullptr;
        if (down(st0[i], state_n) != down(st1[i], state_n)) what = "state";
        if (what) {
            std::fprintf(stderr, "FAIL gdn rows=%d slot=%d: %s differs\n", m, i, what);
            ok = false;
        }
    }
    if (ok && bytes(h0, (size_t) m * C) != bytes(h1, (size_t) m * C)) {
        std::fprintf(stderr, "FAIL gdn rows=%d: h differs\n", m);
        ok = false;
    }
    if (ok && bytes(y0, (size_t) m * ZV) != bytes(y1, (size_t) m * ZV)) {
        std::fprintf(stderr, "FAIL gdn rows=%d: y differs\n", m);
        ok = false;
    }
    // not trivially equal: the conv must have advanced a history
    if (ok && down(st1[0], state_n) == init_state[0]) {
        std::fprintf(stderr, "FAIL gdn rows=%d: the state never moved\n", m);
        ok = false;
    }
    for (auto* p : {d_qkv, d_wc, d_gate, d_beta, d_z, d_gamma}) ck(cudaFree((void*) p));
    ck(cudaFree((void*) d_one)); ck(cudaFree((void*) idx_d)); ck(cudaFree((void*) seg_d));
    ck(cudaFree(h0)); ck(cudaFree(h1)); ck(cudaFree(y0)); ck(cudaFree(y1));
    for (int i = 0; i < m; ++i) {
        ck(cudaFree(st0[i])); ck(cudaFree(st1[i]));
    }
    return ok;
}

// ---- 3. the PLE postops ---------------------------------------------------------------------------------
bool check_ple(int m, uint32_t seed, cudaStream_t s) {
    const int N = 2560, D = 10240, H = 4, HIST = 9;
    std::mt19937 r(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> nk((size_t) D), nq((size_t) D), nc((size_t) D);
    for (auto* v : {&nk, &nq, &nc})
        for (auto& x : *v) x = 0.5f + std::uniform_real_distribution<float>(0.f, 1.f)(r);
    std::vector<uint16_t> taps((size_t) 4 * D);
    for (auto& t : taps) {
        const uint16_t sign = (uint16_t) ((r() & 1u) << 15), e = (uint16_t) (r() % 15), mm = (uint16_t) (r() & 0x3FFu);
        t = (uint16_t) (sign | (e << 10) | mm);
    }
    const float *d_nk = up(nk), *d_nq = up(nq), *d_nc = up(nc);
    const uint16_t* d_taps = up(taps);
    k::PleWeights w;
    w.norm_key = (float*) d_nk; w.norm_query = (float*) d_nq; w.norm_conv = (float*) d_nc; w.conv1d_f16 = (uint16_t*) d_taps;
    std::vector<float> key((size_t) m * D), hidden((size_t) m * D), value((size_t) m * N);
    for (auto& x : key) x = edgy(r);
    for (auto& x : hidden) x = edgy(r);
    for (auto& x : value) x = nd(r);
    const float *d_key = up(key), *d_hidden = up(hidden), *d_value = up(value);
    std::vector<std::vector<float>> init_hist((size_t) m);
    std::vector<float*> h0, h1;
    // the staged buffers are shared per variant (each row writes its own row), the history is per slot
    float *res0 = nullptr, *res1 = nullptr, *snap0 = nullptr, *snap1 = nullptr;
    float *nrm = nullptr, *gtd = nullptr, *gte = nullptr, *nkey = nullptr;
    for (int i = 0; i < m; ++i) {
        init_hist[(size_t) i].resize((size_t) HIST * D);
        for (auto& x : init_hist[(size_t) i]) x = nd(r);
        h0.push_back(up(init_hist[(size_t) i]));
        h1.push_back(up(init_hist[(size_t) i]));
    }
    res0 = up(std::vector<float>((size_t) m * D, 0.f));
    res1 = up(std::vector<float>((size_t) m * D, 0.f));
    snap0 = up(std::vector<float>((size_t) m * HIST * D, 0.f));
    snap1 = up(std::vector<float>((size_t) m * HIST * D, 0.f));
    nrm = up(std::vector<float>((size_t) m * D, 0.f));
    gtd = up(std::vector<float>((size_t) m * D, 0.f));
    gte = up(std::vector<float>((size_t) m * H, 0.f));
    nkey = up(std::vector<float>((size_t) m * D, 0.f));
    std::vector<k::SegRow> seg_h((size_t) m);
    for (int i = 0; i < m; ++i) seg_h[(size_t) i].ple_hist = h1[(size_t) i];
    std::vector<int32_t> row_slot((size_t) m);
    std::iota(row_slot.begin(), row_slot.end(), 0);
    std::shuffle(row_slot.begin(), row_slot.end(), r);
    const int32_t* idx_d = up(row_slot);
    const k::SegRow* seg_d = up(seg_h);
    // 0: one call per row; 1: one launch chain over the rows
    for (int t = 0; t < m; ++t) {
        const int i = row_slot[(size_t) t];
        const k::NativePleTokensBuffers pb{nkey + (size_t) t * D, nrm + (size_t) t * D, gte + (size_t) t * H,
                                           gtd + (size_t) t * D, nrm + (size_t) t * D, res0 + (size_t) t * D};
        k::native_ple_postops_tokens(d_key + (size_t) t * D, d_hidden + (size_t) t * D, d_value + (size_t) t * N,
                                     h0[(size_t) i], w, pb, 1, s, snap0 + (size_t) t * HIST * D);
    }
    const k::NativePleTokensBuffers pb{nkey, nrm, gte, gtd, nrm, res1};
    k::native_ple_postops_tokens_seg(d_key, d_hidden, d_value, seg_d, idx_d, w, pb, m, s, snap1);
    ck(cudaStreamSynchronize(s));
    bool ok = true;
    for (int i = 0; i < m && ok; ++i) {
        if (down(h0[i], (size_t) HIST * D) != down(h1[i], (size_t) HIST * D)) {
            std::fprintf(stderr, "FAIL ple rows=%d slot=%d: history differs\n", m, i);
            ok = false;
        }
    }
    if (ok && bytes(res0, (size_t) m * D) != bytes(res1, (size_t) m * D)) {
        std::fprintf(stderr, "FAIL ple rows=%d: result differs\n", m);
        ok = false;
    }
    if (ok && bytes(snap0, (size_t) m * HIST * D) != bytes(snap1, (size_t) m * HIST * D)) {
        std::fprintf(stderr, "FAIL ple rows=%d: snap differs\n", m);
        ok = false;
    }
    if (ok && down(h1[0], (size_t) HIST * D) == init_hist[0]) {
        std::fprintf(stderr, "FAIL ple rows=%d: the history never moved\n", m);
        ok = false;
    }
    for (auto* p : {d_nk, d_nq, d_nc}) ck(cudaFree((void*) p));
    ck(cudaFree((void*) d_taps));
    for (auto* p : {d_key, d_hidden, d_value}) ck(cudaFree((void*) p));
    ck(cudaFree((void*) idx_d)); ck(cudaFree((void*) seg_d));
    ck(cudaFree(res0)); ck(cudaFree(res1)); ck(cudaFree(snap0)); ck(cudaFree(snap1));
    ck(cudaFree(nrm)); ck(cudaFree(gtd)); ck(cudaFree(gte)); ck(cudaFree(nkey));
    for (int i = 0; i < m; ++i) {
        ck(cudaFree(h0[i])); ck(cudaFree(h1[i]));
    }
    return ok;
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    const char* only = std::getenv("STRATA_P4_ONLY");   // debug: "stage:rows" (0 indexer, 1 gdn, 2 ple)
    cudaStream_t s = cudaStreamPerThread;
    int cases = 0, failed = 0;
    for (int d = 0; d < devices; ++d) {
        ck(cudaSetDevice(d));
        for (int stage = 0; stage < 3; ++stage)
            for (int m : {1, 2, 4, 8}) {
                if (only) {
                    char f2[16];
                    snprintf(f2, sizeof f2, "%d:%d", stage, m);
                    if (std::strcmp(only, f2)) continue;
                }
                std::printf("case stage=%d rows=%d\n", stage, m);
                std::fflush(stdout);
                const bool ok = stage == 0 ? check_indexer(m, 3000u + (uint32_t) m, s)
                              : stage == 1 ? check_gdn(m, 4000u + (uint32_t) m, s)
                                           : check_ple(m, 5000u + (uint32_t) m, s);
                if (!ok) ++failed;
                ++cases;
            }
    }
    if (failed) return 1;
    std::printf("PASS: %d cases on %d device(s) (indexer, GDN and PLE stages x 1/2/4/8 rows of different slots): "
                "state and outputs byte-identical to one call per row\n", cases, devices);
    return 0;
}
