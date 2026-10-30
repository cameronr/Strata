// src/core/session.cpp - the sequence state: its layout, its zeroing and the prompt cache's checkpoints of it.
#include "strata/core/session.hpp"

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/ngram.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::core {
namespace {

constexpr uint64_t SESSION_STATE_ALIGN = 256;

uint64_t align_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }

}  // namespace

/// `gdn_buffers_bytes` carves a GDN layer's state for ONE layer and `GdnBuffers::state`/`conv_state` point INTO that
/// carve, so a session with 36 GDN layers has to give each one its own - they cannot share, because the recurrence
/// is the whole point.
uint64_t gdn_state_floats(const ModelGeometry& g) {
    return (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
           (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
}

/// `NG_HIST` rows of `hc_dim` floats: the PLE conv's history, which is the ONLY PLE state that lives in the
/// session arena.  The table and the weights are model-level and the caller owns them.
static uint64_t ple_hist_bytes() {
    return (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM * sizeof(float);
}

uint64_t session_bytes(const ModelGeometry& g, int64_t max_cells) {
    uint64_t n = 0;
    n += (uint64_t) g.n_gdn_layers() * gdn_state_floats(g) * 4;
    // One QSA state carries the RoPE table; the others borrow it (P7: 64 MiB per layer at 262K).
    if (g.n_qsa_layers() > 0)
        n += qsa_state_bytes(g, max_cells, true) + (uint64_t) (g.n_qsa_layers() - 1) * qsa_state_bytes(g, max_cells, false);
    n += align_up((uint64_t) g.hc * g.n_embd * 4, SESSION_STATE_ALIGN);   // R
    n += ple_hist_bytes();                       // the PLE's NG_HIST normalized history rows
    return align_up(n, SESSION_STATE_ALIGN) + 4096;
}

uint64_t session_init(const ModelGeometry& g, int64_t max_cells, int64_t k, void* base, SessionState& s) {
    uint8_t* p = (uint8_t*) base;
    uint64_t used = 0;
    auto take = [&](uint64_t bytes) {
        uint8_t* r = p + used;
        used = align_up(used + bytes, SESSION_STATE_ALIGN);
        return r;
    };

    s.max_cells = max_cells;
    s.k = k;

    s.gdn_state = (float*) take((uint64_t) g.n_gdn_layers() * gdn_state_floats(g) * 4);

    // the 12 QSA states are separate allocations carved from one arena, because `QsaState` is a struct of
    // pointers and `qsa_state_init` writes them - a contiguous array would need the arena to be laid out the
    // same way, which is a coupling with nothing to gain.
    const uint64_t first = qsa_state_bytes(g, max_cells, true), rest = qsa_state_bytes(g, max_cells, false);
    s.qsa_state_arena = take(g.n_qsa_layers() > 0 ? first + (uint64_t) (g.n_qsa_layers() - 1) * rest : 0);
    s.qsa_states = new QsaState[(size_t) g.n_qsa_layers()];

    uint8_t* qp = (uint8_t*) s.qsa_state_arena;
    // a layer whose pinned RAM could not be had (KV streaming's host copy) is half-built: going on would have the
    // attention read null host pointers at the first request ("illegal memory access"), so the session fails here
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i)
        if (qsa_state_init(g, max_cells, qp + (i == 0 ? 0 : first + (uint64_t) (i - 1) * rest), s.qsa_states[i],
                           i == 0 ? nullptr : &s.qsa_states[0]) == 0)
            return 0;

    s.R = (float*) take((uint64_t) g.hc * g.n_embd * 4);
    // THE PLE HISTORY: NG_HIST rows of hc_dim floats, row-fastest.  Sequence state, carved here and zeroed by
    // `session_zero`; it survives every token, which is the whole point of a conv history.
    s.ple_hist = (float*) take(ple_hist_bytes());
    return used;
}

void session_free(SessionState& s, const ModelGeometry& g) {
    if (s.qsa_states == nullptr) return;
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i) qsa_state_free(s.qsa_states[i]);
    delete[] s.qsa_states;
    s.qsa_states = nullptr;
}

void session_zero(SessionState& s, const ModelGeometry& g, const float* R_init, void* stream) {
    cudaStream_t cs = (cudaStream_t) stream;
    // the residual: `hc` copies of the one vector a caller hands in.  A real sequence's first token is the
    // embedding broadcast to every stream, which is the reference's own initial condition.
    if (R_init != nullptr) {
        for (int64_t c = 0; c < g.hc; ++c)
            cudaMemcpyAsync(s.R + (size_t) c * g.n_embd, R_init, (size_t) g.n_embd * 4,
                            cudaMemcpyDeviceToDevice, cs);
    } else {
        cudaMemsetAsync(s.R, 0, (size_t) g.hc * g.n_embd * 4, cs);
    }
    // every GDN layer's recurrence and conv history
    cudaMemsetAsync(s.gdn_state, 0, (size_t) g.n_gdn_layers() * gdn_state_floats(g) * 4, cs);
    // and every QSA layer's cache and indexer
    for (int64_t i = 0; i < g.n_qsa_layers(); ++i) qsa_state_zero(s.qsa_states[i], g, stream);
    // **AND THE PLE'S CONV HISTORY AND TOKEN WINDOW.**  A sequence that started with a warm history would
    // convolve over rows belonging to a different sequence - the conv reads NG_HIST previous NORMALIZED rows,
    // so a stale one is a real contribution and not a zero.  The token window resets to `NG_HIST`-many nulls
    // for the same reason: `ngram_rows` treats a missing predecessor as the EOS cut, which is what a sequence
    // boundary IS.
    cudaMemsetAsync(s.ple_hist, 0, (size_t) ple_hist_bytes(), cs);
    s.ple_prev[0] = -1;
    s.ple_prev[1] = -1;
    s.ple_token = -1;
}

namespace {

/// One region of a checkpoint: `bytes` of device memory at `dev`, at `host` in the host buffer; `layer`: the model
/// layer whose state it is (the PLE's: layer 1).
struct CkptPart { void* dev; uint64_t bytes; uint64_t host; int64_t layer; };

/// The parts of a checkpoint at `pos` of the layers from `first` on, from a state whose GDN and QSA states are indexed
/// from `first` (a split prompt path's second stage, prefill/split.hpp; 0: a whole session).  Storage order: every
/// GDN layer's state, the PLE history, each QSA layer's indexer tail, spare key, open block's pooled row and block
/// position; the PLE token window follows them on the host.
std::vector<CkptPart> ckpt_parts(const SessionState& s, const ModelGeometry& g, int64_t pos, int64_t first = 0) {
    strata::kernels::QsaShapes sh = strata::kernels::qsa_real_shapes();
    sh.idx_dim = g.idx_key_dim;
    const uint64_t row = (uint64_t) sh.idx_dim * sizeof(float);
    const uint64_t gdn = (uint64_t) gdn_state_floats(g) * sizeof(float);
    std::vector<CkptPart> parts;
    uint64_t at = 0;
    for (int64_t l = 0, i = 0; l < g.n_layers; ++l) {
        if (is_qsa_layer(g, l)) continue;
        if (l >= first) parts.push_back({(uint8_t*) s.gdn_state + (uint64_t) i++ * gdn, gdn, at, l});
        at += gdn;
    }
    if (first <= 1) parts.push_back({s.ple_hist, ple_hist_bytes(), at, 1});
    at += ple_hist_bytes();
    for (int64_t l = 0, i = 0; l < g.n_layers; ++l) {
        if (!is_qsa_layer(g, l)) continue;
        const QsaState* st = l >= first ? &s.qsa_states[i++] : nullptr;
        const CkptPart q[4] = {
            {st ? st->idx_tail : nullptr, (uint64_t) (sh.idx_block - 1) * row, 0, l},
            {st ? st->idx_dead : nullptr, row, 0, l},
            {st ? (uint8_t*) st->idx_pooled + (uint64_t) (pos / sh.idx_block) * row : nullptr, row, 0, l},
            {st ? st->idx_block_pos : nullptr, sizeof(int32_t), 0, l}};
        for (CkptPart p : q) {
            p.host = at;
            at += p.bytes;
            if (st) parts.push_back(p);
        }
    }
    return parts;
}

constexpr uint64_t kCkptHostInts = 3;   // ple_prev[0], ple_prev[1], ple_token

/// Where the token window starts in the host buffer.
uint64_t ckpt_ints_at(const ModelGeometry& g) {
    const uint64_t row = (uint64_t) g.idx_key_dim * sizeof(float);
    const strata::kernels::QsaShapes sh = strata::kernels::qsa_real_shapes();
    return (uint64_t) g.n_gdn_layers() * gdn_state_floats(g) * sizeof(float) + ple_hist_bytes() +
           (uint64_t) g.n_qsa_layers() * ((uint64_t) (sh.idx_block + 1) * row + sizeof(int32_t));
}

}  // namespace

uint64_t session_ckpt_bytes(const ModelGeometry& g) { return ckpt_ints_at(g) + kCkptHostInts * sizeof(int32_t); }

bool session_ckpt_save(const SessionState& s, const ModelGeometry& g, int64_t pos, void* host, void* stream) {
    return session_ckpt_save_layers(s, g, pos, host, stream, 0, g.n_layers, 0);
}

bool session_ckpt_save_layers(const SessionState& s, const ModelGeometry& g, int64_t pos, void* host, void* stream,
                              int64_t lo, int64_t hi, int64_t first) {
    cudaStream_t cs = (cudaStream_t) stream;
    uint8_t* h = (uint8_t*) host;
    for (const CkptPart& p : ckpt_parts(s, g, pos, first))
        if (p.layer >= lo && p.layer < hi &&
            cudaMemcpyAsync(h + p.host, p.dev, (size_t) p.bytes, cudaMemcpyDeviceToHost, cs) != cudaSuccess)
            return session_copy_failed(stream);
    if (lo == 0) {
        const int32_t ints[kCkptHostInts] = {s.ple_prev[0], s.ple_prev[1], s.ple_token};
        std::memcpy(h + ckpt_ints_at(g), ints, sizeof ints);
    }
    return cudaStreamSynchronize(cs) == cudaSuccess || session_copy_failed(stream);
}

bool session_ckpt_load(SessionState& s, const ModelGeometry& g, int64_t pos, const void* host, void* stream) {
    cudaStream_t cs = (cudaStream_t) stream;
    const uint8_t* h = (const uint8_t*) host;
    for (const CkptPart& p : ckpt_parts(s, g, pos))
        if (cudaMemcpyAsync(p.dev, h + p.host, (size_t) p.bytes, cudaMemcpyHostToDevice, cs) != cudaSuccess)
            return session_copy_failed(stream);
    h += ckpt_ints_at(g);
    int32_t ints[kCkptHostInts];
    std::memcpy(ints, h, sizeof ints);
    s.ple_prev[0] = ints[0];
    s.ple_prev[1] = ints[1];
    s.ple_token = ints[2];
    return cudaStreamSynchronize(cs) == cudaSuccess || session_copy_failed(stream);
}

bool session_copy_failed(void* stream) {
    (void) cudaGetLastError();
    (void) cudaStreamSynchronize((cudaStream_t) stream);
    (void) cudaGetLastError();
    return false;
}

}  // namespace strata::core
