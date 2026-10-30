// include/strata/core/session.hpp - the sequence state: every layer's GDN and QSA state, the block scratch and
// the PLE's state, and the prompt cache's checkpoints of it.
#pragma once

#include "strata/core/layer.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// Everything a sequence needs that is not a weight: the per-layer state and the block scratch.
struct SessionState {
    int64_t max_cells = 0;

    float* gdn_state = nullptr;     ///< (n_gdn_layers, gdn_state_floats)

    QsaState* qsa_states = nullptr;      ///< one per QSA layer
    void* qsa_state_arena = nullptr;


    float* R = nullptr;             ///< the residual streams, (hc, n_embd)
    int64_t k = 10;                 ///< experts per token

    // ================================ THE PLE, AND IT IS SEQUENCE STATE ================================
    //
    // **THE PLE IS A LAYER-1 MODULE AND THIS ENGINE DID NOT RUN IT, WHICH IS WHY C1 FAILED** (LEDGER L123).
    // Every part of it was already built and parity-tested - the hash against `ref/ngram.py` with the
    // artifact's own constants, the IQ4_NL row read against numpy over the original GGUF, and the block
    // against ggml's own graph - and none of it was ever called from the model path.  These fields are what
    // was missing on the state side.
    //
    // `ple_prev` is the last two tokens OLDEST FIRST with `-1` where there is none, which is exactly
    // `ngram_rows`' `prev` argument; `ple_token` is the token being decoded.  **They are advanced by the
    // DRIVER, once per token**, because the hash is a function of the sequence and not of a layer.
    int32_t ple_prev[2] = {-1, -1};
    int32_t ple_token = -1;
    /// `NG_HIST` rows of `hc_dim` NORMALIZED history, **ROW-FASTEST**: `ple_hist[row + NG_HIST*channel]`.
    /// That is `ggml_reshape_3d(state, d_conv-1, conv_channels, n_seqs)`, not a layout choice - the transposed
    /// reading fails ggml's own assert (`ple_layer_xcheck.cpp` L79-83).
    float* ple_hist = nullptr;
    /// The PLE run: its history and token window are sequence state.  One whose `ready()` is false is skipped.
    PleRun ple;
};

/// The floats of one GDN layer's state, its recurrence and conv history (`gdn_state` holds one per GDN layer).
uint64_t gdn_state_floats(const ModelGeometry& g);
/// Bytes for a whole session at `max_cells` of context: every layer's state at once, so nothing is allocated
/// while a sequence runs.
uint64_t session_bytes(const ModelGeometry& g, int64_t max_cells);
/// Carves `base` (DEVICE memory) into `s`.  Returns the bytes used.
uint64_t session_init(const ModelGeometry& g, int64_t max_cells, int64_t k, void* base, SessionState& s);
/// Frees what session_init took outside `base`: the QSA states and their pinned staging (`base` stays the caller's).
void session_free(SessionState& s, const ModelGeometry& g);
/// Zeroes every layer's state - the residual to `R_init`, everything else to zero, so a fresh sequence starts
/// from the reference's own `zeros()`.
void session_zero(SessionState& s, const ModelGeometry& g, const float* R_init, void* stream);

// ================================ prompt-cache checkpoints ================================
//
// The K/V caches are indexed by position, so the cells before a position stay valid when a later sequence
// overwrites the ones after it.  What a position does NOT recover is the state that folds the whole prefix
// into a fixed size: every GDN layer's recurrence and conv history, each QSA indexer's raw block tail (plus
// its spare key, the pooled row of the open block and the block position), and the PLE's conv history and
// token window.  A checkpoint is exactly that state at one position; restoring it and overwriting the cells
// from that position on continues the sequence as if it had been processed up to there.

/// Bytes of one checkpoint (host memory).
uint64_t session_ckpt_bytes(const ModelGeometry& g);
/// Copies the sequence state at `pos` (positions [0, pos) processed) into `host`; synchronous.
bool session_ckpt_save(const SessionState& s, const ModelGeometry& g, int64_t pos, void* host, void* stream);
/// The same for the layers [lo, hi) only (the PLE's with layer 1, the token window with lo == 0), from a state whose
/// GDN and QSA states are indexed from layer `first` (a split prompt path's second stage: prefill/split.hpp; the
/// other stage saves the rest of the same buffer).
bool session_ckpt_save_layers(const SessionState& s, const ModelGeometry& g, int64_t pos, void* host, void* stream,
                              int64_t lo, int64_t hi, int64_t first);
/// The inverse: the session continues at `pos`.  The K/V cells [0, pos) must still hold the same prefix.
bool session_ckpt_load(SessionState& s, const ModelGeometry& g, int64_t pos, const void* host, void* stream);
/// After a failed copy of a checkpoint or of K/V cells: waits for the copies queued before it and clears the
/// runtime's error, which the next launch check would otherwise take for its own and end the engine (the caller
/// falls back: no checkpoint, no stash, or the prompt read again).  Returns false.
bool session_copy_failed(void* stream);

/// The VRAM tier's residency table on the device (`n_layers x n_expert`, a slot or -1) and its slot arena: the
/// verify windows decide their GPU hits by it.
struct TokenHits {
    const int32_t* d_res = nullptr;      ///< device [n_layers * n_expert]
    int64_t n_expert = 0;
    const uint8_t* cache_base = nullptr; ///< slot 0 of the VRAM expert arena
    int64_t blob = 0;                    ///< bytes per slot
};


}  // namespace strata::core
