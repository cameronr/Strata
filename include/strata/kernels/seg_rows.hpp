// include/strata/kernels/seg_rows.hpp - the batch window's segment table (phase 4).
//
// A batch window decodes S independent sequences in one window: row r belongs to slot `row_slot[r]`, and every
// state-touching kernel resolves that row's sequence state through a `SegRow` entry.  THE CONTRACT:
//
//   * Resident tables hold the per-sequence pointers, one entry per (stateful layer, slot), built once by the
//     verifier's `init_slots` from the slot sessions and written nowhere else.  They are read-only during graph
//     capture and replay; a slot re-registration rebuilds them host-side, serialized against any in-flight window.
//   * A window contributes only its row -> slot index, staged at a fixed device address exactly like the window's
//     tokens and step records (graph inputs live at fixed addresses).  A kernel resolves row r's state as
//     `tab[row_slot[r]]`; nothing copies the tables, and no kernel keeps another representation of the mapping.
//   * Because the tables and the index carry every per-slot pointer, no batch graph bakes slot identity: batch
//     graphs key on the window size alone.
//
// Bitwise equality of a fused (one-launch-over-rows) kernel with the per-row path is by PROGRAM IDENTITY: the same
// `__device__` body the per-row kernel runs, the same grid.x, the same derived scalars, with only pointers and row
// offsets varying per row.  Where a fused call cannot be program-identical it declines and the per-row launches of
// the same kernel run instead.  Cross-kernel parity (one kernel form vs another) is relied on nowhere.
//
// `ple_hist` is live only in the PLE table, `gdn_state` only in GDN tables (the entry's base already includes the
// layer's stride; the conv history follows at + conv_off), the indexer fields and `kv` only in QSA tables.
// Sizes the kernels use (max_cells, cap, the KV format, the rope mode) are uniform across slots by `init_slots`
// and the global format, so a row entry carries pure pointers.
#pragma once

#include "strata/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace strata::kernels {

/// The QSA pools a table entry can WRITE (KV append stores codes and scales): the same arrays as `kv`, the same
/// per-format nulls, with the const off.  `QsaAttnPools` stays the read-only view the attention launchers take.
struct SegWritePools {
    int8_t* k_q = nullptr;
    uint16_t* k_scale = nullptr;
    int8_t* v_q = nullptr;
    uint16_t* v_scale = nullptr;
    uint8_t* k_q4 = nullptr;
    uint16_t* k_q4s = nullptr;
    uint8_t* v_q4 = nullptr;
    uint16_t* v_q4s = nullptr;
    int32_t* page_table = nullptr;
};

/// One sequence's state-touching pointers, as one table entry.  Pure data: every size a kernel needs is uniform
/// across the window's rows.
struct SegRow {
    float* ple_hist = nullptr;       ///< PLE: the NG_HIST x hc_dim normalized history, advanced in place
    float* gdn_state = nullptr;      ///< GDN: this layer's state block (recurrence at offset 0, conv history after)
    float* idx_tail = nullptr;       ///< indexer: the QsaIndexerBuffers fields
    float* idx_dead = nullptr;
    float* idx_pooled = nullptr;
    int32_t* idx_block_pos = nullptr;
    QsaAttnPools kv{};               ///< QSA: the K/V pools and page table for attention and KV append
    SegWritePools wr{};              ///< QSA: the same pools writable, for the KV append
};

}  // namespace strata::kernels
