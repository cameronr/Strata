// include/strata/kernels/kv_stream.hpp - the host copy of a QSA layer's K/V that the append kernels can write
// (`kv_append_*`'s optional `host`): upstream's KV streaming kept a layer's authoritative K/V there.  This engine keeps
// the K/V in VRAM and passes none.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

/// The host copy of one layer's K/V: device-mapped pointers (UVA) into pinned memory, identity layout
/// `[block][kv_head][page_size][head_dim]`. All null when the layer is fully resident.
struct KvHostPools {
    uint16_t* k_pool = nullptr;   ///< fp16 mode
    uint16_t* v_pool = nullptr;
    int8_t* k_q = nullptr;        ///< int8 mode: codes
    int8_t* v_q = nullptr;
    uint16_t* k_scale = nullptr;  ///< int8 mode: fp16 scale per 64 values
    uint16_t* v_scale = nullptr;
    uint8_t* k_q4 = nullptr;      ///< q4_0 mode (kv_q4.hpp): block_q4_0 codes, 144 B per cell and head
    uint8_t* v_q4 = nullptr;
    bool present() const { return k_pool != nullptr || k_q != nullptr || k_q4 != nullptr; }
};

}  // namespace strata::kernels
