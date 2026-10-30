// Ordering primitives shared by the adaptive tier and its focused host-side regression test.
#pragma once

#include <cstdint>

namespace lamina::core {

struct AdaptiveMoveOrder {
    float gain;
    int32_t layer;
    int32_t in;
    int32_t out;
    int32_t slot;
};

// Async mode deliberately retains the historical gain-only comparator. Deterministic mode adds a
// total order after comparing gains, so equal-gain moves cannot depend on sort/input timing.
inline bool adaptive_move_less(const AdaptiveMoveOrder& a, const AdaptiveMoveOrder& b, bool deterministic) noexcept {
    if (a.gain > b.gain) return true;
    if (b.gain > a.gain) return false;
    if (!deterministic) return false;
    if (a.layer != b.layer) return a.layer < b.layer;
    if (a.in != b.in) return a.in < b.in;
    if (a.out != b.out) return a.out < b.out;
    return a.slot < b.slot;
}

}  // namespace lamina::core
