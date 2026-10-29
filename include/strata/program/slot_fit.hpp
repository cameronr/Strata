// include/strata/program/slot_fit.hpp - how many batch slots the free VRAM can carve (header-only: arithmetic).
//
// The startup sizing (generate.cpp's "--parallel %d: each added slot takes ...") knows what a slot costs; this
// decides how many of them to take.  Basalt's partial-slot start: when the wanted count does not fit, the ones
// that do are carved (the log says so), rather than running with no batch slots at all.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::program {

/// The number of slots to carve: `want` of `bytes_each` bytes each, plus `staging` bytes once for the admission
/// copies, out of `free_bytes` with `reserve` bytes left untouched (the headroom the caller keeps for its own
/// copies).  0 means "no batch slots" (nothing fits, or there was nothing to fit); a value below `want` is a
/// partial start.
inline int slots_that_fit(int want, int64_t bytes_each, int64_t staging, int64_t free_bytes, int64_t reserve) {
    if (want <= 0 || bytes_each <= 0 || free_bytes <= reserve + staging) return 0;
    const int64_t room = (free_bytes - reserve - staging) / bytes_each;
    return (int) std::min<int64_t>((int64_t) want, std::max<int64_t>(0, room));
}

}  // namespace strata::program
