// include/lamina/kernels/dp4a.hpp - the byte-wise dot product and the spin pause the kernels share.
//
// `LAMINA_DP4A` is `__dp4a` (the i-quant kernels are transcribed from llama.cpp's vecdotq.cuh, whose operands are
// the same sites: signed bytes accumulated in int32).  `lamina_spin_pause` paces single-thread doorbell waits.
#pragma once

#include <cstdint>

#define LAMINA_DP4A(a, b, c) __dp4a((a), (b), (c))

/// Q6_K's quants less 32, from their low four bits `lo` (masked 0x0f0f0f0f) and high two `hi` (masked 0x30303030):
/// per byte lo + hi - 32, in [-32, 31].  lo + hi + 96 stays inside its byte (at most 159) and flipping the byte's top
/// bit maps it onto lo + hi - 32: an IADD3 and a LOP3, the bytes of __vsubss4(lo | hi, 0x20202020) (no byte
/// saturates), which sm_86 runs as eight instructions (upstream 6befb3d6's SWAR form, two operations shorter).
__device__ __forceinline__ int lamina_q6_quants(const int lo, const int hi) {
    return (int) (((unsigned) lo + (unsigned) hi + 0x60606060u) ^ 0x80808080u);
}

/// Yield the thread while a doorbell flag is being polled.  The length of the pause is a backoff hint, not a
/// contract.
__device__ __forceinline__ void lamina_spin_pause() { __nanosleep(100); }
