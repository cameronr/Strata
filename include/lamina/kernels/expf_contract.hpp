// include/lamina/kernels/expf_contract.hpp - the SwiGLU's expf under --expert-math exact, on the CPU and the GPU alike.
//
// ARM optimized-routines' expf (glibc's since 2.27; MIT / Apache-2.0 WITH LLVM-exception), with the three products its
// FMA build fuses fused here too: exp(x) = 2^(k/32) * 2^(r/32), k the integer nearest x * 32/ln2, 2^(i/32) from a
// table scaled by k's high bits, 2^(r/32) a cubic.  Within 0.502 ULP.  The C runtimes' expf differ (MSVC's UCRT is not
// glibc's), so the CPU pool takes this one where the GPU must match it bit for bit.  Each operation is its own
// statement, and on the device its own __d*_rn intrinsic, so no compiler contracts or reorders one.  The host form
// is compiled in native_expert.cpp only (no FMA instructions there: nothing to contract); swiglu_expf_parity checks it
// against the device form over every float.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define LAMINA_EXPF_HD __host__ __device__ __forceinline__
#else
#define LAMINA_EXPF_HD inline
#endif

namespace lamina::kernels {
namespace expf_detail {

// 2^(i/32) with i << 47 subtracted from its bits, so adding k << 47 scales it by 2^(k/32 - i/32)
#define LAMINA_EXPF_TAB                                                                           \
    {0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full, 0x3fef9301d0125b51ull, \
     0x3fef72b83c7d517bull, 0x3fef54873168b9aaull, 0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull, \
     0x3fef06fe0a31b715ull, 0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull, \
     0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull, 0x3feea47eb03a5585ull, \
     0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull, 0x3feea11473eb0187ull, 0x3feea589994cce13ull, \
     0x3feeace5422aa0dbull, 0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull, \
     0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull, 0x3fef3720dcef9069ull, \
     0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full, 0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull}
#if defined(__CUDACC__)
static __device__ __constant__ uint64_t tab_dev[32] = LAMINA_EXPF_TAB;
#endif
static constexpr uint64_t tab_host[32] = LAMINA_EXPF_TAB;
#undef LAMINA_EXPF_TAB

constexpr double kInvLn2N = 0x1.71547652b82fep+5;   // 32 / ln 2
constexpr double kShift = 0x1.8p+52;                // rounds to an integer, ties to even
constexpr double kC0 = 0x1.c6af84b912394p-20, kC1 = 0x1.ebfce50fac4f3p-13, kC2 = 0x1.62e42ff0c52d6p-6;

LAMINA_EXPF_HD uint32_t float_bits(float x) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(x);
#else
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return u;
#endif
}

LAMINA_EXPF_HD float bits_float(uint32_t u) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(u);
#else
    float x;
    std::memcpy(&x, &u, 4);
    return x;
#endif
}

}  // namespace expf_detail

LAMINA_EXPF_HD float expf_contract(float x) {
    namespace D = expf_detail;
    const uint32_t ux = D::float_bits(x);
    const uint32_t abstop = (ux >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {   // |x| >= 88, inf or NaN
        if (ux == 0xff800000u) return 0.0f;
        if (abstop >= 0x7f8u) return x;   // NaN (its payload kept) or +inf
        if (x > 0x1.62e42ep6f) return D::bits_float(0x7f800000u);
        if (x < -0x1.9fe368p6f) return 0.0f;
    }
    const double xd = (double) x;
#if defined(__CUDA_ARCH__)
    const double z = __dmul_rn(D::kInvLn2N, xd);
    const double kd0 = __dadd_rn(z, D::kShift);
    const uint64_t ki = (uint64_t) __double_as_longlong(kd0);
    const double kd = __dsub_rn(kd0, D::kShift);
    const double r = __dsub_rn(z, kd);
    const uint64_t t = D::tab_dev[ki & 31u] + (ki << 47);
    const double s = __longlong_as_double((long long) t);
    const double p = fma(D::kC0, r, D::kC1);
    const double r2 = __dmul_rn(r, r);
    double y = fma(D::kC2, r, 1.0);
    y = fma(p, r2, y);
    y = __dmul_rn(y, s);
    return __double2float_rn(y);
#else
    const double z = D::kInvLn2N * xd;
    const double kd0 = z + D::kShift;
    uint64_t ki;
    std::memcpy(&ki, &kd0, 8);
    const double kd = kd0 - D::kShift;
    const double r = z - kd;
    const uint64_t t = D::tab_host[ki & 31u] + (ki << 47);
    double s;
    std::memcpy(&s, &t, 8);
    const double p = std::fma(D::kC0, r, D::kC1);
    const double r2 = r * r;
    double y = std::fma(D::kC2, r, 1.0);
    y = std::fma(p, r2, y);
    y = y * s;
    return (float) y;
#endif
}

}  // namespace lamina::kernels
