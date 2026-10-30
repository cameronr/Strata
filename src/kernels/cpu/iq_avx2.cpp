// src/kernels/cpu/iq_avx2.cpp - the i-quant expert rows in 256-bit lanes, several tokens at once.
//
// The AVX-512 kernels' (iq_avx512.cpp) multi-token scheme on the CPUs without AVX-512 (AMD Zen 2/3, Intel
// Core 12th-14th gen and Core Ultra): each 32-value chunk is decoded once - grid lookups, one sign vector,
// one scale vector - and every token of the verify window applies them with a load, a `sign`, a `maddubs`,
// a `madd` and an `add`.  ggml-cpu's own AVX2 dot products for these formats are single-token: every token
// re-does the codebook lookups and the sign expansion.  The sign vector is ggml's bit_selector pattern
// (ggml-cpu/arch/x86/quants.c): pshufb-broadcast of each sign byte, AND with the bit selector, CMPEQ, OR
// with one, `vpsignb`.  The arithmetic is ggml's (ggml-cpu/quants.c, the `_generic` references) - only the
// order of the float additions differs.
//
// Formats: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22), IQ4_XS (23).  IQ1_M stays on ggml-cpu.
//
// IQ3_XXS, IQ3_S and IQ2_S also have a gathered decode (Fmt32<118>, <121>, <122>): the grid indices built in a
// register and read with one vpgather, the same words into the same lanes, so the same bits.  Where it is faster
// depends on the core, so LAMINA_IQ256_GATHER picks it per thread (expert_layout.hpp, cpu_gather_fast_here).
//
// Where the CPU has AVX-VNNI (cpu_avxvnni_ok: Alder Lake, Sapphire Rapids and later) the row kernels take a second
// copy with vpdpwssd / vpdpbusd (iq_avx2_rows.inl), the same integer sums.
//
// This file is compiled for AVX2 and must not run anything before the engine's cpu_avx2_ok() check: no runtime
// initializer at namespace scope (#391, 016ea2e), the switches are read on first use.
#include "lamina/kernels/cpu/iq_avx2.hpp"
#include "lamina/kernels/cpu/expert_layout.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

// LAMINA_AVXVNNI (CMake: the compiler has the AVX-VNNI intrinsics): GCC and Clang compile them only inside functions
// with target("avxvnni") - the VNNI copy of iq_avx2_rows.inl.
#if !defined(LAMINA_AVXVNNI)
#define LAMINA_AVXVNNI 0
#endif
#if LAMINA_AVXVNNI && (defined(__GNUC__) || defined(__clang__))
#define LAMINA_AVXVNNI_FN __attribute__((target("avxvnni")))
#else
#define LAMINA_AVXVNNI_FN
#endif

namespace lamina::kernels::cpu {
namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
#include "iq_common.inc"

// 32 sign bits -> 32 bytes of -1 (bit set) / +1: ggml's bit_selector pattern, shared by all tokens.
inline __m256i sgn_vec(uint32_t m) {
    const __m128i bm = _mm_set1_epi32((int) m);
    const __m128i lo = _mm_shuffle_epi8(bm, _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1));
    const __m128i hi = _mm_shuffle_epi8(bm, _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i bits = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    const __m256i sel = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80);
    const __m256i nz = _mm256_cmpeq_epi8(_mm256_and_si256(bits, sel), sel);
    return _mm256_or_si256(nz, _mm256_set1_epi8(1));
}

// The same sign vector read in place from the half's four sign bytes (IQ3_S, IQ2_S): the bytes come in as one
// broadcast load (vpbroadcastd from memory, a load uop) and ONE pshufb expands them, instead of sgn_vec's two
// pshufb + inserti128 - three uops on the single shuffle port of Haswell-class cores, one here.
inline __m256i sgn_vec_at(const uint8_t* p) {
    const __m256i bits = _mm256_shuffle_epi8(_mm256_set1_epi32((int) u32(p)),
                                             _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                                              2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i sel = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80,
                                         1, 2, 4, 8, 16, 32, 64, (char) 0x80);
    const __m256i nz = _mm256_cmpeq_epi8(_mm256_and_si256(bits, sel), sel);
    return _mm256_or_si256(nz, _mm256_set1_epi8(1));
}

// The two 16-value scales of one 32-value half (IQ2_XS, IQ2_S): int16 lanes 0-7 = a (values 0-15),
// 8-15 = b (values 16-31).  NOT a broadcast dword [a,b] - that would alternate the scales lane by lane.
inline __m256i sc16(int a, int b) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi16((short) a)),
                                   _mm_set1_epi16((short) b), 1);
}
// One scale for the whole 32-value half (IQ2_XXS, IQ3_XXS, IQ3_S).
inline __m256i sc32(int s) { return _mm256_set1_epi16(s); }


// The (2s+1) scale vectors as tables (IQ3_S: one 4-bit scale per 32-value half; IQ2_S: one scale byte holding
// the two 16-value scales of the half) - one load instead of the shift/or/broadcast chain of sc32()/sc16().
struct ScaleVecs {
    int16_t s32[16][16];    // all 16 lanes = 2s+1
    int16_t s16[256][16];   // lanes 0-7 = 2*(b&15)+1, lanes 8-15 = 2*(b>>4)+1 (the layout sc16() builds)
    constexpr ScaleVecs() : s32{}, s16{} {
        for (int s = 0; s < 16; ++s)
            for (int l = 0; l < 16; ++l) s32[s][l] = (int16_t) (2 * s + 1);
        for (int b = 0; b < 256; ++b)
            for (int l = 0; l < 8; ++l) {
                s16[b][l] = (int16_t) (2 * (b & 15) + 1);
                s16[b][l + 8] = (int16_t) (2 * (b >> 4) + 1);
            }
    }
};
static constexpr ScaleVecs scale_vecs{};
static_assert(scale_vecs.s32[15][0] == 31 && scale_vecs.s16[0x2F][7] == 31 && scale_vecs.s16[0x2F][8] == 5,
              "scale_vecs: 2s+1, low nibble in lanes 0-7");

inline __m256i scale_vec(const int16_t* lanes) { return _mm256_loadu_si256((const __m256i*) lanes); }

// The low index bytes q[0..7] interleaved with the spread high bits: eight 16-bit grid indices in sp[].
inline void grid_indices(const uint8_t* q, uint64_t hi, uint16_t* sp) {
    _mm_storeu_si128((__m128i*) sp, _mm_unpacklo_epi8(_mm_cvtsi64_si128((long long) u64(q)),
                                                      _mm_cvtsi64_si128((long long) hi)));
}

inline float hsum8(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// ---- per format: one 32-value half (values 64*j + 32*half .. +31) -> grid magnitudes, sign vector, scales
template <int TY> struct Fmt32;

template <> struct Fmt32<16> {   // IQ2_XXS: d, qs[32] u16
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        const uint32_t w0 = u32(q), w1 = u32(q + 4);
        g = _mm256_set_epi64x((long long) iq2xxs_grid[w0 >> 24], (long long) iq2xxs_grid[(w0 >> 16) & 255],
                              (long long) iq2xxs_grid[(w0 >> 8) & 255], (long long) iq2xxs_grid[w0 & 255]);
        sgn = _mm256_set_epi64x((long long) even_signs.v[(w1 >> 21) & 127], (long long) even_signs.v[(w1 >> 14) & 127],
                                (long long) even_signs.v[(w1 >> 7) & 127], (long long) even_signs.v[w1 & 127]);
        sc = sc32(2 * (int) (w1 >> 28) + 1);
    }
};

template <> struct Fmt32<17> {   // IQ2_XS: d, qs[32] u16 (9-bit grid index + 7-bit sign index), scales[8]
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        uint16_t v[8];
        std::memcpy(v, b + 2 + 16 * j, 16);
        const int o = 4 * half;
        g = _mm256_set_epi64x((long long) iq2xs_grid[v[o + 3] & 511], (long long) iq2xs_grid[v[o + 2] & 511],
                              (long long) iq2xs_grid[v[o + 1] & 511], (long long) iq2xs_grid[v[o] & 511]);
        uint32_t s = 0;
        for (int l = 0; l < 4; ++l) s |= (uint32_t) ksigns_iq2xs[v[o + l] >> 9] << (8 * l);
        sgn = sgn_vec(s);
        const uint8_t sb = b[66 + 2 * j + half];
        sc = sc16(2 * (sb & 15) + 1, 2 * (sb >> 4) + 1);
    }
};

template <> struct Fmt32<22> {   // IQ2_S: d, qs[64] (32 grid bytes, 32 sign bytes), qh[8], scales[8]
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        // grid_indices reads 8 index bytes; only the first 4 are this half's (the rest stay inside the block)
        alignas(16) uint16_t sp[8];
        grid_indices(b + 2 + 8 * j + 4 * half, hi_spread.iq2s[b[66 + 2 * j + half]], sp);
        g = _mm256_set_epi64x((long long) iq2s_grid[sp[3]], (long long) iq2s_grid[sp[2]],
                              (long long) iq2s_grid[sp[1]], (long long) iq2s_grid[sp[0]]);
        sgn = sgn_vec_at(b + 2 + 32 + 8 * j + 4 * half);
        sc = scale_vec(scale_vecs.s16[b[74 + 2 * j + half]]);
    }
};

template <> struct Fmt32<18> {   // IQ3_XXS: d, qs[64] grid bytes, 8 x u32 (4 x 7-bit sign index + 4-bit scale)
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        g = _mm256_set_epi32((int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t w = u32(b + 2 + 64 + 8 * j + 4 * half);
        sgn = _mm256_set_epi64x((long long) even_signs.v[(w >> 21) & 127], (long long) even_signs.v[(w >> 14) & 127],
                                (long long) even_signs.v[(w >> 7) & 127], (long long) even_signs.v[w & 127]);
        sc = sc32(2 * (int) (w >> 28) + 1);
    }
};

template <> struct Fmt32<21> {   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        const uint32_t h = b[66 + 2 * j + half];
        alignas(16) uint16_t sp[8];
        grid_indices(q, hi_spread.iq3s[h], sp);
#define G3(k) (int) iq3s_grid[sp[k]]
        g = _mm256_set_epi32(G3(7), G3(6), G3(5), G3(4), G3(3), G3(2), G3(1), G3(0));
#undef G3
        sgn = sgn_vec_at(b + 74 + 8 * j + 4 * half);
        const uint8_t s = b[106 + j];
        sc = scale_vec(scale_vecs.s32[half ? s >> 4 : s & 15]);
    }
};

// ---- the gathered decodes (LAMINA_IQ256_GATHER, see the top of the file): each its format's Fmt32 bit for bit
// The 32 sign bits of a half from the j's 8 sign bytes: one broadcast and one shuffle (sgn_vec: two and an insert).
inline __m256i sgn_half(const uint8_t* m8, int half) {
    const __m256i sb = _mm256_shuffle_epi8(
        _mm256_set1_epi64x((long long) u64(m8)),
        half ? _mm256_setr_epi8(4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7)
             : _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i sel = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
    return _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(sb, sel), sel), _mm256_set1_epi8(1));
}
// The four 7-bit even-sign indices of a u32 (bits 0-6, 7-13, 14-20, 21-27) -> their four masks, one gather.
inline __m256i even_signs4(uint32_t w) {
    const __m128i idx = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32((int) w), _mm_setr_epi32(0, 7, 14, 21)),
                                      _mm_set1_epi32(127));
    return _mm256_i32gather_epi64((const long long*) even_signs.v, idx, 8);
}

// IQ3_S: the eight 9-bit grid indices built in one register (8 low bits from qs, the 9th from bit k of qh) and read
// with one gather; the scalar form spends ~7 shuffle-port uops per half on vpinsrd and runs the index arithmetic on
// the scalar ports.  On a 14900KF's P-cores, gate/up rows of an IQ3_S expert at three tokens: 0.40 -> 0.27 ms.
template <> struct Fmt32<121> {
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        const uint32_t h = b[66 + 2 * j + half];
        const __m256i lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) q));
        const __m256i hb = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32((int) h),
                                                              _mm256_setr_epi32(8, 7, 6, 5, 4, 3, 2, 1)),
                                            _mm256_set1_epi32(0x100));
        g = _mm256_i32gather_epi32((const int*) iq3s_grid, _mm256_or_si256(lo, hb), 4);
        sgn = sgn_half(b + 74 + 8 * j, half);
        const uint8_t s = b[106 + j];
        sc = sc32(half ? 2 * (s >> 4) + 1 : 2 * (s & 15) + 1);
    }
};

// IQ3_XXS: the grid entries and the four even-sign masks each by one gather.  P-core gate/up rows at three tokens:
// 0.26 -> 0.24 ms.
template <> struct Fmt32<118> {
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        g = _mm256_i32gather_epi32((const int*) iq3xxs_grid, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) q)), 4);
        const uint32_t w = u32(b + 2 + 64 + 8 * j + 4 * half);
        sgn = even_signs4(w);
        sc = sc32(2 * (int) (w >> 28) + 1);
    }
};

// IQ2_S: the four 10-bit grid indices (qs byte k, bits 2k and 2k+1 of qh at bits 8-9) by one 64-bit gather and the
// explicit sign bits from one broadcast.  P-core gate/up rows at three tokens: 0.31 -> 0.27 ms.  IQ2_XXS measured 4%
// slower gathered (its four 8-byte grid loads are few enough already) and IQ2_XS has its own sign decode, so both
// keep their Fmt32 everywhere.
template <> struct Fmt32<122> {
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const uint8_t* qs = b + 2 + 8 * j + 4 * half;
        const uint32_t h = b[66 + 2 * j + half];
        const __m128i idx = _mm_or_si128(
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) u32(qs))),
            _mm_and_si128(_mm_sllv_epi32(_mm_set1_epi32((int) h), _mm_setr_epi32(8, 6, 4, 2)), _mm_set1_epi32(0x300)));
        g = _mm256_i32gather_epi64((const long long*) iq2s_grid, idx, 8);
        sgn = sgn_half(b + 2 + 32 + 8 * j, half);
        const uint8_t sb = b[74 + 2 * j + half];
        sc = sc16(2 * (sb & 15) + 1, 2 * (sb >> 4) + 1);
    }
};

template <> struct Fmt32<23> {   // IQ4_XS: d, scales_h, scales_l[4], qs[128] - 136 B, 8 signed sub-scales
    // ggml's ggml_vec_dot_iq4_xs_q8_K: the same 16-value codebook as IQ4_NL, but each of the eight 32-value
    // sub-blocks carries its own 6-bit scale, read as two nibbles of scales_l[p] plus two bits of scales_h, and
    // used SIGNED as (ls - 32).  The scale therefore folds into the int16 operand of madd_epi16 instead of
    // becoming a float multiply per sub-block, and the codebook sign is carried the way iq4nl_rows does it.
    static constexpr int bytes = 136;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn, __m256i& sc) {
        const int H = 2 * j + half;   // the eight 32-value sub-blocks, in the order the activation bytes come
        const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
        const __m128i bits = _mm_loadu_si128((const __m128i*) (b + 8 + 16 * H));
        const __m128i m4 = _mm_set1_epi8(0x0f);
        const __m256i q4 = _mm256_inserti128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values, _mm_and_si128(bits, m4))),
            _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), m4)), 1);
        g   = _mm256_sign_epi8(q4, q4);                       // |w|, the unsigned operand of maddubs
        sgn = _mm256_sign_epi8(_mm256_set1_epi8(1), q4);      // w's sign, applied to the activation
        const int ls = ((b[4 + (H >> 1)] >> (4 * (H & 1))) & 0xf) | (((u16(b + 2) >> (2 * H)) & 3) << 4);
        sc = _mm256_set1_epi16((short) (ls - 32));
    }
};

// ---- the row kernels, twice: the AVX2 forms and the AVX-VNNI ones (iq_avx2_rows.inl)
namespace plain {
#define LAMINA_ROWS_VNNI 0
#define LAMINA_ROWS_FN
#include "iq_avx2_rows.inl"
}  // namespace plain
#if LAMINA_AVXVNNI
namespace vnni {
#define LAMINA_ROWS_VNNI 1
#define LAMINA_ROWS_FN LAMINA_AVXVNNI_FN
#include "iq_avx2_rows.inl"
}  // namespace vnni
#endif

bool vnni_on() { return LAMINA_AVXVNNI && cpu_avxvnni_ok(); }

}  // namespace

// ggml's quantize_row_q8_K_ref (ggml-quants.c), which is also what ggml-cpu runs on x86 (no SIMD version there),
// in AVX-2: the same per-value arithmetic - one IEEE multiply by the same iscale, the same 1.5*2^23 rounding add,
// the same clamp - so the bytes are identical (q8k_quant_parity checks it).  The block's sign-carrying max is the
// FIRST value of the largest magnitude, as in the scalar loop; NaNs are skipped as there (`ax > amax` is false for
// them, and max_ps returns its second operand for a NaN first one).  A zero block leaves bsums as the reference
// does.  A verify window quantizes up to 6 tokens per layer on the host before the pool can start.
void q8k_quant_avx2(const float* x, void* vy, int64_t k) {
    block_q8_K* y = (block_q8_K*) vy;
    const int64_t nb = k / QK_K;
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 magic = _mm256_set1_ps(12582912.f);
    const __m256i mant = _mm256_set1_epi32(0x007fffff), off = _mm256_set1_epi32(0x00400000);
    const __m256i c127 = _mm256_set1_epi32(127);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int64_t i = 0; i < nb; ++i, x += QK_K) {
        __m256 m = _mm256_setzero_ps();
        for (int j = 0; j < QK_K; j += 8) m = _mm256_max_ps(_mm256_and_ps(_mm256_loadu_ps(x + j), absm), m);
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        h = _mm_max_ss(h, _mm_movehdup_ps(h));
        const float amax = _mm_cvtss_f32(h);
        if (!amax) {
            y[i].d = 0;
            std::memset(y[i].qs, 0, QK_K);
            continue;
        }
        float mx = 0.f;
        const __m256 va = _mm256_set1_ps(amax);
        for (int j = 0; j < QK_K; j += 8) {
            const int msk = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_and_ps(_mm256_loadu_ps(x + j), absm), va, _CMP_EQ_OQ));
            if (msk) {
                int b = 0;
                while (!((msk >> b) & 1)) ++b;
                mx = x[j + b];
                break;
            }
        }
        const float iscale = -127.f / mx;
        const __m256 vis = _mm256_set1_ps(iscale);
        for (int j = 0; j < QK_K; j += 32) {
            __m256i v[4];
            for (int q = 0; q < 4; ++q) {
                __m256 p = _mm256_mul_ps(vis, _mm256_loadu_ps(x + j + 8 * q));
#if defined(__GNUC__) && !defined(__clang__)
                // GCC fuses a multiply and an add into an FMA by default (-ffp-contract=fast); the reference rounds
                // the product before the add, so the product must stay a separate value
                __asm__("" : "+x"(p));
#endif
                const __m256 t = _mm256_add_ps(p, magic);
                v[q] = _mm256_min_epi32(_mm256_sub_epi32(_mm256_and_si256(_mm256_castps_si256(t), mant), off), c127);
                // the reference stores MIN(127, v) into an int8 (keeps the low byte), and its bsums add those bytes:
                // the same here, so even a degenerate block (iscale overflowing to inf) gives the same bytes
                v[q] = _mm256_srai_epi32(_mm256_slli_epi32(v[q], 24), 24);
            }
            for (int q = 0; q < 2; ++q) {   // two 16-value sums
                const __m256i s = _mm256_add_epi32(v[2 * q], v[2 * q + 1]);
                __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
                s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
                s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
                y[i].bsums[j / 16 + q] = (int16_t) _mm_cvtsi128_si32(s4);
            }
            const __m256i p16a = _mm256_packs_epi32(v[0], v[1]), p16b = _mm256_packs_epi32(v[2], v[3]);
            const __m256i p8 = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(p16a, p16b), perm);
            _mm256_storeu_si256((__m256i*) (y[i].qs + j), p8);
        }
        y[i].d = 1 / iscale;
    }
}

void iq256_gu_rows_exact_one(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                             const void* act, float* ff, int r0, int r1) {
    const auto* y = (const block_q8_K*) act;
#if LAMINA_AVXVNNI
    if (vnni_on()) {
        if (type == 16) vnni::gu_ggml_one<16>(blob, gu_row, up_off, n, y, ff, r0, r1);
        else if (type == 17) vnni::gu_ggml_one<17>(blob, gu_row, up_off, n, y, ff, r0, r1);
        else if (type == 18) {
            if (iq256_variant() & kIq256Gather) vnni::gu_ggml_one<118>(blob, gu_row, up_off, n, y, ff, r0, r1);
            else vnni::gu_ggml_one<18>(blob, gu_row, up_off, n, y, ff, r0, r1);
        } else if (type == 21) {
            if (iq256_variant() & kIq256Gather) vnni::gu_ggml_one<121>(blob, gu_row, up_off, n, y, ff, r0, r1);
            else vnni::gu_ggml_one<21>(blob, gu_row, up_off, n, y, ff, r0, r1);
        } else if (type == 22) {
            if (iq256_variant() & kIq256Gather) vnni::gu_ggml_one<122>(blob, gu_row, up_off, n, y, ff, r0, r1);
            else vnni::gu_ggml_one<22>(blob, gu_row, up_off, n, y, ff, r0, r1);
        }
        return;
    }
#endif
    if (type == 16) plain::gu_ggml_one<16>(blob, gu_row, up_off, n, y, ff, r0, r1);
    else if (type == 17) plain::gu_ggml_one<17>(blob, gu_row, up_off, n, y, ff, r0, r1);
    else if (type == 18) {
        if (iq256_variant() & kIq256Gather) plain::gu_ggml_one<118>(blob, gu_row, up_off, n, y, ff, r0, r1);
        else plain::gu_ggml_one<18>(blob, gu_row, up_off, n, y, ff, r0, r1);
    } else if (type == 21) {
        if (iq256_variant() & kIq256Gather) plain::gu_ggml_one<121>(blob, gu_row, up_off, n, y, ff, r0, r1);
        else plain::gu_ggml_one<21>(blob, gu_row, up_off, n, y, ff, r0, r1);
    } else if (type == 22) {
        if (iq256_variant() & kIq256Gather) plain::gu_ggml_one<122>(blob, gu_row, up_off, n, y, ff, r0, r1);
        else plain::gu_ggml_one<22>(blob, gu_row, up_off, n, y, ff, r0, r1);
    }
}

int iq256_variant() noexcept {
    // LAMINA_IQ256_GATHER=0/1 for every core; unset, where the calling thread's core gathers faster (per thread: a
    // hybrid CPU's pool runs on P- and E-cores).  AVX-VNNI where cpu_avxvnni_ok().  The same bits either way.
    const int s = iq256_gather_setting();
    return ((s >= 0 ? s == 1 : cpu_gather_fast_here()) ? kIq256Gather : 0) | (vnni_on() ? kIq256Vnni : 0);
}

int iq256_variants() noexcept { return kIq256Gather | (vnni_on() ? kIq256Vnni : 0); }

void iq256_gu_rows_v(int variant, int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                     const void* const* act, int nt, float* const* ff, int r0, int r1) {
    const bool g = (variant & kIq256Gather) != 0;
#if LAMINA_AVXVNNI
    if (variant & kIq256Vnni) {
        if (g) vnni::gu_type<true>(type, nt, blob, gu_row, up_off, n, act, ff, r0, r1);
        else vnni::gu_type<false>(type, nt, blob, gu_row, up_off, n, act, ff, r0, r1);
        return;
    }
#endif
    if (g) plain::gu_type<true>(type, nt, blob, gu_row, up_off, n, act, ff, r0, r1);
    else plain::gu_type<false>(type, nt, blob, gu_row, up_off, n, act, ff, r0, r1);
}

void iq256_rows_v(int variant, int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                  float* const* out, int r0, int r1) {
    const bool g = (variant & kIq256Gather) != 0;
#if LAMINA_AVXVNNI
    if (variant & kIq256Vnni) {
        if (g) vnni::dot_type<true>(type, nt, w, row_bytes, n, act, out, r0, r1);
        else vnni::dot_type<false>(type, nt, w, row_bytes, n, act, out, r0, r1);
        return;
    }
#endif
    if (g) plain::dot_type<true>(type, nt, w, row_bytes, n, act, out, r0, r1);
    else plain::dot_type<false>(type, nt, w, row_bytes, n, act, out, r0, r1);
}

void iq256_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    iq256_gu_rows_v(iq256_variant(), type, blob, gu_row, up_off, n, act, nt, ff, r0, r1);
}

void iq256_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    iq256_rows_v(iq256_variant(), type, w, row_bytes, n, act, nt, out, r0, r1);
}

void iq4nl256_down_rows_v(int variant, const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                          float* const* out, int r0, int r1) {
    const block_q8_0* y[8];
    for (int t = 0; t < nt; ++t) y[t] = (const block_q8_0*) hq[t];
#if LAMINA_AVXVNNI
    if (variant & kIq256Vnni) {
        vnni::iq4nl_nt(w, row_bytes, n, y, nt, out, r0, r1);
        return;
    }
#endif
    (void) variant;
    plain::iq4nl_nt(w, row_bytes, n, y, nt, out, r0, r1);
}

void iq4nl256_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                        int r0, int r1) {
    iq4nl256_down_rows_v(vnni_on() ? kIq256Vnni : 0, w, row_bytes, n, hq, nt, out, r0, r1);
}

}  // namespace lamina::kernels::cpu
