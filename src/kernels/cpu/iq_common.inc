// src/kernels/cpu/iq_common.inc - what iq_avx2.cpp and iq_avx1.cpp (its 128-bit port) share: unaligned loads, the
// sign and high-bit tables, the prefetch distance.  Included INSIDE each file's anonymous namespace, after its own
// includes: each file compiles it with its own /arch flags, and the anonymous namespace keeps every copy to its file
// (an ordinary inline function would leave the linker one copy for all callers, possibly the AVX2 build's).

inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// keven_signs_q2xs (ggml keeps it static in arch/x86/quants.c): one u64 per 7-bit sign index, byte k = 0xFF when
// bit k of ksigns_iq2xs[i] is set, 0x01 otherwise.  With this a whole 32-value sign vector is four scalar loads
// and a set_epi64x - no ksigns byte packing chain and no bit_selector expansion.
//
// Computed at COMPILE time (constexpr), so the files have no static constructor at all.  They are compiled for
// AVX2 (or AVX), and a runtime constructor there runs before main() on every CPU: MSVC turned its variable shift
// into BMI2 `shlx` (#391, demetree: strata.exe exited 0xC000001D before printing anything on a Sandy Bridge Xeon)
// and GCC vectorised the loop into AVX2 (`vpbroadcastb`, found on an AVX-only Xeon E5 by the Strata_Dirigo fork).
// ksigns_iq2xs[i] is i's 7 bits plus an even-parity bit 7 (ggml-common.h); a const array is not usable in a
// constant expression, so the byte is derived the same way here, and native_expert_parity checks these kernels
// against ggml-cpu's.
struct EvenSigns {
    uint64_t v[128];
    constexpr EvenSigns() : v{} {
        for (int i = 0; i < 128; ++i) {
            int par = 0;
            for (int k = 0; k < 7; ++k) par ^= (i >> k) & 1;
            const int s = i | (par << 7);          // == ksigns_iq2xs[i]
            uint64_t r = 0;
            for (int k = 0; k < 8; ++k) r |= (uint64_t) (((s >> k) & 1) ? 0xFF : 0x01) << (8 * k);
            v[i] = r;
        }
    }
};
static constexpr EvenSigns even_signs{};
static_assert(even_signs.v[0] == 0x0101010101010101ull && even_signs.v[1] == 0xFF010101010101FFull,
              "keven_signs_q2xs: byte k = 0xFF when bit k of ksigns_iq2xs[i] is set");

// IQ3_S / IQ2_S grid indices: the high index bits of a 32-value half (one qh byte) spread one index per byte,
// so a single punpcklbw puts them next to the low index bytes and every grid lookup is a 16-bit load plus the
// table load - instead of a shift, a mask and an or per index.  IQ3_S: byte k = bit k (the 9th bit of index k).
// IQ2_S: byte k = bits 2k,2k+1 (bits 8-9 of index k), k < 4.  constexpr for the same reason as EvenSigns.
struct HiSpread {
    uint64_t iq3s[256];
    uint64_t iq2s[256];
    constexpr HiSpread() : iq3s{}, iq2s{} {
        for (int h = 0; h < 256; ++h) {
            uint64_t r3 = 0, r2 = 0;
            for (int k = 0; k < 8; ++k) r3 |= (uint64_t) ((h >> k) & 1) << (8 * k);
            for (int k = 0; k < 4; ++k) r2 |= (uint64_t) ((h >> (2 * k)) & 3) << (8 * k);
            iq3s[h] = r3;
            iq2s[h] = r2;
        }
    }
};
static constexpr HiSpread hi_spread{};
static_assert(hi_spread.iq3s[0x81] == 0x0100000000000001ull && hi_spread.iq2s[0xE4] == 0x0000000003020100ull,
              "hi_spread: one high-bit group per index byte");

// E-2 (iq_avx512.cpp) on these paths: the pool streams the expert rows from DRAM at ~25 GB/s (4 KB pages when
// large pages are refused), so ask for the bytes a few blocks before the decode needs them - 2048 B is two gate/up
// rows ahead, a row is ~1 KB.  Same switch as the AVX-512 kernels: LAMINA_IQ_PREFETCH is the distance in bytes,
// 0 = off, default 2048.  Measured on a Zen 3 5700X3D (no AVX-512) on IQ3_S decode: -4% on the gate/up phase,
// +1.0 GB/s over the rows, -1.3% ms/round end to end.  The non-temporal hint measured worse than T0 at the same
// distance, so this keeps T0.
int prefetch_distance() {
    static const int d = [] {
        const char* v = std::getenv("LAMINA_IQ_PREFETCH");
        return v ? std::atoi(v) : 2048;
    }();
    return d;
}

inline void rows_ahead(const uint8_t* blk, int pf) {
    if (pf <= 0) return;
    _mm_prefetch((const char*) blk + pf, _MM_HINT_T0);
    _mm_prefetch((const char*) blk + pf + 64, _MM_HINT_T0);
}
