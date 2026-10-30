// src/kernels/cuda/native_mma.cu - see include/lamina/kernels/native_mma.hpp.
//
// A CTA takes TM tiles of 16 weight rows (mma's M) and walks their K in stages of SB weight blocks, ST in flight: all
// its threads copy a stage of its rows into shared memory with 16-byte cp.async, a row's bytes from the 16-byte
// boundary below them (a row of a non-multiple of 16 starts `shift` bytes into its copy). A tile's UPT warps take a
// unit of the stage each and decode it into A fragments; the window's tokens are mma's N (8, zero-padded), read from
// the interleaved q8_1 copy (a fragment's 32 ints contiguous) a stage ahead. A tile's warps' sums are added in warp
// order. Each column's arithmetic is its own, so a token's result does not depend on how many tokens the window holds.
//
// mma.m16n8k16 / m16n8k32 .s8 fragments (lane = 4 g + t): A row g (and g + 8), k 4t..4t+3 (and 16 + 4t..); B column g,
// k 4t..4t+3 (and 16 + 4t..); C rows g / g + 8, columns 2t, 2t + 1.
#include "lamina/kernels/native_mma.hpp"
#include "lamina/kernels/warp.cuh"

#include "lamina/kernels/native_mmvq.hpp"
#include "q8_1_il.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace lamina::kernels {
namespace {

__device__ __forceinline__ void mma16(int (&c)[4], int a0, int a1, int b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ void mma32(int (&c)[4], int a0, int a1, int a2, int a3, int b0, int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(unsigned dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp4(unsigned dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src));
}
// byte K of w, sign-extended: prmt's selector takes byte K, then its sign (bit 3 of a nibble) into bytes 1-3
template <int K> __device__ __forceinline__ int sbyte(int w) {
    int r;
    asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(w), "n"(K | ((8 | K) << 4) | ((8 | K) << 8) | ((8 | K) << 12)));
    return r;
}
// 4 bytes of shared memory: at a 4-byte aligned address (AL) or 2 bytes past one
template <bool AL> __device__ __forceinline__ int rd(const uint8_t* p) {
    if constexpr (AL) {
        return *reinterpret_cast<const int*>(p);
    } else {
        const unsigned* q = reinterpret_cast<const unsigned*>(p - 2);
        return (int) __byte_perm(q[0], q[1], 0x5432);
    }
}
__device__ __forceinline__ float half_lo(int w) { return __half2float(__ushort_as_half((unsigned short) w)); }
__device__ __forceinline__ float half_hi(int w) { return __half2float(__ushort_as_half((unsigned short) (w >> 16))); }

// The window's tokens: the interleaved q8_1 copy of 2-8 columns, cp of them (2, 4, 8; zero-padded)
struct Acts {
    const int* bm;     // bm[(kb 8 + i) cp + c]: int i of column c's block kb
    const float* dl;   // dl[kb cp + c]: its scale
    int cp;
    __device__ __forceinline__ int b(int kb, int i, int g) const {
        return g < cp ? __ldg(bm + ((size_t) kb * 8 + i) * cp + g) : 0;
    }
    __device__ __forceinline__ float2 d(int kb, int c0) const {   // columns c0, c0 + 1 (c0 even)
        return c0 < cp ? __ldg(reinterpret_cast<const float2*>(dl + (size_t) kb * cp + c0)) : make_float2(0.0f, 0.0f);
    }
};
// a unit's tokens: NK q8_1 blocks from kb0, both ints of a lane per block (k 4t.., 16 + 4t..) and its columns' scales
template <int NK> struct Pre {
    int b[NK][2];
    float2 d[NK];
    __device__ __forceinline__ void load(const Acts& x, int kb0, int t, int g) {
#pragma unroll
        for (int k = 0; k < NK; ++k) {
            b[k][0] = x.b(kb0 + k, t, g);
            b[k][1] = x.b(kb0 + k, 4 + t, g);
            d[k] = x.d(kb0 + k, 2 * t);
        }
    }
};

// ---------------------------------------------------------------- Q6_K: 210 bytes a 256 values
// ql[128] low 4 bits, qh[64] high 2 bits, scales[16] int8 (a 16 values each), d (half). Value 128h + 32q + l:
// ql[64h + 32(q & 1) + l] >> 4(q >> 1), qh[32h + l] >> 2q, scale 8h + 2q + l/16; q8_1 block 8b + 4h + q, byte l.
// A stage holds 2 blocks, the second 2 bytes off its first one's word parity (rows of 4-byte multiples). Unit u: block
// u >> 1's h = u & 1: 8 mma (q x the 32 values' two halves), each q8_1 block's halves added in integers.
// 4 bytes of 6-bit values v (0..63) as int8 v - 32: bit 7 set, 32 subtracted, bit 7 flipped (no borrow across bytes)
__device__ __forceinline__ int q6(int lo4, int hi2) {
    const int v = (lo4 & 0x0f0f0f0f) | (hi2 & 0x30303030) | (int) 0x80808080;
    return (v - 0x20202020) ^ (int) 0x80808080;
}
struct Q6K {
    static constexpr int VALUES = 256, BLOCK = 210, SB = 2, UPT = 4, NK = 4;
    static constexpr int PITCH = 432;   // 420 bytes + a shift of up to 12, at 108 words (12 mod 32: no conflicts)
    __device__ __forceinline__ static int kb0(int s, int u) { return 16 * s + 8 * (u >> 1) + 4 * (u & 1); }
    template <bool AL>
    __device__ __forceinline__ static void half(const uint8_t* r0, const uint8_t* r1, int h, int t, const Pre<NK>& x,
                                                float (&acc)[4]) {
        const uint8_t* p[2] = {r0, r1};
        int la[2][2], lb[2][2], qh[2][2], sa[2], sb[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                const int o = 16 * hf + 4 * t;
                la[r][hf] = rd<AL>(p[r] + 64 * h + o);
                lb[r][hf] = rd<AL>(p[r] + 64 * h + 32 + o);
                qh[r][hf] = rd<AL>(p[r] + 128 + 32 * h + o);
            }
            sa[r] = rd<AL>(p[r] + 192 + 8 * h);
            sb[r] = rd<AL>(p[r] + 196 + 8 * h);
        }
        float bacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            int c[2][4];
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                int a[2];
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const int l = (q & 1) ? lb[r][hf] : la[r][hf];
                    a[r] = q6((q >> 1) ? l >> 4 : l, (2 * q <= 4) ? qh[r][hf] << (4 - 2 * q) : qh[r][hf] >> (2 * q - 4));
                }
                c[hf][0] = c[hf][1] = c[hf][2] = c[hf][3] = 0;
                mma16(c[hf], a[0], a[1], x.b[q][hf]);
            }
            int s0[2], s1[2];   // the halves' scales 2q, 2q + 1 of the h's 8 (bytes of sa for q < 2, else sb)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int w = q < 2 ? sa[r] : sb[r];
                s0[r] = (q & 1) ? sbyte<2>(w) : sbyte<0>(w);
                s1[r] = (q & 1) ? sbyte<3>(w) : sbyte<1>(w);
            }
            bacc[0] += x.d[q].x * (float) (c[0][0] * s0[0] + c[1][0] * s1[0]);
            bacc[1] += x.d[q].y * (float) (c[0][1] * s0[0] + c[1][1] * s1[0]);
            bacc[2] += x.d[q].x * (float) (c[0][2] * s0[1] + c[1][2] * s1[1]);
            bacc[3] += x.d[q].y * (float) (c[0][3] * s0[1] + c[1][3] * s1[1]);
        }
        const float d0 = half_hi(rd<!AL>(p[0] + 206)), d1 = half_hi(rd<!AL>(p[1] + 206));
        acc[0] += d0 * bacc[0];
        acc[1] += d0 * bacc[1];
        acc[2] += d1 * bacc[2];
        acc[3] += d1 * bacc[3];
    }
    __device__ __forceinline__ static void unit(const uint8_t* r0, const uint8_t* r1, int u, int t, const Pre<NK>& x,
                                                float (&acc)[4]) {
        if (u >> 1) half<false>(r0 + 210, r1 + 210, u & 1, t, x, acc);
        else half<true>(r0, r1, u & 1, t, x, acc);
    }
};

// ---------------------------------------------------------------- Q4_K / Q5_K: 144 / 176 bytes a 256 values
// dm (half2: d, dmin), scales[12] (6-bit scales and mins of 8 sub-blocks of 32), [Q5_K: qh[32]], qs[128]. Values
// 64j + l (l < 32): qs[32j + l] & 15, 64j + 32 + l: qs[32j + l] >> 4 [| bit 2j / 2j + 1 of qh[l], << 4]; sub-block
// 2j / 2j + 1 = q8_1 block 8b + 2j (+ 1). y = d sc sum(q u) - dmin m sum(u) per sub-block, sum(u) the column's
// quantized sum (an mma with A = ones gives it in the C layout). A stage holds 2 blocks; unit u: block u >> 1's
// j = 2(u & 1), + 1: 4 sub-blocks.
template <bool Q5>
struct QK {
    static constexpr int VALUES = 256, BLOCK = Q5 ? 176 : 144, SB = 2, UPT = 4, NK = 4, QS = Q5 ? 48 : 16;
    static constexpr int PITCH = Q5 ? 368 : 304;   // 352 / 288 bytes at 92 / 76 words (28 / 12 mod 32)
    __device__ __forceinline__ static int kb0(int s, int u) { return 16 * s + 8 * (u >> 1) + 4 * (u & 1); }
    __device__ __forceinline__ static void unit(const uint8_t* r0, const uint8_t* r1, int u, int t, const Pre<NK>& x,
                                                float (&acc)[4]) {
        const uint8_t* p[2] = {r0 + BLOCK * (u >> 1), r1 + BLOCK * (u >> 1)};
        const int jp = u & 1;   // j = 2 jp, 2 jp + 1
        int lo[2][2], hi[2][2], hl[2] = {0, 0}, hh[2] = {0, 0}, sc[2][4], mn[2][4];
        float dd[2], dm[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int dmw = rd<true>(p[r]);
            dd[r] = half_lo(dmw);
            dm[r] = half_hi(dmw);
            const int s0 = rd<true>(p[r] + 4), s4 = rd<true>(p[r] + 8), s8 = rd<true>(p[r] + 12);
            // get_scale_min_k4 of sub-blocks 4 jp .. 4 jp + 3
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int j = 4 * jp + k;
                if (j < 4) {
                    sc[r][k] = (s0 >> (8 * j)) & 63;
                    mn[r][k] = (s4 >> (8 * j)) & 63;
                } else {
                    const int jj = j - 4;
                    sc[r][k] = ((s8 >> (8 * jj)) & 0xF) | ((((s0 >> (8 * jj)) >> 6) & 3) << 4);
                    mn[r][k] = ((s8 >> (8 * jj + 4)) & 0xF) | ((((s4 >> (8 * jj)) >> 6) & 3) << 4);
                }
            }
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int j = 2 * jp + k;
                lo[r][k] = rd<true>(p[r] + QS + 32 * j + 4 * t);
                hi[r][k] = rd<true>(p[r] + QS + 32 * j + 16 + 4 * t);
            }
            if (Q5) {
                hl[r] = rd<true>(p[r] + 16 + 4 * t);
                hh[r] = rd<true>(p[r] + 32 + 4 * t);
            }
        }
        float bd[4] = {0.0f, 0.0f, 0.0f, 0.0f}, bm[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int k = 0; k < 4; ++k) {   // sub-block 4 jp + k: j = 2 jp + k / 2, nibble k & 1
            const int jj = k >> 1, nib = k & 1, j = 2 * jp + jj;
            int a[4];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                a[r] = (lo[r][jj] >> (4 * nib)) & 0x0f0f0f0f;
                a[r + 2] = (hi[r][jj] >> (4 * nib)) & 0x0f0f0f0f;
                if (Q5) {
                    a[r] |= ((hl[r] >> (2 * j + nib)) & 0x01010101) << 4;
                    a[r + 2] |= ((hh[r] >> (2 * j + nib)) & 0x01010101) << 4;
                }
            }
            int c[4] = {0, 0, 0, 0}, s[4] = {0, 0, 0, 0};
            mma32(c, a[0], a[1], a[2], a[3], x.b[k][0], x.b[k][1]);
            mma32(s, 0x01010101, 0x01010101, 0x01010101, 0x01010101, x.b[k][0], x.b[k][1]);
            const float2 d = x.d[k];
            bd[0] += d.x * (float) (c[0] * sc[0][k]);
            bd[1] += d.y * (float) (c[1] * sc[0][k]);
            bd[2] += d.x * (float) (c[2] * sc[1][k]);
            bd[3] += d.y * (float) (c[3] * sc[1][k]);
            bm[0] += d.x * (float) (s[0] * mn[0][k]);
            bm[1] += d.y * (float) (s[1] * mn[0][k]);
            bm[2] += d.x * (float) (s[2] * mn[1][k]);
            bm[3] += d.y * (float) (s[3] * mn[1][k]);
        }
        acc[0] += dd[0] * bd[0] - dm[0] * bm[0];
        acc[1] += dd[0] * bd[1] - dm[0] * bm[1];
        acc[2] += dd[1] * bd[2] - dm[1] * bm[2];
        acc[3] += dd[1] * bd[3] - dm[1] * bm[3];
    }
};

// ---------------------------------------------------------------- IQ4_XS: 136 bytes a 256 values
// d (half), scales_h (u16), scales_l[4], qs[128]: sub-block ib of 32 (q8_1 block 8b + ib), values 32ib + l from
// qs[16ib + l % 16], the low nibble below 16 and the high one above, through the IQ4 codebook; its 6-bit scale - 32.
// A stage holds 2 blocks; unit u: block u >> 1's sub-blocks 4(u & 1) .. + 3.
__device__ __align__(4) int8_t mma_iq4_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                                     1,    13,   25,  38,  53,  69,  89,  113};
__device__ __forceinline__ int2 iq4_lookup(int q4, const uint32_t (&tb)[4]) {   // the low nibbles', the high ones'
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(tb[0], tb[1], q4 >> shift);
        const uint32_t high = __byte_perm(tb[2], tb[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, sel >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}
struct IQ4XS {
    static constexpr int VALUES = 256, BLOCK = 136, SB = 2, UPT = 4, NK = 4;
    static constexpr int PITCH = 272;   // 68 words (4 mod 32)
    __device__ __forceinline__ static int kb0(int s, int u) { return 16 * s + 8 * (u >> 1) + 4 * (u & 1); }
    __device__ __forceinline__ static void unit(const uint8_t* r0, const uint8_t* r1, int u, int t, const Pre<NK>& x,
                                                float (&acc)[4]) {
        const uint8_t* p[2] = {r0 + 136 * (u >> 1), r1 + 136 * (u >> 1)};
        const int ib0 = 4 * (u & 1);
        const uint32_t* t32 = reinterpret_cast<const uint32_t*>(mma_iq4_values);
        const uint32_t tb[4] = {t32[0], t32[1], t32[2], t32[3]};
        int2 v[2][4];
        int ls[2][4];
        float dw[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int w0 = rd<true>(p[r]), w1 = rd<true>(p[r] + 4);   // d, scales_h; scales_l
            dw[r] = half_lo(w0);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int ib = ib0 + k;
                v[r][k] = iq4_lookup(rd<true>(p[r] + 8 + 16 * ib + 4 * t), tb);
                ls[r][k] = (((w1 >> (8 * (ib >> 1))) >> (4 * (ib & 1))) & 0x0f) | (((w0 >> (16 + 2 * ib)) & 3) << 4);
            }
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            int c[4] = {0, 0, 0, 0};
            mma32(c, v[0][k].x, v[1][k].x, v[0][k].y, v[1][k].y, x.b[k][0], x.b[k][1]);
            const float2 d = x.d[k];
            acc[0] += (dw[0] * d.x) * (float) (c[0] * (ls[0][k] - 32));
            acc[1] += (dw[0] * d.y) * (float) (c[1] * (ls[0][k] - 32));
            acc[2] += (dw[1] * d.x) * (float) (c[2] * (ls[1][k] - 32));
            acc[3] += (dw[1] * d.y) * (float) (c[3] * (ls[1][k] - 32));
        }
    }
};

// TM tiles of 16 rows a CTA, F::UPT warps a tile (a unit each a stage), ST stages in flight
template <class F, int TM, int ST>
__launch_bounds__(TM * F::UPT * 32)
__global__ void native_mma_kernel(const uint8_t* __restrict__ w, Acts acts, float* __restrict__ y, int n_in,
                                  int n_out, int ncols) {
    constexpr int UPT = F::UPT, W = UPT * TM, ROWS = 16 * TM, PITCH = F::PITCH, BYTES = F::SB * F::BLOCK;
    constexpr int CH = PITCH / 16, NCP = (ROWS * CH + W * 32 - 1) / (W * 32);
    extern __shared__ __align__(16) uint8_t smem[];
    float* red = reinterpret_cast<float*>(smem + ST * ROWS * PITCH);   // [W][16][8]
    const int lane = (int) threadIdx.x & 31, warp = (int) threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int tile = warp % TM, u = warp / TM;
    const int row0 = (int) blockIdx.x * ROWS;
    const size_t row_bytes = (size_t) n_in / F::VALUES * F::BLOCK;
    const int stages = n_in / (F::SB * F::VALUES);
    const uint8_t* end = w + row_bytes * (size_t) n_out;
    // this thread's copies of a stage: rows and chunks fixed, the stage's offset added; the row's stage starts
    // (row + offset) & 15 bytes into its copy, chunk c covering its copy's bytes [16c, 16c + 16)
    uintptr_t rowa[NCP];
    unsigned dst[NCP];
    int ch[NCP];
    const unsigned wb0 = (unsigned) __cvta_generic_to_shared(smem);
#pragma unroll
    for (int k = 0; k < NCP; ++k) {
        const int i = (int) threadIdx.x + k * W * 32, r = i / CH, c = i - r * CH;
        rowa[k] = (uintptr_t) (w + (size_t) min(row0 + r, n_out - 1) * row_bytes);
        dst[k] = wb0 + r * PITCH + 16 * c;
        ch[k] = i < ROWS * CH ? 16 * c : 1 << 30;   // past the rows: never copied
    }
    auto issue = [&](int s) {
        if (s < stages) {
            const size_t off = (size_t) s * BYTES;
            const unsigned sb = (s % ST) * ROWS * PITCH;
#pragma unroll
            for (int k = 0; k < NCP; ++k) {
                const uintptr_t a = rowa[k] + off;
                if (ch[k] < (int) ((a & 15) + BYTES)) {
                    const uint8_t* src = reinterpret_cast<const uint8_t*>((a & ~(uintptr_t) 15) + ch[k]);
                    if (src + 16 <= end) {
                        cp16(dst[k] + sb, src);
                    } else {   // the matrix's last row: its words, not past its end
                        for (int o = 0; o < 16 && src + o < end; o += 4) cp4(dst[k] + sb + o, src + o);
                    }
                }
            }
        }
        cp_commit();   // an empty group past the last stage keeps the wait count fixed
    };
    const uintptr_t rp0 = (uintptr_t) (w + (size_t) min(row0 + 16 * tile + g, n_out - 1) * row_bytes);
    const uintptr_t rp1 = (uintptr_t) (w + (size_t) min(row0 + 16 * tile + g + 8, n_out - 1) * row_bytes);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int s = 0; s < ST - 1; ++s) issue(s);
    Pre<F::NK> pf;
    pf.load(acts, F::kb0(0, u), t, g);
    for (int s = 0; s < stages; ++s) {
        issue(s + ST - 1);
        const Pre<F::NK> cur = pf;
        if (s + 1 < stages) pf.load(acts, F::kb0(s + 1, u), t, g);
        cp_wait<ST - 1>();
        __syncthreads();
        const uint8_t* b = smem + (s % ST) * ROWS * PITCH + 16 * tile * PITCH;
        F::unit(b + g * PITCH + (int) ((rp0 + (size_t) s * BYTES) & 15),
                b + (g + 8) * PITCH + (int) ((rp1 + (size_t) s * BYTES) & 15), u, t, cur, acc);
        __syncthreads();
    }
    // each tile's warps' partial sums, added in warp order
    float* rw = red + warp * 128;
    rw[g * 8 + 2 * t] = acc[0];
    rw[g * 8 + 2 * t + 1] = acc[1];
    rw[(g + 8) * 8 + 2 * t] = acc[2];
    rw[(g + 8) * 8 + 2 * t + 1] = acc[3];
    __syncthreads();
    for (int i = (int) threadIdx.x; i < TM * 128; i += W * 32) {
        const int tl = i >> 7, r = (i >> 3) & 15, c = i & 7;
        float sum = red[tl * 128 + r * 8 + c];
#pragma unroll
        for (int k = 1; k < UPT; ++k) sum += red[(k * TM + tl) * 128 + r * 8 + c];
        const int row = row0 + 16 * tl + r;
        if (c < ncols && row < n_out) y[(size_t) c * n_out + row] = sum;
    }
}

template <class F, int TM, int ST>
void launch(const void* w, const Acts& a, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    constexpr int smem = ST * 16 * TM * F::PITCH + F::UPT * TM * 512;
    static_assert(smem <= 48 * 1024, "within the default dynamic shared memory");
    native_mma_kernel<F, TM, ST><<<(unsigned) ((n_out + 16 * TM - 1) / (16 * TM)), TM * F::UPT * 32, smem, s>>>(
        static_cast<const uint8_t*>(w), a, y, n_in, n_out, ncols);
}

// Where the tensor cores are the faster on the RTX 3090 (scratch/bi/mma_bench.cu against native_mmvq_il's kernels and
// MMQ, every format and shape of the two model files at 1-8 tokens): by format and rows (the first class a matrix
// reaches), from n0 tokens, with tm tiles a CTA and st stages in flight. The heads' class also on the RTX 5070 Ti.
// Not Q8_0 (UD-Q4_K_XL's): faster alone from 5 tokens, but its shorter dense chain and head took time from the
// adaptive tiers' copies, and that model's CPU-bound rounds were 0.3-1.3% longer.
struct Rule { int type, min_out, n0, tm, st; };
constexpr Rule kRules[] = {
    {14, 65536, 2, 2, 2}, {14, 12288, 2, 1, 2}, {14, 8192, 2, 2, 2}, {14, 4096, 2, 1, 2}, {14, 2048, 2, 1, 3},   // Q6_K
    {13, 12288, 6, 1, 2}, {13, 8192, 5, 2, 2},  {13, 4096, 6, 1, 3}, {13, 2048, 6, 1, 2},                        // Q5_K
    {12, 12288, 5, 1, 2}, {12, 8192, 5, 2, 2},  {12, 4096, 5, 1, 3}, {12, 2048, 6, 1, 2},                        // Q4_K
    {23, 12288, 6, 1, 2}, {23, 8192, 5, 1, 2},  {23, 4096, 6, 1, 3},                                             // IQ4_XS
};

template <class F>
bool run(const Rule& r, const void* w, const Acts& a, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    if (n_in % (F::SB * F::VALUES) != 0) return false;
    // stages from 16-byte boundaries: Q6_K's rows of 4-byte multiples (their blocks' parity in rd<AL>), the others'
    // rows and stages of 16-byte multiples
    const size_t row_bytes = (size_t) n_in / F::VALUES * F::BLOCK;
    const unsigned al = F::BLOCK == 210 ? 4u : 16u;
    if ((uintptr_t) w % al != 0 || row_bytes % al != 0) return false;
    if (r.tm == 2) launch<F, 2, 2>(w, a, y, n_in, n_out, ncols, s);
    else if (r.st == 3) launch<F, 1, 3>(w, a, y, n_in, n_out, ncols, s);
    else launch<F, 1, 2>(w, a, y, n_in, n_out, ncols, s);
    return true;
}

}  // namespace

bool native_mma(int ggml_type, const void* weights, const void* x_il, float* y, int n_in, int n_out, int ncols,
                void* stream) {
    if (ncols < 2 || ncols > 8) return false;
    const Rule* r = nullptr;
    for (const Rule& e : kRules)
        if (e.type == ggml_type && n_out >= e.min_out) {
            r = &e;
            break;
        }
    if (r == nullptr || ncols < r->n0) return false;
    const Q81IlParts parts = q8_1_il_parts(x_il, n_in, ncols);
    const Acts a{parts.bm, parts.d, native_q8_1_il_cp(ncols)};
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ggml_type) {
    case 14: return run<Q6K>(*r, weights, a, y, n_in, n_out, ncols, s);
    case 13: return run<QK<true>>(*r, weights, a, y, n_in, n_out, ncols, s);
    case 12: return run<QK<false>>(*r, weights, a, y, n_in, n_out, ncols, s);
    case 23: return run<IQ4XS>(*r, weights, a, y, n_in, n_out, ncols, s);
    default: return false;
    }
}

}  // namespace lamina::kernels
