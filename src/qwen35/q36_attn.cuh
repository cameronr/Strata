// src/qwen35/q36_attn.cuh - the Qwen3.6 path's prompt attention on the tensor cores (flash attention, causal, the cache
// in FP16 or q8_0 rows): a block takes 64 query rows, its tokens' G query heads of one KV head (row r: token r / G, head
// r % G), and walks the cells its last token sees in tiles of 32, double-buffered through cp.async (q8_0: through
// registers, scaled into FP16 tiles).  Per tile:
//   scores   warp w: rows of m-tile w & 3, cells (w >> 2) * 16 .. +16; Q's FP16 fragments in registers, K through
//            ldmatrix; mma m16n8k16, FP32 sums;
//   softmax  the causal mask and the online max / sum, the two warps of an m-tile combined in shared memory; the
//            probabilities to shared memory in FP16;
//   values   warp w: rows of m-tile w & 3, dims (w >> 2) * 128 .. +128; P and V (ldmatrix.trans) through mma.
// The output, divided by the sum and times sigmoid(gate), goes to `out` [T][H][256].  K and V tiles in shared memory:
// cell c's 16-byte chunk j at j ^ (c & 7); the probabilities: row r's chunk j at j ^ ((r >> 1) & 3) (ldmatrix's eight
// rows on distinct banks).
#pragma once

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

namespace q36a {

using strata::kernels::cp_async16;
using strata::kernels::cp_commit;
using strata::kernels::cp_wait;
using strata::kernels::ldsm4;
using strata::kernels::ldsm4t;

constexpr int kHD = 256;     // the head size
constexpr int kBc = 32;      // cells a tile
constexpr int kRows = 64;    // query rows a block

__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned h2(float a, float b) {
    const __half2 v = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&v);
}
// a K or V tile's cell c, 16-byte chunk j, in halves
__device__ __forceinline__ int kv_at(int c, int j) { return c * kHD + ((j ^ (c & 7)) << 3); }
// the probabilities' row r, 16-byte chunk j, in halves
__device__ __forceinline__ int p_at(int r, int j) { return r * kBc + ((j ^ ((r >> 1) & 3)) << 3); }

struct __align__(16) FaSmem {
    __half k[2][kBc * kHD];
    __half v[2][kBc * kHD];
    __half p[kRows * kBc];
    float red[2][2][kRows];   // [max | sum][the m-tile's warp: cells 0-15 | 16-31][row]
};

// cells [base, base + kBc) of a head's K or V (n of them there: zeros past them, which the mask keeps out) into a tile
__device__ __forceinline__ void load_tile(__half* dst, const __half* src, int n) {
    for (int i = (int) threadIdx.x; i < kBc * (kHD / 8); i += (int) blockDim.x) {
        const int c = i >> 5, j = i & 31;
        __half* d = dst + kv_at(c, j);
        if (c < n) cp_async16(d, src + (size_t) c * kHD + j * 8);
        else *reinterpret_cast<uint4*>(d) = make_uint4(0, 0, 0, 0);
    }
}

// The cache in q8_0 rows (a cell's 256 values in int8, then its eight blocks' FP16 scales): a tile's 8-value chunks,
// four a thread (blockDim 256), held in registers from their load until they go to the tile as FP16 (value times scale)
constexpr int kQ8Row = kHD + 16;
struct Q8Stage {
    uint2 v[4];
    __half d[4];
};
__device__ __forceinline__ void q8_load(Q8Stage& s, const uint8_t* src, int n) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int i = (int) threadIdx.x + 256 * k, c = i >> 5, j = i & 31;
        if (c < n) {
            const uint8_t* row = src + (size_t) c * kQ8Row;
            s.v[k] = *reinterpret_cast<const uint2*>(row + j * 8);
            s.d[k] = reinterpret_cast<const __half*>(row + kHD)[j >> 2];
        } else {
            s.v[k] = make_uint2(0, 0);
            s.d[k] = __float2half(0.f);
        }
    }
}
__device__ __forceinline__ void q8_store(__half* dst, const Q8Stage& s) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int i = (int) threadIdx.x + 256 * k, c = i >> 5, j = i & 31;
        const int8_t* b = reinterpret_cast<const int8_t*>(&s.v[k]);
        const float d = __half2float(s.d[k]);
        *reinterpret_cast<uint4*>(dst + kv_at(c, j)) =
            make_uint4(h2(b[0] * d, b[1] * d), h2(b[2] * d, b[3] * d), h2(b[4] * d, b[5] * d), h2(b[6] * d, b[7] * d));
    }
}

/// A prompt chunk's attention: T tokens at positions p0 + t, the cache (FP16, [HKV][ctx][256], or q8_0 rows with Q8)
/// holding every cell up to p0 + T - 1, queries q and gates [T][H][256] (FP32).  grid (ceil(T / (64 / G)), HKV), block
/// 256, shared sizeof(FaSmem).  With Q8 the next tile waits in registers through the current one's products.
template <int G, bool Q8>
__global__ void __launch_bounds__(256) k_fa_prompt(const float* __restrict__ q, const __half* __restrict__ kc,
                                                   const __half* __restrict__ vc, const float* __restrict__ gate,
                                                   float* __restrict__ out, int T, int H, int ctx, int p0, float scale) {
    constexpr int TT = kRows / G;   // tokens a block
    extern __shared__ __align__(16) unsigned char fa_raw[];
    FaSmem& sm = *reinterpret_cast<FaSmem*>(fa_raw);
    const int w = (int) threadIdx.x >> 5, lane = (int) threadIdx.x & 31, tig = lane & 3;
    const int mt = w & 3, hf = w >> 2;
    const int kvh = (int) blockIdx.y, t0 = (int) blockIdx.x * TT;
    // this thread's two rows of the m-tile, their tokens and positions (a padding row sees nothing)
    const int rA = mt * 16 + (lane >> 2), rB = rA + 8;
    const int tA = t0 + rA / G, tB = t0 + rB / G;
    const bool okA = rA < TT * G && tA < T, okB = rB < TT * G && tB < T;
    const int posA = okA ? p0 + tA : -1, posB = okB ? p0 + tB : -1;
    const size_t qA = okA ? ((size_t) tA * H + kvh * G + rA % G) * kHD : 0;
    const size_t qB = okB ? ((size_t) tB * H + kvh * G + rB % G) * kHD : 0;
    // the queries as A fragments: a0 (row rA, dims 16k + 2 tig, +1), a1 (row rB), a2 / a3 the same 8 dims on
    unsigned qa[16][4];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int d = 16 * k + 2 * tig;
        const float2 x0 = okA ? *reinterpret_cast<const float2*>(q + qA + d) : make_float2(0.f, 0.f);
        const float2 x1 = okB ? *reinterpret_cast<const float2*>(q + qB + d) : make_float2(0.f, 0.f);
        const float2 x2 = okA ? *reinterpret_cast<const float2*>(q + qA + d + 8) : make_float2(0.f, 0.f);
        const float2 x3 = okB ? *reinterpret_cast<const float2*>(q + qB + d + 8) : make_float2(0.f, 0.f);
        qa[k][0] = h2(x0.x, x0.y);
        qa[k][1] = h2(x1.x, x1.y);
        qa[k][2] = h2(x2.x, x2.y);
        qa[k][3] = h2(x3.x, x3.y);
    }
    const int n_cells = p0 + min(t0 + TT, T);   // the block's last token sees the cells below this
    const int n_tiles = (n_cells + kBc - 1) / kBc;
    const __half* kh = kc + (size_t) kvh * ctx * kHD;
    const __half* vh = vc + (size_t) kvh * ctx * kHD;
    const uint8_t* kq = reinterpret_cast<const uint8_t*>(kc) + (size_t) kvh * ctx * kQ8Row;
    const uint8_t* vq = reinterpret_cast<const uint8_t*>(vc) + (size_t) kvh * ctx * kQ8Row;
    float mA = -INFINITY, mB = -INFINITY, lA = 0.f, lB = 0.f;
    float o[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    Q8Stage sk, sv;
    if constexpr (Q8) {
        q8_load(sk, kq, min(kBc, n_cells));
        q8_load(sv, vq, min(kBc, n_cells));
        q8_store(sm.k[0], sk);
        q8_store(sm.v[0], sv);
    } else {
        load_tile(sm.k[0], kh, min(kBc, n_cells));
        load_tile(sm.v[0], vh, min(kBc, n_cells));
        cp_commit();
    }
    const int li = lane >> 3, lr = lane & 7;   // ldmatrix: this lane's matrix and row in it
    for (int it = 0; it < n_tiles; ++it) {
        const int st = it & 1, base = it * kBc;
        if constexpr (Q8) {
            if (it + 1 < n_tiles) {
                const int nb = base + kBc;
                q8_load(sk, kq + (size_t) nb * kQ8Row, min(kBc, n_cells - nb));
                q8_load(sv, vq + (size_t) nb * kQ8Row, min(kBc, n_cells - nb));
            }
        } else {
            if (it + 1 < n_tiles) {
                const int nb = base + kBc;
                load_tile(sm.k[st ^ 1], kh + (size_t) nb * kHD, min(kBc, n_cells - nb));
                load_tile(sm.v[st ^ 1], vh + (size_t) nb * kHD, min(kBc, n_cells - nb));
            }
            cp_commit();
            cp_wait<1>();
        }
        __syncthreads();
        // scores: n-tiles 0 and 1 of this warp's 16 cells
        float s[2][4];
#pragma unroll
        for (int n = 0; n < 2; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
        const __half* kt = sm.k[st];
        const int nc0 = hf * 16;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            unsigned b[4];
            ldsm4(b, kt + kv_at(nc0 + ((li >> 1) << 3) + lr, 2 * k + (li & 1)));
            mma16816(s[0], qa[k], b[0], b[1]);
            mma16816(s[1], qa[k], b[2], b[3]);
        }
        // the mask, the tile's row maxima (the four lanes of a row, then the m-tile's other warp)
        float xA = -INFINITY, xB = -INFINITY;
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int cell = base + nc0 + 8 * n + 2 * tig + e;
                s[n][e] = cell <= posA ? s[n][e] * scale : -INFINITY;
                s[n][2 + e] = cell <= posB ? s[n][2 + e] * scale : -INFINITY;
                xA = fmaxf(xA, s[n][e]);
                xB = fmaxf(xB, s[n][2 + e]);
            }
        xA = fmaxf(xA, __shfl_xor_sync(0xffffffffu, xA, 1));
        xA = fmaxf(xA, __shfl_xor_sync(0xffffffffu, xA, 2));
        xB = fmaxf(xB, __shfl_xor_sync(0xffffffffu, xB, 1));
        xB = fmaxf(xB, __shfl_xor_sync(0xffffffffu, xB, 2));
        if (tig == 0) {
            sm.red[0][hf][rA] = xA;
            sm.red[0][hf][rB] = xB;
        }
        __syncthreads();
        const float nA = fmaxf(mA, fmaxf(sm.red[0][0][rA], sm.red[0][1][rA]));
        const float nB = fmaxf(mB, fmaxf(sm.red[0][0][rB], sm.red[0][1][rB]));
        const float aA = nA == -INFINITY ? 1.f : expf(mA - nA), aB = nB == -INFINITY ? 1.f : expf(mB - nB);
        // the probabilities (FP16 into shared memory) and their row sums
        float sA = 0.f, sB = 0.f;
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            const float p0A = s[n][0] == -INFINITY ? 0.f : expf(s[n][0] - nA);
            const float p1A = s[n][1] == -INFINITY ? 0.f : expf(s[n][1] - nA);
            const float p0B = s[n][2] == -INFINITY ? 0.f : expf(s[n][2] - nB);
            const float p1B = s[n][3] == -INFINITY ? 0.f : expf(s[n][3] - nB);
            sA += p0A + p1A;
            sB += p0B + p1B;
            const int c = nc0 + 8 * n + 2 * tig;   // the pair's first cell in the tile
            *reinterpret_cast<unsigned*>(sm.p + p_at(rA, c >> 3) + (c & 7)) = h2(p0A, p1A);
            *reinterpret_cast<unsigned*>(sm.p + p_at(rB, c >> 3) + (c & 7)) = h2(p0B, p1B);
        }
        sA += __shfl_xor_sync(0xffffffffu, sA, 1);
        sA += __shfl_xor_sync(0xffffffffu, sA, 2);
        sB += __shfl_xor_sync(0xffffffffu, sB, 1);
        sB += __shfl_xor_sync(0xffffffffu, sB, 2);
        if (tig == 0) {
            sm.red[1][hf][rA] = sA;
            sm.red[1][hf][rB] = sB;
        }
        __syncthreads();
        lA = lA * aA + sm.red[1][0][rA] + sm.red[1][1][rA];
        lB = lB * aB + sm.red[1][0][rB] + sm.red[1][1][rB];
        mA = nA;
        mB = nB;
        // values: this warp's 128 dims of its m-tile's rows
#pragma unroll
        for (int n = 0; n < 16; ++n) {
            o[n][0] *= aA;
            o[n][1] *= aA;
            o[n][2] *= aB;
            o[n][3] *= aB;
        }
        const __half* vt = sm.v[st];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            unsigned a[4];
            const int row = mt * 16 + ((li & 1) << 3) + lr;
            ldsm4(a, sm.p + p_at(row, 2 * k + (li >> 1)));
#pragma unroll
            for (int n2 = 0; n2 < 8; ++n2) {
                unsigned b[4];
                ldsm4t(b, vt + kv_at(16 * k + ((li & 1) << 3) + lr, ((hf * 128 + n2 * 16) >> 3) + (li >> 1)));
                mma16816(o[2 * n2], a, b[0], b[1]);
                mma16816(o[2 * n2 + 1], a, b[2], b[3]);
            }
        }
        if constexpr (Q8) {   // the next tile, into the stage the last iteration read (all done since the barrier above)
            if (it + 1 < n_tiles) {
                q8_store(sm.k[st ^ 1], sk);
                q8_store(sm.v[st ^ 1], sv);
            }
        }
        __syncthreads();   // the stage's K, V and probabilities are read before they are written again
    }
    // the output: divided by the sum, times sigmoid(gate)
    const float iA = lA > 0.f ? 1.f / lA : 0.f, iB = lB > 0.f ? 1.f / lB : 0.f;
#pragma unroll
    for (int n = 0; n < 16; ++n) {
        const int d = hf * 128 + n * 8 + 2 * tig;
        if (okA) {
            const float2 g = *reinterpret_cast<const float2*>(gate + qA + d);
            *reinterpret_cast<float2*>(out + qA + d) =
                make_float2(o[n][0] * iA / (1.0f + expf(-g.x)), o[n][1] * iA / (1.0f + expf(-g.y)));
        }
        if (okB) {
            const float2 g = *reinterpret_cast<const float2*>(gate + qB + d);
            *reinterpret_cast<float2*>(out + qB + d) =
                make_float2(o[n][2] * iB / (1.0f + expf(-g.x)), o[n][3] * iB / (1.0f + expf(-g.y)));
        }
    }
}

}  // namespace q36a
