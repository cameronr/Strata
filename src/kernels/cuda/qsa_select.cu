// src/kernels/cuda/qsa_select.cu - see include/strata/kernels/qsa_select.hpp.
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <type_traits>
#include <unordered_map>

// src/core/verify.cpp's batch windows: see include/strata/kernels/seg_rows.hpp for the segment table contract.
#include "strata/kernels/seg_rows.hpp"

namespace strata::kernels {
namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr unsigned FULL = 0xffffffffu;
constexpr int SCORE_WARPS = 8;
constexpr int SCORE_KB = 2;                                // key blocks per 8-lane group
constexpr int SCORE_TILE = SCORE_WARPS * 4 * SCORE_KB;     // key blocks per thread block and round
constexpr int SCORE_QC = 16;                               // queries staged in shared memory at a time (32 KB)
constexpr int BATCH_T = 256, BATCH_W = BATCH_T / 32;      // batch_scores_kernel: one block an SM
constexpr int BATCH_QT = 2, BATCH_BT = 4;                  // a lane: 2 queries x 4 key blocks, all heads
constexpr int BATCH_TILE = 32 * BATCH_BT;                  // key blocks in shared memory (64 KB)
constexpr int BATCH_QC = BATCH_W * BATCH_QT;               // queries a unit (a warp's slots: 32 KB)
constexpr int QD = IDX_HEADS * IDX_DIM;                    // a query's floats
constexpr int kBatchMinQueries = 12;                       // fewer (decode windows): block_scores_kernel
constexpr int TOPK_T = 1024, TOPK_W = TOPK_T / 32;
constexpr int TOPK_ROWS = 32;                              // rows of TOPK_T blocks held in registers
constexpr int TOPK_TAB_ROWS = 64;                          // the (row, warp) tables' least rows: in the histograms' room

__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// One leaf of a score (dims 4l..4l+3 of a head): k.x*q.x + k.y*q.y + k.z*q.z + k.w*q.w as the original warp-per-pair
// kernel compiled it (qsa_select_parity keeps it)
__device__ __forceinline__ float dot4(float4 k, float4 q) {
    return __fmaf_rn(k.w, q.w, __fmaf_rn(k.z, q.z, __fmaf_rn(k.x, q.x, __fmul_rn(k.y, q.y))));
}

// One query's scores for the group's SCORE_KB key blocks.  Lane j of the 8-lane group holds each key's dims 4l..4l+3
// for l = j, j+8, j+16, j+24; per head its four 4-term dot products d_l are summed in the lane as
// (d_j + d_j+16) + (d_j+8 + d_j+24), then across the group by xor 4, 2 and 1: the tree in which a warp-wide xor
// butterfly (16, 8, 4, 2, 1) over lanes l sums the same d_l, so each score is bit for bit that of one warp per
// (query, block).  TAIL: key `tail_kb` (-1: none) is the query's tail block and takes the `dead` key instead.
template <bool TAIL>
__device__ __forceinline__ void group_scores(const float4 (&k)[SCORE_KB][4], const float* __restrict__ dead,
                                             int tail_kb, const float* q, int j, float (&score)[SCORE_KB]) {
#pragma unroll
    for (int kb = 0; kb < SCORE_KB; ++kb) score[kb] = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        float4 q4[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) q4[i] = *reinterpret_cast<const float4*>(q + h * IDX_DIM + 4 * (j + 8 * i));
#pragma unroll
        for (int kb = 0; kb < SCORE_KB; ++kb) {
            float d[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float4 k4 = k[kb][i];
                if (TAIL && kb == tail_kb) k4 = *reinterpret_cast<const float4*>(dead + 4 * (j + 8 * i));
                d[i] = dot4(k4, q4[i]);
            }
            float c = (d[0] + d[2]) + (d[1] + d[3]);
            c += __shfl_xor_sync(FULL, c, 4);
            c += __shfl_xor_sync(FULL, c, 2);
            c += __shfl_xor_sync(FULL, c, 1);
            score[kb] += c > 0.0f ? c : 0.0f;
        }
    }
}

// 8-lane groups of SCORE_KB consecutive key blocks each, read once for all the queries, which go through shared
// memory SCORE_QC at a time.  Block n_bid of a query is its incomplete tail: the `dead` key, +1e9 when it has cells.
// The body is shared with the batch form (block_scores_seg_kernel): one launch whose rows are queries of different
// slots, each row program-identical to a per-row nq=1 call of this kernel.
__device__ __forceinline__ void block_scores_body(const float* __restrict__ pooled, const float* __restrict__ dead,
                                                  const float* __restrict__ q_idx, const int32_t* __restrict__ steps,
                                                  int nq, int64_t max_blocks, float* __restrict__ out) {
    __shared__ __align__(16) float qs[SCORE_QC * IDX_HEADS * IDX_DIM];
    __shared__ int s_nkv[SCORE_QC], s_nbid[SCORE_QC];
    const int t = threadIdx.x, lane = t & 31, j = lane & 7;
    const int slot = ((t >> 5) * 4 + (lane >> 3)) * SCORE_KB;   // the group's first block within a tile
    int64_t top = -1;   // the queries' last block: the rounds end there
    for (int qi = 0; qi < nq; ++qi) {
        const int64_t n_bid = steps[(int64_t) qi * kStepCount + kStepNBid];
        const int64_t last = n_bid < max_blocks - 1 ? n_bid : max_blocks - 1;
        top = last > top ? last : top;
    }
    for (int64_t b0 = (int64_t) blockIdx.x * SCORE_TILE; b0 <= top; b0 += (int64_t) gridDim.x * SCORE_TILE) {
        const int64_t bg = b0 + slot;
        float4 k[SCORE_KB][4];
#pragma unroll
        for (int kb = 0; kb < SCORE_KB; ++kb) {
            const float* key = pooled + (bg + kb <= top ? bg + kb : 0) * IDX_DIM;
#pragma unroll
            for (int i = 0; i < 4; ++i) k[kb][i] = __ldg(reinterpret_cast<const float4*>(key + 4 * (j + 8 * i)));
        }
        for (int q0 = 0; q0 < nq; q0 += SCORE_QC) {
            const int qn = nq - q0 < SCORE_QC ? nq - q0 : SCORE_QC;
            __syncthreads();
            const float4* src = reinterpret_cast<const float4*>(q_idx + (int64_t) q0 * IDX_HEADS * IDX_DIM);
            for (int i = t; i < qn * IDX_HEADS * IDX_DIM / 4; i += SCORE_WARPS * 32)
                reinterpret_cast<float4*>(qs)[i] = src[i];
            if (t < qn) {
                s_nkv[t] = steps[(int64_t) (q0 + t) * kStepCount + kStepNKv];
                s_nbid[t] = steps[(int64_t) (q0 + t) * kStepCount + kStepNBid];
            }
            __syncthreads();
            for (int qq = 0; qq < qn; ++qq) {
                const int64_t n_kv = s_nkv[qq], n_bid = s_nbid[qq];
                const int64_t last = n_bid < max_blocks - 1 ? n_bid : max_blocks - 1;
                const int tail_kb = n_bid >= bg && n_bid < bg + SCORE_KB ? (int) (n_bid - bg) : -1;
                const float* q = qs + qq * IDX_HEADS * IDX_DIM;
                float score[SCORE_KB];
                // warp-uniform branches: every lane of the warp runs the group shuffles
                if (__any_sync(FULL, tail_kb >= 0)) group_scores<true>(k, dead, tail_kb, q, j, score);
                else group_scores<false>(k, dead, -1, q, j, score);
                if (j == 0) {
                    float* o = out + (int64_t) (q0 + qq) * max_blocks;
#pragma unroll
                    for (int kb = 0; kb < SCORE_KB; ++kb) {
                        if (bg + kb > last) break;
                        o[bg + kb] = kb == tail_kb && n_kv % R != 0 ? score[kb] + 1e9f : score[kb];
                    }
                }
            }
        }
    }
}

__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_kernel(const float* __restrict__ pooled,
                                                                        const float* __restrict__ dead,
                                                                        const float* __restrict__ q_idx,
                                                                        const int32_t* __restrict__ steps, int nq,
                                                                        int64_t max_blocks, float* __restrict__ out) {
    block_scores_body(pooled, dead, q_idx, steps, nq, max_blocks, out);
}

// The batch window's form (qsa_block_scores_seg): row blockIdx.y is one query of its own slot, whose keys the
// resident segment table names. One row's program IS the nq=1 per-row call's - same body, same grid.x, and the
// per-row launcher always picks block_scores_kernel at nq=1, so no cross-kernel parity is relied on. The table
// entry is one dependent global load per block; rows never index a parameter array (that spills to local memory).
__global__ void __launch_bounds__(SCORE_WARPS * 32)
block_scores_seg_kernel(const SegRow* __restrict__ seg, const int32_t* __restrict__ row_slot,
                        const float* __restrict__ q_idx, const int32_t* __restrict__ steps, int64_t max_blocks,
                        float* __restrict__ out) {
    const unsigned r = blockIdx.y;
    const SegRow row = seg[row_slot[r]];
    block_scores_body(row.idx_pooled, row.idx_dead, q_idx + (int64_t) r * QD, steps + (int64_t) r * kStepCount, 1,
                      max_blocks, out + (int64_t) r * max_blocks);
}

// The decode windows' scores with the queries in registers: a warp a key block, lane l holding leaf l (dims
// 4l..4l+3) of the key and of each of its queries' heads, REG_UB consecutive blocks a warp at a time (their loads
// issued first).  A block's V = 4 x QG values (QG queries x 4 heads) are summed over the lanes by recursive halving:
// at offset o (16, 8, ...) a lane keeps half of its values and adds its partner's copy of them, the butterfly's
// addition for each value at that offset, then the plain butterfly at the offsets left (upstream's #2072 scorer), so
// every score is bitwise the warp form's and block_scores_kernel's.  Lane l ends with value l >> (5 - log2 V); a
// query's head-0 lane gathers heads 1-3 and adds their relus in order.  Two groups of 4 warps for 5-8 queries, each
// over the same blocks (the second group's key reads mostly hit L1).  2-8 queries: block_scores_kernel read the queries
// from shared memory every block, its limit; one query is bound by the keys' reads either way (and 1.03x as long here).
template <int QG>
__global__ void __launch_bounds__(SCORE_WARPS * 32, 2) block_scores_reg_kernel(const float* __restrict__ pooled,
                                                                               const float* __restrict__ dead,
                                                                               const float* __restrict__ q_idx,
                                                                               const int32_t* __restrict__ steps,
                                                                               int nq, int64_t max_blocks,
                                                                               float* __restrict__ out) {
    constexpr int V = 4 * QG, LV = QG == 2 ? 3 : 4, SH = 5 - LV;
    constexpr int REG_UB = 4;                          // key blocks a warp a round (8: slower)
    static_assert(QG == 2 || QG == 4, "QG");
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, mb = (int) max_blocks - 1;
    const bool two = nq > QG;                          // 5-8 queries: two groups of 4 warps
    const int wpg = two ? SCORE_WARPS / 2 : SCORE_WARPS, wi = two ? warp & 3 : warp, q0 = two ? (warp >> 2) * QG : 0;
    float4 qr[QG][IDX_HEADS];
    int n_bid[QG];
    int top = -1;                                      // the group's last block
#pragma unroll
    for (int a = 0; a < QG; ++a) {
        const bool in = q0 + a < nq;
        n_bid[a] = in ? steps[(int64_t) (q0 + a) * kStepCount + kStepNBid] : -1;
        top = max(top, min(n_bid[a], mb));
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h)
            qr[a][h] = in ? *reinterpret_cast<const float4*>(q_idx + (int64_t) (q0 + a) * QD + h * IDX_DIM + 4 * lane)
                          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    // lane l ends with value j = l >> SH = a * 4 + h; query a's head-0 lane writes its scores
    const int jw = lane >> SH, aw = jw >> 2;
    const bool writer = (jw & 3) == 0 && (lane & ((1 << SH) - 1)) == 0 && q0 + aw < nq;
    int w_bid = -1, w_last = -1;
    bool w_tail = false;
    float* w_out = out;
    if (writer) {
        w_bid = steps[(int64_t) (q0 + aw) * kStepCount + kStepNBid];
        w_last = min(w_bid, mb);
        w_tail = steps[(int64_t) (q0 + aw) * kStepCount + kStepNKv] % R != 0;
        w_out = out + (int64_t) (q0 + aw) * max_blocks;
    }
    const float4 kd = *reinterpret_cast<const float4*>(dead + 4 * lane);
    const int tile = wpg * REG_UB;
    for (int b0 = (int) blockIdx.x * tile + wi * REG_UB; b0 <= top; b0 += (int) gridDim.x * tile) {
        float4 kp[REG_UB];
#pragma unroll
        for (int u = 0; u < REG_UB; ++u)
            kp[u] = __ldg(reinterpret_cast<const float4*>(pooled + (int64_t) (b0 + u <= top ? b0 + u : b0) * IDX_DIM +
                                                          4 * lane));
#pragma unroll
        for (int u = 0; u < REG_UB; ++u) {
            const int b = b0 + u;
            if (b > top) break;                        // warp-uniform
            float v[V];
#pragma unroll
            for (int a = 0; a < QG; ++a) {
                const float4 k4 = b == n_bid[a] ? kd : kp[u];
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) v[a * IDX_HEADS + h] = dot4(k4, qr[a][h]);
            }
            int cnt = V;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                if (cnt > 1) {
                    const int half = cnt >> 1;
                    const bool hi = (lane & o) != 0;
#pragma unroll
                    for (int i = 0; i < V / 2; ++i) {
                        if (i < half) {
                            const float send = hi ? v[i] : v[i + half], keep = hi ? v[i + half] : v[i];
                            v[i] = keep + __shfl_xor_sync(FULL, send, o);
                        }
                    }
                    cnt = half;
                } else {
                    v[0] += __shfl_xor_sync(FULL, v[0], o);
                }
            }
            const float d1 = __shfl_sync(FULL, v[0], (lane + (1 << SH)) & 31);
            const float d2 = __shfl_sync(FULL, v[0], (lane + (2 << SH)) & 31);
            const float d3 = __shfl_sync(FULL, v[0], (lane + (3 << SH)) & 31);
            if (writer && b <= w_last) {
                const float d[IDX_HEADS] = {v[0], d1, d2, d3};
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) score += d[h] > 0.0f ? d[h] : 0.0f;
                w_out[b] = b == w_bid && w_tail ? score + 1e9f : score;
            }
        }
    }
}

__host__ __device__ constexpr int rev5(int i) {
    return ((i & 1) << 4) | ((i & 2) << 2) | (i & 4) | ((i & 8) >> 2) | ((i & 16) >> 4);
}
__host__ __device__ constexpr int popc5(int i) {
    return (i & 1) + ((i >> 1) & 1) + ((i >> 2) & 1) + ((i >> 3) & 1) + ((i >> 4) & 1);
}
// key rows in shared memory, 16-byte chunk c of row r at c ^ (r & 7): a quarter warp's rows on distinct bank quads
__device__ __forceinline__ int key_at(int r, int c) { return r * IDX_DIM + 4 * (c ^ (r & 7)); }
__device__ __forceinline__ void cp_async16(void* s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}

// The same scores for batches of queries, in a per-thread GEMM form.  Lane l of a warp scores key blocks l + 32 j
// (j < BATCH_BT) of the tile against the warp's BATCH_QT queries, all heads: a dot4 per leaf (dims 4l..4l+3), the 32
// leaves summed on a stack in the tree of the warp form's butterfly (bit-reversed order: leaves l and l + 16 first),
// so each score is bit for bit the warp form's.  The warp reads its queries' values all at once, a shared-memory
// broadcast at about half the cost of distinct reads.  Persistent and balanced: block g takes units
// [g U / G, (g + 1) U / G) of the U = tiles x chunks of BATCH_QC queries, tile-major.  The tile's keys are loaded
// between barriers when the tile changes; each warp stages its queries into a slot of its own, the next unit's even
// leaves once this unit is done with them, then the odd ones, so the copies run while it computes.  A query's tail
// block scores the `dead` key in the warp form.
__global__ void __launch_bounds__(BATCH_T, 1) batch_scores_kernel(const float* __restrict__ pooled,
                                                                  const float* __restrict__ dead,
                                                                  const float* __restrict__ q_idx,
                                                                  const int32_t* __restrict__ steps, int nq,
                                                                  int64_t max_blocks, float* __restrict__ out) {
    extern __shared__ __align__(16) float smem[];
    __shared__ int s_top;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float* ks = smem;                                          // BATCH_TILE x IDX_DIM
    float* qs = smem + BATCH_TILE * IDX_DIM + warp * BATCH_QT * QD;
    {
        int top = -1;                                          // the batch's last block
        for (int i = t; i < nq; i += BATCH_T) {
            const int n_bid = steps[(int64_t) i * kStepCount + kStepNBid];
            top = max(top, (int64_t) n_bid < max_blocks - 1 ? n_bid : (int) (max_blocks - 1));
        }
        top = __reduce_max_sync(FULL, top);
        if (t == 0) s_top = -1;
        __syncthreads();
        if (lane == 0) atomicMax(&s_top, top);
        __syncthreads();
    }
    const int top = s_top;
    if (top < 0) return;
    const int n_chunks = (nq + BATCH_QC - 1) / BATCH_QC, units = (top / BATCH_TILE + 1) * n_chunks;
    const int u1 = (int) ((int64_t) (blockIdx.x + 1) * units / gridDim.x);
    // leaves of parity `half` of queries q0, q0 + 1 into the warp's slot, one cp.async group
    auto stage = [&](int q0, int half) {
#pragma unroll
        for (int e = lane; e < BATCH_QT * IDX_HEADS * 16; e += 32) {
            const int a = e / (IDX_HEADS * 16), off = (e / 16) % IDX_HEADS * IDX_DIM + 4 * (2 * (e % 16) + half);
            if (q0 + a < nq) cp_async16(qs + a * QD + off, q_idx + (int64_t) (q0 + a) * QD + off);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    int have = -1;                                             // the first query of the slot's rows
    for (int u = (int) ((int64_t) blockIdx.x * units / gridDim.x); u < u1;) {
        const int b0 = u / n_chunks * BATCH_TILE, ue = min(u1, (u / n_chunks + 1) * n_chunks);
        __syncthreads();                                       // every warp is done with the previous tile
        for (int e = t; e < BATCH_TILE * 32; e += BATCH_T) {
            const int r = e >> 5, c = e & 31;
            cp_async16(ks + key_at(r, c), pooled + (int64_t) min(b0 + r, top) * IDX_DIM + 4 * c);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        for (; u < ue; ++u) {
            const int q0 = u % n_chunks * BATCH_QC + warp * BATCH_QT;
            int n_bid[BATCH_QT], n_kv[BATCH_QT], wlast = -1;
#pragma unroll
            for (int a = 0; a < BATCH_QT; ++a) {
                const bool in = q0 + a < nq;
                n_bid[a] = in ? steps[(int64_t) (q0 + a) * kStepCount + kStepNBid] : -1;
                n_kv[a] = in ? steps[(int64_t) (q0 + a) * kStepCount + kStepNKv] : 0;
                wlast = max(wlast, n_bid[a]);
            }
            if (wlast < b0) continue;                          // the warp's queries end before the tile
            if (have != q0) {
                // a skipped unit's prefetch may still be landing in the slot: copies to one address are unordered
                asm volatile("cp.async.wait_group 0;\n" ::);
                __syncwarp();
                stage(q0, 0);
                stage(q0, 1);
                have = q0;
            }
            const int q0n = u + 1 < u1 ? (u + 1) % n_chunks * BATCH_QC + warp * BATCH_QT : -1;
            const bool ahead = q0n >= 0 && q0n != q0;
            asm volatile("cp.async.wait_group 1;\n" ::);       // the even leaves
            __syncwarp();
            float stk[BATCH_QT][IDX_HEADS][BATCH_BT][6];
            auto leaves = [&](auto first) {
#pragma unroll
                for (int ii = 0; ii < 16; ++ii) {
                    const int i = decltype(first)::value + ii, l = rev5(i), depth = popc5(i);
                    float4 kf[BATCH_BT], qf[BATCH_QT][IDX_HEADS];
#pragma unroll
                    for (int j = 0; j < BATCH_BT; ++j)
                        kf[j] = *reinterpret_cast<const float4*>(ks + key_at(lane + 32 * j, l));
#pragma unroll
                    for (int a = 0; a < BATCH_QT; ++a)
#pragma unroll
                        for (int h = 0; h < IDX_HEADS; ++h)
                            qf[a][h] = *reinterpret_cast<const float4*>(qs + a * QD + h * IDX_DIM + 4 * l);
#pragma unroll
                    for (int a = 0; a < BATCH_QT; ++a)
#pragma unroll
                        for (int h = 0; h < IDX_HEADS; ++h)
#pragma unroll
                            for (int j = 0; j < BATCH_BT; ++j) {
                                float* s = stk[a][h][j];
                                s[depth] = dot4(kf[j], qf[a][h]);
#pragma unroll
                                for (int lv = 0; lv < 5; ++lv) {   // complete subtrees combine
                                    if (((i + 1) >> lv) & 1) break;
                                    s[depth - lv - 1] = s[depth - lv - 1] + s[depth - lv];
                                }
                            }
                }
            };
            leaves(std::integral_constant<int, 0>());           // rev5(0..15): the even leaves
            __syncwarp();
            if (ahead) {
                stage(q0n, 0);
                asm volatile("cp.async.wait_group 1;\n" ::);   // this unit's odd leaves
            } else {
                asm volatile("cp.async.wait_group 0;\n" ::);
            }
            __syncwarp();
            leaves(std::integral_constant<int, 16>());
            __syncwarp();
            if (ahead) {
                stage(q0n, 1);
                have = q0n;
            }
#pragma unroll
            for (int a = 0; a < BATCH_QT; ++a) {
                const int qi = q0 + a;
                const int64_t last = (int64_t) n_bid[a] < max_blocks - 1 ? n_bid[a] : max_blocks - 1;
#pragma unroll
                for (int j = 0; j < BATCH_BT; ++j) {
                    float score = 0.0f;
#pragma unroll
                    for (int h = 0; h < IDX_HEADS; ++h) {
                        const float d = stk[a][h][j][0];
                        score += d > 0.0f ? d : 0.0f;
                    }
                    const int b = b0 + lane + 32 * j;
                    if (qi < nq && b <= last && b != n_bid[a]) out[(int64_t) qi * max_blocks + b] = score;
                }
                if (qi < nq && n_bid[a] >= b0 && n_bid[a] < b0 + BATCH_TILE && (int64_t) n_bid[a] <= max_blocks - 1) {
                    const float4 k4 = *reinterpret_cast<const float4*>(dead + 4 * lane);
                    float score = 0.0f;
#pragma unroll
                    for (int h = 0; h < IDX_HEADS; ++h) {
                        float d = dot4(k4, *reinterpret_cast<const float4*>(q_idx + (int64_t) qi * QD + h * IDX_DIM +
                                                                            4 * lane));
#pragma unroll
                        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(FULL, d, o);
                        score += d > 0.0f ? d : 0.0f;
                    }
                    if (lane == 0) out[(int64_t) qi * max_blocks + n_bid[a]] = n_kv[a] % R != 0 ? score + 1e9f : score;
                }
            }
        }
    }
}

__device__ __forceinline__ int warp_sum(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

// The exclusive prefixes of a and b over the block's threads in thread order, and their totals.
__device__ __forceinline__ void block_exclusive2(int a, int b, int& ea, int& eb, int& ta, int& tb, int* s_w) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int ia = a, ib = b;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int ya = __shfl_up_sync(FULL, ia, o), yb = __shfl_up_sync(FULL, ib, o);
        if (lane >= o) { ia += ya; ib += yb; }
    }
    if (lane == 31) { s_w[warp] = ia; s_w[TOPK_W + warp] = ib; }
    __syncthreads();
    if (warp == 0) {
        const int wa = s_w[lane], wb = s_w[TOPK_W + lane];
        int xa = wa, xb = wb;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int ya = __shfl_up_sync(FULL, xa, o), yb = __shfl_up_sync(FULL, xb, o);
            if (lane >= o) { xa += ya; xb += yb; }
        }
        s_w[lane] = xa - wa;
        s_w[TOPK_W + lane] = xb - wb;
        if (lane == 31) { s_w[2 * TOPK_W] = xa; s_w[2 * TOPK_W + 1] = xb; }
    }
    __syncthreads();
    ea = s_w[warp] + ia - a;
    eb = s_w[TOPK_W + warp] + ib - b;
    ta = s_w[2 * TOPK_W];
    tb = s_w[2 * TOPK_W + 1];
    __syncthreads();
}

// The same selection as the per-cell topk_kernel: the `width` cells of the largest block scores (a block's score
// counts for each of its cells), ties to the lowest cell, emitted ascending.  A radix select, 8 bits at a time, over
// the order keys, with a histogram per warp.  The complete blocks go in rows of TOPK_T: thread t has block
// row * TOPK_T + t, the first TOPK_ROWS rows in registers (read once, 131K cells), the others from memory in each
// pass.  A block's cells then land at (cells above the threshold before it) + min(cells at it before it, budget):
// ballots within a (row, warp), one scan over the (row, warp) totals.  The tail block (n_bid) is thread 0's.
// Shared memory (topk_smem_bytes): the per-warp histograms, then in the same room the (row, warp) tables of `tab_rows`
// rows and the output; past TOPK_TAB_ROWS rows the tables outgrow the histograms.
__global__ void __launch_bounds__(TOPK_T) block_topk_kernel(const float* __restrict__ scores,
                                                            const int32_t* __restrict__ steps, int64_t max_blocks,
                                                            int64_t cap, int tab_rows, int32_t* __restrict__ ids) {
    extern __shared__ int sm[];                        // the per-warp histograms; then the scan table and the output
    __shared__ int hist[256];
    __shared__ int s_w[2 * TOPK_W + 2];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int tail_w = (int) (n_kv - n_bid * R);       // the cells of block n_bid, 0..3
    const int nrows = (int) ((n_bid + TOPK_T - 1) / TOPK_T);
    uint32_t key[TOPK_ROWS];
#pragma unroll
    for (int i = 0; i < TOPK_ROWS; ++i) {
        const int64_t b = (int64_t) i * TOPK_T + t;
        key[i] = b < n_bid ? order_key(sc[b]) : 0u;
    }
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width
    uint32_t prefix = 0;
    int above = 0;                                     // cells strictly above the digits fixed so far
    int* whist = sm + warp * 256;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32) whist[i] = 0;
        __syncwarp();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        int cd = -1, cc = 0;                           // a run of one digit is added at once
        auto count = [&](uint32_t k) {
            if ((k & hi_mask) != (prefix & hi_mask)) return;
            const int d = (int) ((k >> shift) & 255);
            if (d != cd) {
                if (cc != 0) atomicAdd(&whist[cd], cc);
                cd = d;
                cc = 0;
            }
            cc += R;
        };
#pragma unroll
        for (int i = 0; i < TOPK_ROWS; ++i)
            if ((int64_t) i * TOPK_T + t < n_bid) count(key[i]);
        for (int i = TOPK_ROWS; i < nrows; ++i) {
            const int64_t b = (int64_t) i * TOPK_T + t;
            if (b < n_bid) count(order_key(sc[b]));
        }
        if (cc != 0) atomicAdd(&whist[cd], cc);
        if (t == 0 && tail_w > 0) {
            const uint32_t k = order_key(sc[n_bid]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&whist[(k >> shift) & 255], tail_w);
        }
        __syncthreads();
        if (t < 256) {
            int s = 0;
#pragma unroll 8
            for (int w = 0; w < TOPK_W; ++w) s += sm[w * 256 + t];
            hist[t] = s;
        }
        __syncthreads();
        if (warp == 0) {
            // the digit: the largest d >= 1 with above + (cells with a digit >= d) >= width, else 0; lane l holds the
            // bins 255 - 8l down to 248 - 8l
            int v[8], s = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) { v[i] = hist[255 - (lane * 8 + i)]; s += v[i]; }
            int incl = s;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up_sync(FULL, incl, o);
                if (lane >= o) incl += y;
            }
            int run = incl - s, below = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                run += v[i];
                if (255 - (lane * 8 + i) >= 1 && above + run < width) ++below;
            }
            const int digit = 255 - warp_sum(below);
            int gt = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i)
                if (255 - (lane * 8 + i) > digit) gt += v[i];
            gt = warp_sum(gt);
            if (lane == 0) {
                s_digit = digit;
                s_above = above + gt;
            }
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
    }
    const uint32_t thr = prefix;
    const int budget = (int) (width - above);          // cells equal to thr that fit, lowest index first
    // ---- the cells above and at thr per (row, warp), their exclusive prefixes in block order
    int* tab_gt = sm;
    int* tab_eq = sm + tab_rows * 32;
    int* obuf = sm + 2 * tab_rows * 32;                // the selection, copied out coalesced
    __syncthreads();                                   // the histograms are no longer read
    auto row_key = [&](int i) { return i < TOPK_ROWS ? 0u : order_key(sc[(int64_t) i * TOPK_T + t]); };
#pragma unroll
    for (int i = 0; i < TOPK_ROWS; ++i) {
        if (i >= nrows) break;
        const bool in = (int64_t) i * TOPK_T + t < n_bid;
        const unsigned mg = __ballot_sync(FULL, in && key[i] > thr), me = __ballot_sync(FULL, in && key[i] == thr);
        if (lane == 0) { tab_gt[i * 32 + warp] = R * __popc(mg); tab_eq[i * 32 + warp] = R * __popc(me); }
    }
    for (int i = TOPK_ROWS; i < nrows; ++i) {
        const bool in = (int64_t) i * TOPK_T + t < n_bid;
        const uint32_t k = in ? row_key(i) : 0u;
        const unsigned mg = __ballot_sync(FULL, in && k > thr), me = __ballot_sync(FULL, in && k == thr);
        if (lane == 0) { tab_gt[i * 32 + warp] = R * __popc(mg); tab_eq[i * 32 + warp] = R * __popc(me); }
    }
    __syncthreads();
    int gt_all, eq_all;
    {
        // entries (row, warp) in block order, `per` consecutive ones a thread: two up to TOPK_TAB_ROWS rows
        const int e = nrows * 32, per = e > 2 * TOPK_T ? (e + TOPK_T - 1) / TOPK_T : 2, k0 = per * t;
        const int k1 = k0 + per < e ? k0 + per : e;
        int gs = 0, es = 0;
        for (int k = k0; k < k1; ++k) { gs += tab_gt[k]; es += tab_eq[k]; }
        int gx, ex;
        block_exclusive2(gs, es, gx, ex, gt_all, eq_all, s_w);
        for (int k = k0; k < k1; ++k) {
            const int g = tab_gt[k], q = tab_eq[k];
            tab_gt[k] = gx;
            tab_eq[k] = ex;
            gx += g;
            ex += q;
        }
    }
    __syncthreads();
    // ---- the cells, into shared memory
    const unsigned lt = (1u << lane) - 1u;
    auto emit = [&](int i, uint32_t k, bool in) {
        const bool g = in && k > thr, q = in && k == thr;
        const unsigned mg = __ballot_sync(FULL, g), me = __ballot_sync(FULL, q);
        if (!g && !q) return;
        const int gb = tab_gt[i * 32 + warp] + R * __popc(mg & lt);
        const int eb = tab_eq[i * 32 + warp] + R * __popc(me & lt);
        const int pos = gb + (eb < budget ? eb : budget);
        const int take = g ? R : (budget - eb < 0 ? 0 : (budget - eb > R ? R : budget - eb));
        const int64_t b = (int64_t) i * TOPK_T + t;
        for (int c = 0; c < take; ++c) obuf[pos + c] = (int32_t) (b * R + c);
    };
#pragma unroll
    for (int i = 0; i < TOPK_ROWS; ++i) {
        if (i >= nrows) break;
        emit(i, key[i], (int64_t) i * TOPK_T + t < n_bid);
    }
    for (int i = TOPK_ROWS; i < nrows; ++i) {
        const bool in = (int64_t) i * TOPK_T + t < n_bid;
        emit(i, in ? row_key(i) : 0u, in);
    }
    if (t == 0 && tail_w > 0) {                        // the tail block comes last
        const uint32_t k = order_key(sc[n_bid]);
        const int pos = gt_all + (eq_all < budget ? eq_all : budget);
        const int take = k > thr ? tail_w : (k == thr ? (budget - eq_all < 0 ? 0 : (budget - eq_all > tail_w ? tail_w
                                                                                        : budget - eq_all)) : 0);
        for (int c = 0; c < take; ++c) obuf[pos + c] = (int32_t) (n_bid * R + c);
    }
    __syncthreads();
    for (int64_t j = t; j < width; j += TOPK_T) out[j] = obuf[j];
}

// block_topk_kernel's selection for a decode window's few queries in one launch over many thread blocks: a query's
// blocks split into A contiguous chunks of at least MT_CHUNK (A from its own length, so a short context takes one
// thread block and no barrier), each chunk's thread block making the radix passes over its blocks, adding its
// histogram to the query's (global) and meeting the query's other thread blocks at a barrier before each digit, which
// every one takes from the whole histogram with block_topk_kernel's rule: the same threshold, `above` and cells.  Each
// chunk's cells above / at the threshold, one more barrier, and a chunk emits its cells at the counts of the chunks
// before it.  The grid covers the capacity (a captured graph's), its blocks past a query's A return at once; at most
// as many as can be resident at once take part (the barriers wait for all of a query's).  The window graphs run one
// selection at a time on a GPU: a query's barrier state lives in device memory, which its last thread block out
// clears.  (Upstream's #2072 split top-k took six launches.)
constexpr int MT_T = 256, MT_W = MT_T / 32;
constexpr int MT_ROWS = 8;                         // rows of MT_T blocks a thread holds in registers
constexpr int MT_CHUNK = MT_ROWS * MT_T;           // blocks a thread block takes at least
constexpr int MT_MAXQ = 8, MT_MAXCH = 32;          // queries a launch; thread blocks a query
static_assert(MT_T == 256, "a thread a histogram bin");
struct TopkSync {
    int hist[4][256];
    int cnt[MT_MAXCH][2];                          // a chunk's cells above the threshold, at it
    unsigned arrived, left;
};
__device__ TopkSync g_topk_sync[MT_MAXQ];

__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
// until the query's thread blocks have arrived `target` times in all
__device__ __forceinline__ void query_barrier(unsigned* arrived, unsigned target) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(arrived, 1u);
        while (ld_acquire(arrived) < target) __nanosleep(32);
    }
    __syncthreads();
}

__global__ void __launch_bounds__(MT_T) block_topk_multi_kernel(const float* __restrict__ scores,
                                                                const int32_t* __restrict__ steps, int64_t max_blocks,
                                                                int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int whist[MT_W][256];
    __shared__ int hist[256];
    __shared__ int s_g[MT_W], s_e[MT_W];
    __shared__ int s_digit, s_above, s_gt0, s_eq0, s_last;
    const int q = blockIdx.y, c = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int32_t* st = steps + (int64_t) q * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + (int64_t) q * cap;
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        for (int64_t j = (int64_t) c * MT_T + t; j < n_kv; j += (int64_t) gridDim.x * MT_T) out[j] = (int32_t) j;
        return;
    }
    const int64_t nb = n_bid + 1;                      // with the tail block
    const int A = (int) min((int64_t) gridDim.x, (nb + MT_CHUNK - 1) / MT_CHUNK);
    if (c >= A) return;
    TopkSync& sy = g_topk_sync[q];
    const int64_t b0 = nb * c / A, b1 = nb * (c + 1) / A, e1 = b1 < n_bid ? b1 : n_bid;   // complete blocks [b0, e1)
    const int tail_w = c == A - 1 ? (int) (n_kv - n_bid * R) : 0;                         // block n_bid's cells
    const int nrows = (int) ((e1 - b0 + MT_T - 1) / MT_T);
    const float* sc = scores + (int64_t) q * max_blocks;
    uint32_t key[MT_ROWS];
#pragma unroll
    for (int i = 0; i < MT_ROWS; ++i) {
        const int64_t b = b0 + (int64_t) i * MT_T + t;
        key[i] = b < e1 ? order_key(sc[b]) : 0u;
    }
    const uint32_t tkey = tail_w > 0 ? order_key(sc[n_bid]) : 0u;
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width
    uint32_t prefix = 0;
    int above = 0;
    for (int p = 0; p < 4; ++p) {
        const int shift = 24 - 8 * p;
        const uint32_t hi_mask = p == 0 ? 0u : (0xffffffffu << (shift + 8));
        for (int i = lane; i < 256; i += 32) whist[warp][i] = 0;
        __syncwarp();
        int cd = -1, cc = 0;                           // a run of one digit is added at once
        auto count = [&](uint32_t k) {
            if ((k & hi_mask) != (prefix & hi_mask)) return;
            const int d = (int) ((k >> shift) & 255);
            if (d != cd) {
                if (cc != 0) atomicAdd(&whist[warp][cd], cc);
                cd = d;
                cc = 0;
            }
            cc += R;
        };
#pragma unroll
        for (int i = 0; i < MT_ROWS; ++i)
            if (b0 + (int64_t) i * MT_T + t < e1) count(key[i]);
        for (int i = MT_ROWS; i < nrows; ++i) {
            const int64_t b = b0 + (int64_t) i * MT_T + t;
            if (b < e1) count(order_key(sc[b]));
        }
        if (cc != 0) atomicAdd(&whist[warp][cd], cc);
        if (t == 0 && tail_w > 0 && (tkey & hi_mask) == (prefix & hi_mask))
            atomicAdd(&whist[0][(tkey >> shift) & 255], tail_w);
        __syncthreads();
        int s = 0;
#pragma unroll
        for (int w = 0; w < MT_W; ++w) s += whist[w][t];
        if (A > 1) {
            if (s != 0) atomicAdd(&sy.hist[p][t], s);
            query_barrier(&sy.arrived, (unsigned) A * (unsigned) (p + 1));
            s = __ldcg(&sy.hist[p][t]);
        }
        hist[t] = s;
        __syncthreads();
        if (warp == 0) {
            // block_topk_kernel's digit: the largest d >= 1 with above + (cells with a digit >= d) >= width, else 0;
            // lane l holds the bins 255 - 8l down to 248 - 8l
            int v[8], sum = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) { v[i] = hist[255 - (lane * 8 + i)]; sum += v[i]; }
            int incl = sum;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up_sync(FULL, incl, o);
                if (lane >= o) incl += y;
            }
            int run = incl - sum, below = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                run += v[i];
                if (255 - (lane * 8 + i) >= 1 && above + run < width) ++below;
            }
            const int digit = 255 - warp_sum(below);
            int gt = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i)
                if (255 - (lane * 8 + i) > digit) gt += v[i];
            gt = warp_sum(gt);
            if (lane == 0) {
                s_digit = digit;
                s_above = above + gt;
            }
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
    }
    const uint32_t thr = prefix;
    const int budget = (int) (width - above);          // cells equal to thr that fit, lowest index first
    // ---- the chunk's cells above and at thr; the chunks' before it
    int gt = 0, eq = 0;
#pragma unroll
    for (int i = 0; i < MT_ROWS; ++i)
        if (b0 + (int64_t) i * MT_T + t < e1) { gt += key[i] > thr ? R : 0; eq += key[i] == thr ? R : 0; }
    for (int i = MT_ROWS; i < nrows; ++i) {
        const int64_t b = b0 + (int64_t) i * MT_T + t;
        if (b < e1) {
            const uint32_t k = order_key(sc[b]);
            gt += k > thr ? R : 0;
            eq += k == thr ? R : 0;
        }
    }
    gt = warp_sum(gt);
    eq = warp_sum(eq);
    if (lane == 0) { s_g[warp] = gt; s_e[warp] = eq; }
    __syncthreads();
    if (A > 1) {
        if (t == 0) {
            int g = 0, e = 0;
            for (int w = 0; w < MT_W; ++w) { g += s_g[w]; e += s_e[w]; }
            if (tail_w > 0) { g += tkey > thr ? tail_w : 0; e += tkey == thr ? tail_w : 0; }
            sy.cnt[c][0] = g;
            sy.cnt[c][1] = e;
        }
        query_barrier(&sy.arrived, (unsigned) A * 5u);
        if (t == 0) {
            int g0 = 0, e0 = 0;
            for (int k = 0; k < c; ++k) { g0 += __ldcg(&sy.cnt[k][0]); e0 += __ldcg(&sy.cnt[k][1]); }
            s_gt0 = g0;
            s_eq0 = e0;
            s_last = atomicAdd(&sy.left, 1u) == (unsigned) A - 1;   // every thread block is past the last barrier
        }
        __syncthreads();
        if (s_last) {                                  // the state back to zero for the next call
            for (int i = t; i < 4 * 256; i += MT_T) (&sy.hist[0][0])[i] = 0;
            if (t == 0) { sy.arrived = 0; sy.left = 0; }
        }
    } else if (t == 0) {
        s_gt0 = 0;
        s_eq0 = 0;
    }
    __syncthreads();
    // ---- the cells: a row of MT_T blocks at a time, positions in block order
    int g_run = s_gt0, e_run = s_eq0;
    const unsigned lt = (1u << lane) - 1u;
    for (int i = 0; i < nrows; ++i) {
        const int64_t b = b0 + (int64_t) i * MT_T + t;
        const bool in = b < e1;
        uint32_t k = 0;
#pragma unroll
        for (int r = 0; r < MT_ROWS; ++r)
            if (r == i) k = key[r];
        if (i >= MT_ROWS && in) k = order_key(sc[b]);
        const bool g = in && k > thr, e = in && k == thr;
        const unsigned mg = __ballot_sync(FULL, g), me = __ballot_sync(FULL, e);
        if (lane == 0) { s_g[warp] = R * __popc(mg); s_e[warp] = R * __popc(me); }
        __syncthreads();
        int gb = g_run, eb = e_run, rg = 0, re = 0;
#pragma unroll
        for (int w = 0; w < MT_W; ++w) {
            if (w < warp) { gb += s_g[w]; eb += s_e[w]; }
            rg += s_g[w];
            re += s_e[w];
        }
        __syncthreads();
        if (g || e) {
            gb += R * __popc(mg & lt);
            eb += R * __popc(me & lt);
            const int pos = gb + (eb < budget ? eb : budget);
            const int take = g ? R : (budget - eb < 0 ? 0 : (budget - eb > R ? R : budget - eb));
            for (int j = 0; j < take; ++j) out[pos + j] = (int32_t) (b * R + j);
        }
        g_run += rg;
        e_run += re;
    }
    if (t == 0 && tail_w > 0) {                        // the tail block comes last
        const int pos = g_run + (e_run < budget ? e_run : budget);
        const int take = tkey > thr ? tail_w : (tkey == thr ? (budget - e_run < 0 ? 0 : (budget - e_run > tail_w ? tail_w
                                                                                     : budget - e_run)) : 0);
        for (int j = 0; j < take; ++j) out[pos + j] = (int32_t) (n_bid * R + j);
    }
}

}  // namespace

// the block_scores_kernel grid.x: the covered blocks in tiles of SCORE_TILE. One source for the per-row launcher
// and the batch seg launcher, so a batch row's blocks tile the same key blocks in the same rounds.
static int64_t score_grid_x(int64_t grid_blocks, int64_t max_blocks) {
    const int64_t covered = grid_blocks < max_blocks ? grid_blocks : max_blocks;
    return (covered + SCORE_TILE - 1) / SCORE_TILE;
}

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t grid_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535 || grid_blocks < 1) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    const int64_t covered = grid_blocks < max_blocks ? grid_blocks : max_blocks;
    if (nq >= kBatchMinQueries) {
        // one block an SM (the prompt path's GPU), each with the same share of the units
        constexpr int smem = (BATCH_TILE * IDX_DIM + BATCH_QC * QD) * (int) sizeof(float);
        static std::mutex mu;
        static std::unordered_map<int, int> sms_of;   // per device (a split prompt path runs this on two)
        int dev = 0, sms = 0;
        cudaGetDevice(&dev);
        {
            const std::lock_guard<std::mutex> lock(mu);
            int& n = sms_of[dev];
            if (n == 0) {
                cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
                cudaFuncSetAttribute(batch_scores_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            }
            sms = n;
        }
        const int64_t units = ((covered + BATCH_TILE - 1) / BATCH_TILE) * ((nq + BATCH_QC - 1) / BATCH_QC);
        const unsigned grid = (unsigned) (units < sms ? units : sms);
        batch_scores_kernel<<<grid, BATCH_T, smem, (cudaStream_t) stream>>>(pooled, dead, q_idx, steps, (int) nq,
                                                                            max_blocks, scores);
    } else {
        const unsigned grid = (unsigned) score_grid_x(grid_blocks, max_blocks);
        const cudaStream_t st = (cudaStream_t) stream;
        if (nq == 2) block_scores_reg_kernel<2><<<grid, SCORE_WARPS * 32, 0, st>>>(pooled, dead, q_idx, steps, 2, max_blocks, scores);
        else if (nq >= 3 && nq <= 8)
            block_scores_reg_kernel<4><<<grid, SCORE_WARPS * 32, 0, st>>>(pooled, dead, q_idx, steps, (int) nq, max_blocks,
                                                                         scores);
        else block_scores_kernel<<<grid, SCORE_WARPS * 32, 0, st>>>(pooled, dead, q_idx, steps, (int) nq, max_blocks, scores);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

void qsa_block_scores_seg(const SegRow* seg, const int32_t* row_slot, const float* q_idx, const int32_t* steps,
                          int64_t n, int64_t max_blocks, const QsaShapes& s, float* scores, void* stream,
                          int64_t grid_blocks) {
    if (n <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || n > 65535 || grid_blocks < 1) {
        std::fprintf(stderr, "qsa_block_scores_seg: unsupported indexer geometry\n");
        std::exit(1);
    }
    // no decline path: the per-row nq=1 call always runs block_scores_kernel, which is exactly what the seg
    // kernel is. n <= kVerifyMaxT (8) by the verifier's window cap; the rows are validated by run_slot_rows.
    const dim3 grid((unsigned) score_grid_x(grid_blocks, max_blocks), (unsigned) n);
    block_scores_seg_kernel<<<grid, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(seg, row_slot, q_idx, steps,
                                                                                   max_blocks, scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores_seg: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

namespace {

// The rows the tables need for a capacity: the complete blocks, at most max_blocks - 2 (the callers' max_cells / 4 + 2
// leaves room for the tail block and one more), at least TOPK_TAB_ROWS
int64_t topk_tab_rows(int64_t max_blocks) {
    const int64_t rows = (max_blocks - 2 + TOPK_T - 1) / TOPK_T;
    return rows > TOPK_TAB_ROWS ? rows : TOPK_TAB_ROWS;
}

size_t topk_smem_bytes(int64_t tab_rows, int64_t cap) {
    const int64_t tabs = 2 * tab_rows * 32 + cap, hists = (int64_t) TOPK_W * 256;
    return (size_t) (tabs > hists ? tabs : hists) * sizeof(int);
}

// The dynamic shared memory a block of the kernel can have on the current device
size_t topk_smem_limit() {
    int dev = 0, optin = 0;
    cudaFuncAttributes fa{};
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess ||
        cudaFuncGetAttributes(&fa, block_topk_kernel) != cudaSuccess)
        return 0;
    return (size_t) optin > fa.sharedSizeBytes ? (size_t) optin - fa.sharedSizeBytes : 0;
}

}  // namespace

int64_t qsa_select_max_cells(const QsaShapes& s) {
    const int64_t ints = (int64_t) (topk_smem_limit() / sizeof(int)) - qsa_selection_width(kTopkMaxCells, s);
    return ints > 0 ? ints / 64 * TOPK_T * R : 0;
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    const int64_t tab_rows = topk_tab_rows(max_blocks);
    const size_t smem = topk_smem_bytes(tab_rows, cap);
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s) || max_blocks < 2) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    if (nq <= MT_MAXQ) {
        // the chunks a query: from the capacity, at most what stays resident with the other queries' (the barriers)
        static std::mutex mu;
        static std::unordered_map<int, int> resident_of;
        int dev = 0, resident = 0;
        cudaGetDevice(&dev);
        {
            const std::lock_guard<std::mutex> lock(mu);
            int& r = resident_of[dev];
            if (r == 0) {
                int sms = 0, per_sm = 0;
                cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, block_topk_multi_kernel, MT_T, 0);
                r = sms * per_sm;
            }
            resident = r;
        }
        const int64_t want = (max_blocks - 1 + MT_CHUNK - 1) / MT_CHUNK;
        const int64_t nch = std::max<int64_t>(1, std::min<int64_t>({want, (int64_t) MT_MAXCH, resident / nq}));
        block_topk_multi_kernel<<<dim3((unsigned) nch, (unsigned) nq), MT_T, 0, (cudaStream_t) stream>>>(
            scores, steps, max_blocks, cap, ids);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e)); std::exit(1); }
        return;
    }
    if (smem > 48 * 1024) {
        // once a device for the largest size asked: a function attribute, not a stream operation (capturable)
        static std::mutex mu;
        static std::unordered_map<int, size_t> set;
        int dev = 0;
        cudaGetDevice(&dev);
        const std::lock_guard<std::mutex> lock(mu);
        if (set[dev] < smem) {
            if (smem > topk_smem_limit() ||
                cudaFuncSetAttribute(block_topk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem) !=
                    cudaSuccess) {
                std::fprintf(stderr, "qsa_block_topk: %lld blocks are past this GPU's selection capacity\n",
                             (long long) max_blocks);
                std::exit(1);
            }
            set[dev] = smem;
        }
    }
    block_topk_kernel<<<(unsigned) nq, TOPK_T, smem, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap,
                                                                             (int) tab_rows, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace strata::kernels
