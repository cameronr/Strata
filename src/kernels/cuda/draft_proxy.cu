// src/kernels/cuda/draft_proxy.cu - see include/strata/kernels/draft_proxy.hpp.
//
// Built with --use_fast_math like native_mmvq.cu, whose Q6_K and Q8_0 x q8_1 dots the exact rescoring repeats (the
// same thread-to-block mapping, sums in the same order: bitwise its logits).
#include "strata/kernels/draft_proxy.hpp"
#include "strata/kernels/warp.cuh"
#include "strata/kernels/dp4a.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int WARP = 32;
constexpr int K = 2560;                    // a row's weights
constexpr int G = K / 32;                  // its 32-weight groups (one q8_1 block each)
constexpr int W4 = G / 2;                  // its uint4 of codes, two groups each
constexpr int SCORE_WARPS = 4;             // a tile's warps, a quarter of the row each
constexpr int W4_WARP = W4 / SCORE_WARPS;
constexpr int PER = 4;                     // candidates a tile of 32 rows keeps
constexpr float WINDOW = 8.0f;             // rescored: a copy score within this of the best one
constexpr int RS_BLOCKS = 64, RS_T = 256, RS_W = RS_T / WARP;   // the rescoring: a warp a row, a fixed 64-block grid
static_assert(W4 % SCORE_WARPS == 0 && RS_BLOCKS == 2 * WARP);   // the last block folds 2 * WARP = RS_BLOCKS shares

struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
struct Q80Block {
    half d;
    int8_t qs[32];
};
static_assert(sizeof(Q81Block) == 36 && sizeof(Q6KBlock) == 210 && sizeof(Q80Block) == 34);

int64_t tiles_of(int64_t n_rows) { return (n_rows + 31) / 32; }

// a float's order as an unsigned (0 below every float): the tiles' atomic max
__device__ __forceinline__ unsigned order_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float key_float(unsigned k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

__host__ __device__ constexpr size_t row_bytes_of(int type) {
    return type == 14 ? (size_t) (K / 256) * sizeof(Q6KBlock) : (size_t) G * sizeof(Q80Block);
}

// ---- the copy: codes [tile][w4][lane] uint4 (two groups: 2 x 64 bits of 2-bit codes), then scales [tile][w4][lane]
// uint32 (the two groups' halves).  A warp's 32 rows read 512 contiguous bytes.

// group g's 32 weights of a Q6_K row, in ggml's dequantize_row_q6_K order
__device__ void group_q6k(const uint8_t* row, int g, float* w) {
    const Q6KBlock* b = reinterpret_cast<const Q6KBlock*>(row) + g / 8;
    const int j = g % 8, h = j / 4, part = j % 4;
    const uint8_t* ql = b->ql + 64 * h;
    const uint8_t* qh = b->qh + 32 * h;
    const int8_t* sc = b->scales + 8 * h;
    const float d = __half2float(b->d);
    for (int l = 0; l < 32; ++l) {
        const int lo = (part & 1) ? ql[l + 32] : ql[l];
        const int q = ((part & 2) ? (lo >> 4) : (lo & 0xF)) | (((qh[l] >> (2 * part)) & 3) << 4);
        w[l] = d * (float) sc[l / 16 + 2 * part] * (float) (q - 32);
    }
}

__device__ void group_q80(const uint8_t* row, int g, float* w) {
    const Q80Block* b = reinterpret_cast<const Q80Block*>(row) + g;
    const float d = __half2float(b->d);
    for (int l = 0; l < 32; ++l) w[l] = d * (float) b->qs[l];
}

// One thread per (row, group): the group's weights at four levels (-1.5, -0.5, 0.5, 1.5) x a half scale, the scale
// (of six) that leaves the least squared error
template <int TYPE>
__global__ void pack_kernel(const uint8_t* __restrict__ rows, int64_t n, uint4* __restrict__ wq,
                            uint16_t* __restrict__ sc) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * G) return;
    const int64_t row = i / G;
    const int g = (int) (i % G);
    float w[32], amax = 1e-9f;
    if (TYPE == 14) group_q6k(rows + (size_t) row * row_bytes_of(14), g, w);
    else group_q80(rows + (size_t) row * row_bytes_of(8), g, w);
    for (int l = 0; l < 32; ++l) amax = fmaxf(amax, fabsf(w[l]));
    float best_s = amax / 1.5f, best_e = INFINITY;
    for (int t = 0; t < 6; ++t) {
        const float s = amax / 1.5f * (1.0f - 0.1f * (float) t);
        const float sh = __half2float(__float2half(s));
        float e = 0.0f;
        for (int l = 0; l < 32; ++l) {
            const int q = min(3, max(0, (int) floorf(w[l] / sh + 2.0f)));
            const float r = ((float) q - 1.5f) * sh - w[l];
            e += r * r;
        }
        if (e < best_e) { best_e = e; best_s = s; }
    }
    const __half hs = __float2half(best_s);
    const float sh = __half2float(hs);
    // word c of the group, byte b, bits 2i: weight 16c + 4i + b (the dot pairs byte b of a code word with int i's
    // byte b: weights 16c + 4i .. + 3 of the q8_1 block)
    uint32_t words[2] = {0u, 0u};
    for (int c = 0; c < 2; ++c)
        for (int ii = 0; ii < 4; ++ii)
            for (int bb = 0; bb < 4; ++bb) {
                const int q = min(3, max(0, (int) floorf(w[c * 16 + 4 * ii + bb] / sh + 2.0f)));
                words[c] |= (uint32_t) q << (8 * bb + 2 * ii);
            }
    const int64_t tile = row >> 5;
    const int lane = (int) (row & 31), w4 = g / 2, h = g & 1;
    const size_t at = ((size_t) tile * W4 + w4) * 32 + lane;
    uint32_t* wd = reinterpret_cast<uint32_t*>(wq + at) + 2 * h;
    wd[0] = words[0];
    wd[1] = words[1];
    sc[at * 2 + h] = __half_as_ushort(hs);
}

// One tile of 32 rows a block, four warps splitting the row, lane = row: the approximate scores; the tile's best PER
// (score, row), its softmax partial (max, sum of exp(score - max)) and its max into `gmax`; with `scores`, every row's
// score.
__global__ void __launch_bounds__(SCORE_WARPS * WARP) score_kernel(const uint4* __restrict__ wq,
                                                                   const uint32_t* __restrict__ sc,
                                                                   const Q81Block* __restrict__ x, int n_rows,
                                                                   float2* __restrict__ cand, float2* __restrict__ part,
                                                                   unsigned* __restrict__ gmax,
                                                                   float* __restrict__ scores) {
    __shared__ __align__(16) int xi[G * 8];
    __shared__ float xs[G], xdd[G];
    __shared__ float red[SCORE_WARPS][WARP];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, tile = blockIdx.x;
    const size_t base = ((size_t) tile * W4 + warp * W4_WARP) * 32 + lane;
    uint4 w[W4_WARP];
    uint32_t s[W4_WARP];
#pragma unroll
    for (int j = 0; j < W4_WARP; ++j) {   // read once: streamed
        w[j] = __ldcs(wq + base + (size_t) j * 32);
        s[j] = __ldcs(sc + base + (size_t) j * 32);
    }
    for (int i = tid; i < G * 8; i += SCORE_WARPS * WARP) xi[i] = reinterpret_cast<const int*>(x[i / 8].qs)[i % 8];
    if (tid < G) {
        const int* q = reinterpret_cast<const int*>(x[tid].qs);
        int sum = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) sum = __dp4a(0x01010101, q[i], sum);
        const float d8 = __low2float(x[tid].ds);
        xdd[tid] = d8;
        xs[tid] = d8 * (float) sum;
    }
    __syncthreads();
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < W4_WARP; ++j) {
        const int w4 = warp * W4_WARP + j;
        const uint32_t ww[4] = {w[j].x, w[j].y, w[j].z, w[j].w};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int g = 2 * w4 + h;
            int P = 0;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const uint32_t v = ww[2 * h + c];
                const int4 x0 = *reinterpret_cast<const int4*>(&xi[g * 8 + c * 4]);
                P = __dp4a((int) (v & 0x03030303u), x0.x, P);
                P = __dp4a((int) ((v >> 2) & 0x03030303u), x0.y, P);
                P = __dp4a((int) ((v >> 4) & 0x03030303u), x0.z, P);
                P = __dp4a((int) ((v >> 6) & 0x03030303u), x0.w, P);
            }
            const float sg = __half2float(__ushort_as_half((unsigned short) (h ? (s[j] >> 16) : (s[j] & 0xffff))));
            acc += sg * (xdd[g] * (float) P - 1.5f * xs[g]);
        }
    }
    red[warp][lane] = acc;
    __syncthreads();
    if (warp != 0) return;
    const float ninf = __int_as_float(0xff800000);
    const int row = tile * 32 + lane;
    float v = red[0][lane] + red[1][lane] + red[2][lane] + red[3][lane];
    if (row >= n_rows) v = ninf;
    else if (scores != nullptr) scores[row] = v;
    if (cand == nullptr) return;
    float m = v;
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float e = v > ninf ? __expf(v - m) : 0.0f;
#pragma unroll
    for (int o = 16; o; o >>= 1) e += __shfl_xor_sync(0xffffffffu, e, o);
    if (lane == 0) {
        part[tile] = make_float2(m, e);
        atomicMax(gmax, order_key(m));
    }
    for (int k = 0; k < PER; ++k) {   // the best left, the lowest row on a tie
        float bv = v;
        int bi = lane;
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) cand[tile * PER + k] = make_float2(bv, __int_as_float(tile * 32 + bi));
        if (lane == bi) v = ninf;
    }
}

// ---- the exact rows: native_mmvq's single-column dots, one warp standing for its row's four warps (lane L for
// threads L, L + 32, L + 64, L + 96), each virtual thread's blocks in its order, the four added in the cross-warp
// order before the same xor tree

// native_mmvq's load_int_b2: the blocks are two-byte aligned
__device__ __forceinline__ int load_int_b2(const void* ptr, int i32) {
    const uint16_t* x16 = static_cast<const uint16_t*>(ptr);
    return (int) x16[2 * i32] | ((int) x16[2 * i32 + 1] << 16);
}

// Q6_K (native_q6_k_mmvq_kernel, not small-K: thread t's blocks t / 32 + 4k, lane t % 32 of each); the activations
// as {int, its block's scale} in shared memory
__device__ __forceinline__ float q6_dot(const Q6KBlock* __restrict__ b, const int2* __restrict__ x, int kby, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = load_int_b2(b->ql, iqs);
    const int vh = load_int_b2(b->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    const int8_t* scales = b->scales + scale_offset;
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int2 e = x[(kby + bq8_offset + 2 * i) * 8 + iqs % 8];
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = strata_q6_quants(vil, vih);
        sumf += __int_as_float(e.y) * (__dp4a(vi, e.x, 0) * sc);
    }
    return __half2float(b->d) * sumf;
}

__device__ float q6k_row(const Q6KBlock* __restrict__ w, const int2* __restrict__ x, int lane) {
    float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int kbx = 0; kbx < K / 256; ++kbx) p[kbx % 4] += q6_dot(w + kbx, x, kbx * 8, lane);
    float t = p[0];
#pragma unroll
    for (int v = 1; v < 4; ++v) t += p[v];
    return warp_sum(t);
}

// Q8_0 (native_small_mmvq_kernel, Qi 8, not small-K: thread t's blocks t / 4 + 32k, ints 2 (t % 4) and the next)
__device__ __forceinline__ float q80_dot(const Q80Block* __restrict__ w, const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
        sumi = __dp4a(v, u, sumi);
    }
    const float d0 = w->d;
    const float d1 = __low2float(x->ds);
    return d0 * d1 * float(sumi);
}

__device__ float q80_row(const Q80Block* __restrict__ w, const Q81Block* __restrict__ x, int lane) {
    const int kqs = 2 * (lane % 4);
    float p[4];
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        float tmp = 0.0f;
        for (int kbx = 8 * v + lane / 4; kbx < G; kbx += 32) tmp += q80_dot(w + kbx, x + kbx, kqs);
        p[v] = tmp;
    }
    float t = p[0];
#pragma unroll
    for (int v = 1; v < 4; ++v) t += p[v];
    return warp_sum(t);
}

// Every candidate within WINDOW of the best copy score m (the tiles' atomic max), a warp each across the GPU.  Each
// block sums its share of the tiles' softmax partials relative to m, and of the rescored rows' swaps (the exact term
// for the copy's), and keeps its best exact row (the lowest on a tie); the last block to finish combines the blocks in
// order: the pick, and its probability 1 / (sum * exp(m - best)).  Fixed shares and orders, so the result does not
// depend on timing.  It leaves `gmax` and `ticket` zero for the next launch.
template <int TYPE>
__global__ void __launch_bounds__(RS_T) rescore_kernel(const uint8_t* __restrict__ rows, const Q81Block* __restrict__ x,
                                                       int n_tiles, const int32_t* __restrict__ map,
                                                       const float2* __restrict__ cand,
                                                       const float2* __restrict__ part, float4* __restrict__ blk,
                                                       unsigned* __restrict__ gmax, unsigned* __restrict__ ticket,
                                                       int32_t* __restrict__ out_id, float* __restrict__ out_prob) {
    __shared__ __align__(16) int2 xe[G * 8];
    __shared__ float wsum[RS_W], wbv[RS_W];
    __shared__ int wbi[RS_W];
    __shared__ bool last;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (TYPE == 14)
        for (int i = tid; i < G * 8; i += RS_T) {
            const Q81Block* b = x + i / 8;
            xe[i] = make_int2(reinterpret_cast<const int*>(b->qs)[i % 8], __float_as_int(__low2float(b->ds)));
        }
    const float ninf = __int_as_float(0xff800000);
    const float m = key_float(*gmax), lo = m - WINDOW;
    float sum = 0.0f;   // this thread's tiles, then (lane 0) its warp's swaps
    for (int t = blockIdx.x + gridDim.x * tid; t < n_tiles; t += gridDim.x * RS_T) {
        const float2 p = part[t];
        if (p.y > 0.0f) sum += p.y * __expf(p.x - m);
    }
    __syncthreads();   // xe
    float bv = ninf;
    int bi = 0x7fffffff;
    const int nc = n_tiles * PER;
    // the candidates a warp each, grid-stride: any number, in a fixed order (each read from L2 as it comes: loaded
    // ahead, 12 a warp, they took 1-2 us longer a launch)
#pragma unroll 1
    for (int slot = blockIdx.x * RS_W + warp; slot < nc; slot += gridDim.x * RS_W) {
        const float2 c = cand[slot];
        if (!(c.x > ninf && c.x >= lo)) continue;
        const int row = __float_as_int(c.y);
        float y;
        if (TYPE == 14) y = q6k_row(reinterpret_cast<const Q6KBlock*>(rows) + (size_t) row * (K / 256), xe, lane);
        else y = q80_row(reinterpret_cast<const Q80Block*>(rows) + (size_t) row * G, x, lane);
        if (lane == 0) {
            sum += __expf(y - m) - __expf(c.x - m);
            if (y > bv || (y == bv && row < bi)) { bv = y; bi = row; }
        }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) { wsum[warp] = sum; wbv[warp] = bv; wbi[warp] = bi; }
    __syncthreads();
    if (tid == 0) {
        float s = wsum[0], v = wbv[0];
        int i0 = wbi[0];
        for (int i = 1; i < RS_W; ++i) {
            s += wsum[i];
            if (wbv[i] > v || (wbv[i] == v && wbi[i] < i0)) { v = wbv[i]; i0 = wbi[i]; }
        }
        blk[blockIdx.x] = make_float4(s, v, __int_as_float(i0), 0.0f);
        __threadfence();
        last = atomicAdd(ticket, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last || warp != 0) return;
    __threadfence();
    const float4 qa = __ldcg(blk + lane), qb = __ldcg(blk + lane + WARP);
    float s = qa.x + qb.x, v = qa.y;
    int i0 = __float_as_int(qa.z);
    if (qb.y > v || (qb.y == v && __float_as_int(qb.z) < i0)) { v = qb.y; i0 = __float_as_int(qb.z); }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, o);
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, i0, o);
        if (ov > v || (ov == v && oi < i0)) { v = ov; i0 = oi; }
    }
    if (lane == 0) {
        *out_id = map[i0 < 0x7fffffff ? i0 : 0];
        *out_prob = 1.0f / fmaxf(s * __expf(m - v), 1.0f);   // the pick's own term is 1
        *gmax = 0u;
        *ticket = 0u;
    }
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

bool draft_proxy_supported(int ggml_type, int n_in) noexcept {
    return n_in == K && (ggml_type == 14 || ggml_type == 8);
}

size_t draft_proxy_bytes(int64_t n_rows) { return (size_t) tiles_of(n_rows) * W4 * 32 * (16 + 4); }

size_t draft_proxy_scratch_bytes(int64_t n_rows) {
    return ((size_t) tiles_of(n_rows) * (PER + 1) * sizeof(float2) + 15) / 16 * 16 + RS_BLOCKS * sizeof(float4) +
           2 * sizeof(unsigned);
}

void draft_proxy_pack(int ggml_type, const void* rows, int64_t n_rows, void* proxy, void* stream) {
    if (!draft_proxy_supported(ggml_type, K) || n_rows <= 0)
        throw std::invalid_argument("draft_proxy_pack: unsupported");
    const int64_t tiles = tiles_of(n_rows);
    cudaMemsetAsync(proxy, 0, draft_proxy_bytes(n_rows), (cudaStream_t) stream);
    uint4* wq = (uint4*) proxy;
    uint16_t* sc = (uint16_t*) ((char*) proxy + (size_t) tiles * W4 * 32 * 16);
    const unsigned blocks = (unsigned) ((n_rows * G + 127) / 128);
    if (ggml_type == 14)
        pack_kernel<14><<<blocks, 128, 0, (cudaStream_t) stream>>>((const uint8_t*) rows, n_rows, wq, sc);
    else
        pack_kernel<8><<<blocks, 128, 0, (cudaStream_t) stream>>>((const uint8_t*) rows, n_rows, wq, sc);
    check("draft_proxy_pack");
}

void draft_proxy_top1(int ggml_type, const void* proxy, const void* rows, const void* x_q8_1, int64_t n_rows,
                      const int32_t* map, int32_t* out_id, float* out_prob, void* scratch, void* stream) {
    const int64_t tiles = tiles_of(n_rows);
    if (n_rows <= 0 || n_rows > INT32_MAX)   // the kernels' row ids are ints
        throw std::invalid_argument("draft_proxy_top1: n_rows out of range");
    const uint4* wq = (const uint4*) proxy;
    const uint32_t* sc = (const uint32_t*) ((const char*) proxy + (size_t) tiles * W4 * 32 * 16);
    float2* cand = (float2*) scratch;
    float2* part = cand + (size_t) tiles * PER;
    float4* blk = (float4*) ((char*) scratch + ((size_t) tiles * (PER + 1) * sizeof(float2) + 15) / 16 * 16);
    unsigned* gmax = (unsigned*) (blk + RS_BLOCKS);
    const auto s = (cudaStream_t) stream;
    const auto* x = (const Q81Block*) x_q8_1;
    score_kernel<<<(unsigned) tiles, SCORE_WARPS * WARP, 0, s>>>(wq, sc, x, (int) n_rows, cand, part, gmax, nullptr);
    if (ggml_type == 14)
        rescore_kernel<14><<<RS_BLOCKS, RS_T, 0, s>>>((const uint8_t*) rows, x, (int) tiles, map, cand, part, blk, gmax,
                                                      gmax + 1, out_id, out_prob);
    else
        rescore_kernel<8><<<RS_BLOCKS, RS_T, 0, s>>>((const uint8_t*) rows, x, (int) tiles, map, cand, part, blk, gmax,
                                                     gmax + 1, out_id, out_prob);
    check("draft_proxy_top1");
}

}  // namespace strata::kernels
