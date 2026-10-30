// src/kernels/cuda/s2_expert_grouped.cu - R4's grouped GPU expert.  Read the header first.
//
// THE ARITHMETIC IS `src/kernels/cpu/expert.cpp`'s, over the same bytes, one warp per output row.
//
//     sum_j (c_j - 1) * d_w * xhat_j * d_x   ==   d_w * d_x * ( sum_j c_j*xhat_j  -  sum_j xhat_j )
//
// Both sums are INT8 x INT8, so both are `__dp4a` - four multiply-accumulates per instruction, exact in
// integer and with no dequantization inside the loop.  The weight scale `d_w` is fp16 (one per 64 elements)
// and the activation scale `d_x` is fp16 (one per 32 elements), so the float work is one multiply-add per
// 32-element chunk rather than one per element.
//
// **THE SUMMATION ORDER IS NOT THE CPU's AND CANNOT BE.**  Lane `L` takes chunks `L, L+32, ...` and the
// partials are reduced through shuffle; the CPU walks every chunk in order with its own accumulator shape.
// The two agree to float rounding and not to the bit, which is the same contract `s_gemv_parity` carries for
// the same reason.  `bench/micro/moe_hit_parity.cu` is the check.
#include "lamina/kernels/s2_expert_grouped.hpp"
#include "lamina/kernels/warp.cuh"
#include "lamina/core/geometry.hpp"
#include "lamina/kernels/dp4a.hpp"
#include "lamina/kernels/f16_bits.hpp"

#include "lamina/kernels/quantize_act.hpp"
#include "lamina/kernels/verify_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace lamina::kernels {
namespace {

// THE BLOB'S OWN GEOMETRY, from `include/lamina/kernels/cpu/expert.hpp`.  Restated as literals because that
// header is the CPU path's and this file must not silently follow it if the two ever disagree: the sizes below
// are what the CPU kernel's indexing computes, and `moe_hit_parity` compares the two end to end.
constexpr int H = 2560;
constexpr int FF = 640;
static_assert(H == lamina::core::kCompiledGeometry.n_embd && FF == lamina::core::kCompiledGeometry.n_ff, "s2_expert_grouped is compiled for core/geometry.hpp");
constexpr int QK = 64;                       // Q2_0's group: one fp16 scale per 64 weights
constexpr int ROW_GU = H / 4;                // 640 B of codes per gate/up row (2 bits per element)
constexpr int ROW_D = FF / 4;                // 160 B per down row
constexpr int SC_GU = H / QK;                // 40 fp16 scales per gate/up row
constexpr int SC_D = FF / QK;                // 10 per down row
constexpr size_t O_D_CODES = (size_t) 2 * FF * ROW_GU;
constexpr size_t O_GU_SCALES = O_D_CODES + (size_t) H * ROW_D;
constexpr size_t O_D_SCALES = O_GU_SCALES + (size_t) 2 * FF * SC_GU * 2;

__device__ __forceinline__ float f16_at(const uint8_t* p) {
    return __half2float(__ushort_as_half((uint16_t) (p[0] | (p[1] << 8))));
}

/// `silu(gate) * up`, in place, over a GATE-MAJOR buffer: `[0, n_pairs)` is every hit's gate and
/// `[n_pairs, 2*n_pairs)` is every hit's up, so hit `h`'s row `r` meets itself at `h*FF + r`.
///
/// SiLU on the GATE and multiplied by up - the reading `docs/semantics.md` records, and the one that is wrong
/// the other way round in a way that still produces a finite number.
__global__ void swiglu_kernel(float* __restrict__ gate_up, long long n_pairs) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pairs) return;
    const float g = gate_up[i];
    const float u = gate_up[n_pairs + i];
    gate_up[i] = (g / (1.0f + __expf(-g))) * u;
}

__global__ void swiglu_quantize_q8_0_scaled_kernel(float* __restrict__ gate_up, long long n_pairs,
                                                   uint8_t* __restrict__ blocks, float* __restrict__ scales) {
    const long long n_blocks = n_pairs >> 5;
    const long long b = ((long long) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (b >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const long long idx = b * 32 + lane;
    const float g = gate_up[idx];
    const float u = gate_up[n_pairs + idx];
    const float xv = (g / (1.0f + __expf(-g))) * u;
    gate_up[idx] = xv;
    uint8_t* out = blocks + b * 34;

    float amax = fabsf(xv);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float s = amax > 0.f ? amax / 127.f : 0.f;
    const float inv = s > 0.f ? 1.f / s : 0.f;
    if (lane == 0) {
        scales[b] = s;
        const uint16_t d16bits = f16_from_f32(s);
        out[0] = (uint8_t) (d16bits & 0xFF);
        out[1] = (uint8_t) (d16bits >> 8);
    }
    const float t = xv * inv;
    const float r = t + (t >= 0.f ? 0.5f : -0.5f);
    int v = (int) r;
    v = v < -127 ? -127 : (v > 127 ? 127 : v);
    out[2 + lane] = (uint8_t) (int8_t) v;
}

// ---- THE SAME INTEGERS FROM FEWER INSTRUCTIONS.  The kernels above stay for activations that are not word-aligned.
//
// **INSIDE A CHUNK, WHICH CODE MEETS WHICH ACTIVATION IN A `dp4a` WORD IS FREE.**  `s = sum_e code_e * x_e` and
// `hx = sum_e x_e` are exact integer sums (|s| <= 32 * 3 * 128), so any grouping of the 32 products into words
// gives the same two integers.  The grouping used here costs no transposition of the codes: `(w >> 2f) & 0x03030303`
// holds field `f` of each of a code word's four bytes - elements `4b + f`, plus 16 for the second word - so it is
// the ACTIVATIONS that are regrouped, once per chunk, into
//
//     X[4h + f] = { x[16h + f], x[16h + 4 + f], x[16h + 8 + f], x[16h + 12 + f] }
//
// a 4x4 byte transpose of each half of the chunk, eight `__byte_perm`.  The kernels above spend seven integer
// operations per code BYTE and a second `dp4a` chain for `hx` on every row; here a code word costs seven for 16
// elements and `hx` is summed once per chunk.  `s`, `hx` and the float expression `dw * dx * (float) (s - hx)` are
// unchanged, so every output is bitwise the previous kernels' - `s2_expert_grouped_parity` checks exactly that.

// fp16 at an even address, one 16-bit load (`f16_at` reads it as two bytes); same value.
__device__ __forceinline__ float f16_ld(const uint8_t* p) {
    return __half2float(__ushort_as_half(*(const unsigned short*) p));
}

// A chunk's two code words as the eight `dp4a` operands that pair with `X[0..7]`: `m[4h + f]` byte `b` is the
// code of element `16h + 4b + f`.
__device__ __forceinline__ void expand_codes(uint2 cb, int m[8]) {
    const unsigned M = 0x03030303u;
    m[0] = (int) (cb.x & M);
    m[1] = (int) ((cb.x >> 2) & M);
    m[2] = (int) ((cb.x >> 4) & M);
    m[3] = (int) ((cb.x >> 6) & M);
    m[4] = (int) (cb.y & M);
    m[5] = (int) ((cb.y >> 2) & M);
    m[6] = (int) ((cb.y >> 4) & M);
    m[7] = (int) ((cb.y >> 6) & M);
}

// The 32 int8 of the `block_q8_0` at `xb` regrouped into `X[0..7]` (see above), and their sum `hx`.
//
// **NINE ALIGNED WORD LOADS INSTEAD OF 32 BYTE LOADS.**  The caller guarantees a 4-byte aligned activation row, so
// chunk `c`'s int8 start at `34c + 2`, i.e. 0 bytes (odd `c`) or 2 bytes (even `c`) into an aligned word; the
// words are read aligned and funnel-shifted into place.  The ninth word is read only in the 2-byte case, and then
// it lies inside block `c + 1` of the same row: both row lengths (80 and 20 chunks) are even, so an even chunk is
// never a row's last.
__device__ __forceinline__ int load_x_chunk(const uint8_t* __restrict__ xb, int X[8]) {
    const uint8_t* q = xb + 2;
    const int off = (int) ((uintptr_t) q & 3);
    const unsigned* p = (const unsigned*) (q - off);
    const unsigned sh = (unsigned) off * 8;
    unsigned v[9];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = __ldg(p + j);
    v[8] = off != 0 ? __ldg(p + 8) : 0u;
    unsigned n[8];
    int hx = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        // natural word j: x[4j .. 4j+3]; a 64-bit shift (sh is 0, 8, 16 or 24) rather than __funnelshift_r, so the same
        // source needs no CUDA-only intrinsic.
        n[j] = (unsigned) ((((unsigned long long) v[j + 1] << 32) | v[j]) >> sh);
        hx = LAMINA_DP4A(0x01010101, (int) n[j], hx);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const unsigned t0 = __byte_perm(n[4 * h], n[4 * h + 1], 0x5140);      // a0 b0 a1 b1
        const unsigned t1 = __byte_perm(n[4 * h], n[4 * h + 1], 0x7362);      // a2 b2 a3 b3
        const unsigned t2 = __byte_perm(n[4 * h + 2], n[4 * h + 3], 0x5140);  // c0 d0 c1 d1
        const unsigned t3 = __byte_perm(n[4 * h + 2], n[4 * h + 3], 0x7362);  // c2 d2 c3 d3
        X[4 * h + 0] = (int) __byte_perm(t0, t2, 0x5410);                    // a0 b0 c0 d0
        X[4 * h + 1] = (int) __byte_perm(t0, t2, 0x7632);                    // a1 b1 c1 d1
        X[4 * h + 2] = (int) __byte_perm(t1, t3, 0x5410);                    // a2 b2 c2 d2
        X[4 * h + 3] = (int) __byte_perm(t1, t3, 0x7632);                    // a3 b3 c3 d3
    }
    return hx;
}

// `s` for one chunk: the eight expanded code words against the eight regrouped activation words.
__device__ __forceinline__ int chunk_s(const int m[8], const int X[8]) {
    int s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) s = LAMINA_DP4A(m[j], X[j], s);
    return s;
}

constexpr int THREADS = 256;

void check(const char* who, void* stream) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", who, cudaGetErrorString(e));
        std::exit(1);
    }
    // Deliberately NOT synchronising for a non-null stream: this is called once per layer from a captured
    // graph's worth of work, and `finish()`'s null-stream sync in `s_gemv.cu` is the pattern that made a whole
    // round of measurements the driver's cost instead of the kernel's (Memory/ERRORS.md RC-7).
    (void) stream;
}


// The new kernels read the activations as aligned words (`load_x_chunk`), so they need 4-byte aligned activation
// rows - the input's and the intermediate's in `scratch`; the per-hit ones also read a blob's codes as uint2, which
// needs an 8-byte aligned arena and slot size (the grouped kernels above already did).  Anything else keeps the
// previous kernels rather than issuing a misaligned load.
bool new_grouped(const void* x_q8_0, const void* scratch) {
    return ((uintptr_t) x_q8_0 & 3) == 0 && ((uintptr_t) scratch & 3) == 0;
}

}  // namespace

uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff) {
    if (n_hits <= 0) return 0;
    const uint64_t gu = (uint64_t) n_hits * (uint64_t) (2 * n_ff) * 4;
    const uint64_t q8 = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 34;
    // R4.2h: the fp32 scales for the INTERMEDIATE's own quantization, one per 32-element chunk per hit.
    const uint64_t hs = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 4;
    const uint64_t xh = (uint64_t) (n_embd / 32) * 4;
    return ((gu + 15) & ~15ull) + ((q8 + 15) & ~15ull) + 2 * ((hs + 15) & ~15ull) +
           ((xh + 15) & ~15ull);
}

namespace {

__global__ void add_hits_kernel(float* __restrict__ parts, const float* __restrict__ hit_out,
                                const int32_t* __restrict__ dst, const int32_t* __restrict__ count, int n_embd) {
    const int h = blockIdx.y;
    if (h >= *count) return;
    const size_t row = (size_t) dst[h] * (size_t) n_embd;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_embd; i += gridDim.x * blockDim.x)
        parts[row + i] += hit_out[row + i];
}
}  // namespace

namespace {
// Plan v0.3 P6: experts GROUPED - one blob pointer per group (a VRAM slot or a mapped host blob read over PCIe),
// every entry of the group (a token routed to that expert) computed from ONE read of each row.  Per entry the
// arithmetic runs chunk by chunk in the same lane order, so an entry's result does not depend on its group.
constexpr int GU_CHUNKS = (H / 32 + 31) / 32;   // 3: chunks of a gate/up row per lane (80 chunks / 32 lanes)
constexpr int GMAX = 8;                          // entries per group (tokens routed to one expert in a window)
// a group holds one entry per token of the window routed to its expert, and the kernels below keep at
// most GMAX of them (`min(..., GMAX)`): a longer window would drop entries without a word.
static_assert(GMAX >= kVerifyMaxT, "a verify window's group can exceed GMAX entries");
constexpr int GU_ROWS = 32;                      // gate/up rows per block: 4 per warp
constexpr int D_ROWS = 64;                       // down rows per block: 8 per warp

// One activation chunk's contribution: the 32 int8 of the chunk in aligned words against the row's 2-bit codes,
// four dp4a and one float expression.
__device__ __forceinline__ float chunk_dot(uint2 cb, const int* xw, float dw, float dx) {
    const uint8_t* cbytes = (const uint8_t*) &cb;
    int s = 0, hx = 0;
    const int ones = 0x01010101;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const unsigned cbyte = cbytes[j];
        const int cw = (int) ((cbyte & 3u) | (((cbyte >> 2) & 3u) << 8) | (((cbyte >> 4) & 3u) << 16) |
                              (((cbyte >> 6) & 3u) << 24));
        s = LAMINA_DP4A(cw, xw[j], s);
        hx = LAMINA_DP4A(ones, xw[j], hx);
    }
    return dw * dx * (float) (s - hx);
}

// Gate/up: a block = GU_ROWS rows of ONE group.  The group's activations (each entry's token row of x_q8_0 and
// its fp32 scales) are staged once into shared memory as aligned words; each warp then walks its rows, loading
// each lane's code chunks once and dotting them with every entry.
__global__ void __launch_bounds__(256) gu_grouped_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                         const int32_t* __restrict__ grp_start,
                                                         const int32_t* __restrict__ n_groups,
                                                         const int32_t* __restrict__ ent_tok,
                                                         const uint8_t* __restrict__ x_q8_0,
                                                         const float* __restrict__ x_scales,
                                                         float* __restrict__ gate_up, int cap_entries) {
    __shared__ int xs_q[GMAX][H / 4];          // the entries' int8 activations as words (2560 B each)
    __shared__ float xs_d[GMAX][H / 32];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], ne = min(grp_start[g + 1] - e0, GMAX);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    for (int i = t; i < ne * (H / 32); i += blockDim.x) {
        const int k = i / (H / 32), c = i - k * (H / 32);
        const uint8_t* xb = x_q8_0 + (size_t) ent_tok[e0 + k] * (size_t) (H / 32) * 34 + (size_t) c * 34;
        xs_d[k][c] = x_scales ? x_scales[(size_t) ent_tok[e0 + k] * (H / 32) + c] : f16_at(xb);
        const int8_t* q = (const int8_t*) (xb + 2);
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            int v;
            memcpy(&v, q + 4 * w, 4);
            xs_q[k][c * 8 + w] = v;
        }
    }
    __syncthreads();
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int row0 = blockIdx.x * GU_ROWS;
    for (int rr = warp; rr < GU_ROWS; rr += 8) {
        const int i = row0 + rr;
        const uint8_t* codes = blob + (size_t) i * ROW_GU;
        const uint8_t* scales = blob + O_GU_SCALES + (size_t) i * SC_GU * 2;
        uint2 cb[GU_CHUNKS];
        float dw[GU_CHUNKS];
#pragma unroll
        for (int q = 0; q < GU_CHUNKS; ++q) {
            const int c = lane + 32 * q;
            if (c < H / 32) {
                cb[q] = *(const uint2*) (codes + (size_t) c * 8);
                dw[q] = f16_at(scales + (size_t) (c >> 1) * 2);
            }
        }
        for (int k = 0; k < ne; ++k) {
            float acc = 0.0f;
#pragma unroll
            for (int q = 0; q < GU_CHUNKS; ++q) {
                const int c = lane + 32 * q;
                if (c >= H / 32) break;
                acc += chunk_dot(cb[q], &xs_q[k][c * 8], dw[q], xs_d[k][c]);
            }
            const float sum = warp_sum(acc);
            if (lane == 0) {
                const int e = e0 + k, r = i >> 1;
                const size_t base = (i & 1) ? ((size_t) cap_entries * FF + (size_t) e * FF) : ((size_t) e * FF);
                gate_up[base + (size_t) r] = sum;
            }
        }
    }
}

// Down: a block = D_ROWS rows of ONE group; the entries' quantized intermediates staged once.  A down row is 20
// chunks, so lanes 0..19 each hold one chunk, as in the per-entry kernel.
__global__ void __launch_bounds__(256) down_grouped_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                           const int32_t* __restrict__ grp_start,
                                                           const int32_t* __restrict__ n_groups,
                                                           const int32_t* __restrict__ ent_dst,
                                                           const uint8_t* __restrict__ h_q8_0,
                                                           const float* __restrict__ h_scales,
                                                           float* __restrict__ out) {
    __shared__ int hs_q[GMAX][FF / 4];
    __shared__ float hs_d[GMAX][FF / 32];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], ne = min(grp_start[g + 1] - e0, GMAX);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    for (int i = t; i < ne * (FF / 32); i += blockDim.x) {
        const int k = i / (FF / 32), c = i - k * (FF / 32);
        const uint8_t* xb = h_q8_0 + (size_t) (e0 + k) * (size_t) (FF / 32) * 34 + (size_t) c * 34;
        hs_d[k][c] = h_scales ? h_scales[(size_t) (e0 + k) * (FF / 32) + c] : f16_at(xb);
        const int8_t* q = (const int8_t*) (xb + 2);
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            int v;
            memcpy(&v, q + 4 * w, 4);
            hs_q[k][c * 8 + w] = v;
        }
    }
    __syncthreads();
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int row0 = blockIdx.x * D_ROWS;
    for (int rr = warp; rr < D_ROWS; rr += 8) {
        const int r = row0 + rr;
        const uint8_t* codes = blob + O_D_CODES + (size_t) r * ROW_D;
        const uint8_t* scales = blob + O_D_SCALES + (size_t) r * SC_D * 2;
        const int c = lane;
        uint2 cb = make_uint2(0, 0);
        float dw = 0.0f;
        if (c < FF / 32) {
            cb = *(const uint2*) (codes + (size_t) c * 8);
            dw = f16_at(scales + (size_t) (c >> 1) * 2);
        }
        for (int k = 0; k < ne; ++k) {
            float acc = 0.0f;
            if (c < FF / 32) acc += chunk_dot(cb, &hs_q[k][c * 8], dw, hs_d[k][c]);
            const float sum = warp_sum(acc);
            if (lane == 0) out[(size_t) ent_dst[e0 + k] * H + r] = sum;
        }
    }
}

// ---- The grouped kernels with the staged activations laid out for the reads.
//
// **THE LAYOUT ABOVE IS AN 8-WAY BANK CONFLICT.**  `xs_q[k][c * 8 + j]` puts lane `L`'s word `j` at `8L + j`, so
// lanes `L, L + 4, L + 8, ...` share a bank on every one of the chunk's eight reads; and `chunk_dot` recomputes
// `hx`, which depends on the entry alone, for every row.  Here the staging writes, once per (entry, chunk):
//
//     xs_w[j][k * NC + c]   word j of the regrouped chunk (`load_x_chunk`), so lanes read consecutive words
//     xs_dh[k * NC + c]     (dx, hx) - one 64-bit read per chunk instead of a float and eight `dp4a`
//
// Word-major, the staging writes are consecutive as well (the flat index `i` IS `k * NC + c`), so neither side
// conflicts and no padding is needed.  Each warp takes its rows TWO at a time (adjacent rows: gate `r` and up `r`,
// or down rows `r, r + 1`), so every shared read serves both; the codes of both are expanded once per row, not
// once per entry.  Per (row, entry) the lane's chunks, their order, the float expression and the shuffle
// reduction are the kernels' above, so every output is bitwise theirs.

// Stages chunk `i = k * NC + c` of a group: its regrouped words at `xs_w[j * STRIDE + i]` and (dx, hx) at `xs_dh[i]`.
template <int STRIDE>
__device__ __forceinline__ void stage_chunk(const uint8_t* __restrict__ xb, float dx, int* xs_w, int2* xs_dh, int i) {
    int X[8];
    const int hx = load_x_chunk(xb, X);
#pragma unroll
    for (int j = 0; j < 8; ++j) xs_w[j * STRIDE + i] = X[j];
    xs_dh[i] = make_int2(__float_as_int(dx), hx);
}

__global__ void __launch_bounds__(256) gu_grouped_t_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                           const int32_t* __restrict__ grp_start,
                                                           const int32_t* __restrict__ n_groups,
                                                           const int32_t* __restrict__ ent_tok,
                                                           const uint8_t* __restrict__ x_q8_0,
                                                           const float* __restrict__ x_scales,
                                                           float* __restrict__ gate_up, int cap_entries) {
    constexpr int NC = H / 32;
    __shared__ int xs_w[8 * GMAX * NC];          // 20 KB: word j of entry k's chunk c at [j][k * NC + c]
    __shared__ int2 xs_dh[GMAX * NC];            // 5 KB: (dx as bits, hx)
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], ne = min(grp_start[g + 1] - e0, GMAX);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    for (int i = t; i < ne * NC; i += blockDim.x) {
        const int k = i / NC, c = i - k * NC;
        const int tok = ent_tok[e0 + k];
        const uint8_t* xb = x_q8_0 + (size_t) tok * (size_t) NC * 34 + (size_t) c * 34;
        const float dx = x_scales ? x_scales[(size_t) tok * NC + c] : f16_ld(xb);
        stage_chunk<GMAX * NC>(xb, dx, xs_w, xs_dh, i);
    }
    __syncthreads();
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int row0 = blockIdx.x * GU_ROWS;                 // even: a pair is always (gate r, up r)
    for (int pp = warp; pp < GU_ROWS / 2; pp += 8) {
        const int i = row0 + 2 * pp;
        const uint8_t* codes = blob + (size_t) i * ROW_GU;
        const uint8_t* scales = blob + O_GU_SCALES + (size_t) i * SC_GU * 2;
        int m0[GU_CHUNKS][8], m1[GU_CHUNKS][8];
        float dw0[GU_CHUNKS], dw1[GU_CHUNKS];
#pragma unroll
        for (int q = 0; q < GU_CHUNKS; ++q) {
            const int c = lane + 32 * q;
            if (c < NC) {
                expand_codes(*(const uint2*) (codes + (size_t) c * 8), m0[q]);
                expand_codes(*(const uint2*) (codes + ROW_GU + (size_t) c * 8), m1[q]);
                dw0[q] = f16_ld(scales + (size_t) (c >> 1) * 2);
                dw1[q] = f16_ld(scales + SC_GU * 2 + (size_t) (c >> 1) * 2);
            }
        }
        for (int k = 0; k < ne; ++k) {
            float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
            for (int q = 0; q < GU_CHUNKS; ++q) {
                const int c = lane + 32 * q;
                if (c >= NC) break;
                const int at = k * NC + c;
                int X[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) X[j] = xs_w[j * (GMAX * NC) + at];
                const int2 dh = xs_dh[at];
                const float dx = __int_as_float(dh.x);
                acc0 += dw0[q] * dx * (float) (chunk_s(m0[q], X) - dh.y);
                acc1 += dw1[q] * dx * (float) (chunk_s(m1[q], X) - dh.y);
            }
            const float s0 = warp_sum(acc0);
            const float s1 = warp_sum(acc1);
            if (lane == 0) {
                const int e = e0 + k, r = i >> 1;
                gate_up[(size_t) e * FF + (size_t) r] = s0;
                gate_up[(size_t) cap_entries * FF + (size_t) e * FF + (size_t) r] = s1;
            }
        }
    }
}

__global__ void __launch_bounds__(256) down_grouped_t_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                             const int32_t* __restrict__ grp_start,
                                                             const int32_t* __restrict__ n_groups,
                                                             const int32_t* __restrict__ ent_dst,
                                                             const uint8_t* __restrict__ h_q8_0,
                                                             const float* __restrict__ h_scales,
                                                             float* __restrict__ out) {
    constexpr int NC = FF / 32;
    __shared__ int hs_w[8 * GMAX * NC];
    __shared__ int2 hs_dh[GMAX * NC];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int e0 = grp_start[g], ne = min(grp_start[g + 1] - e0, GMAX);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    for (int i = t; i < ne * NC; i += blockDim.x) {
        const int k = i / NC, c = i - k * NC;
        const uint8_t* xb = h_q8_0 + (size_t) (e0 + k) * (size_t) NC * 34 + (size_t) c * 34;
        const float dx = h_scales ? h_scales[(size_t) (e0 + k) * NC + c] : f16_ld(xb);
        stage_chunk<GMAX * NC>(xb, dx, hs_w, hs_dh, i);
    }
    __syncthreads();
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int row0 = blockIdx.x * D_ROWS;
    for (int pp = warp; pp < D_ROWS / 2; pp += 8) {
        const int r = row0 + 2 * pp;
        const uint8_t* codes = blob + O_D_CODES + (size_t) r * ROW_D;
        const uint8_t* scales = blob + O_D_SCALES + (size_t) r * SC_D * 2;
        const int c = lane;                                // lanes 0..19 hold one chunk each, as above
        int m0[8], m1[8];
        float dw0 = 0.0f, dw1 = 0.0f;
        if (c < NC) {
            expand_codes(*(const uint2*) (codes + (size_t) c * 8), m0);
            expand_codes(*(const uint2*) (codes + ROW_D + (size_t) c * 8), m1);
            dw0 = f16_ld(scales + (size_t) (c >> 1) * 2);
            dw1 = f16_ld(scales + SC_D * 2 + (size_t) (c >> 1) * 2);
        }
        for (int k = 0; k < ne; ++k) {
            float acc0 = 0.0f, acc1 = 0.0f;
            if (c < NC) {
                const int at = k * NC + c;
                int X[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) X[j] = hs_w[j * (GMAX * NC) + at];
                const int2 dh = hs_dh[at];
                const float dx = __int_as_float(dh.x);
                acc0 += dw0 * dx * (float) (chunk_s(m0, X) - dh.y);
                acc1 += dw1 * dx * (float) (chunk_s(m1, X) - dh.y);
            }
            const float s0 = warp_sum(acc0);
            const float s1 = warp_sum(acc1);
            if (lane == 0) {
                const size_t o = (size_t) ent_dst[e0 + k] * H + (size_t) r;
                out[o] = s0;
                out[o + 1] = s1;
            }
        }
    }
}
}  // namespace

namespace {
// Plan v0.3 P6: groups built on the device when every expert is resident at `base + id * blob` (the MTP layer):
// one block of 128 threads, groups in first-appearance order, entries of a group in routing order.
__global__ void group_resident_kernel(const int32_t* __restrict__ ids, int n, int k_per_tok, const uint8_t* base,
                                      long long blob, unsigned long long* __restrict__ grp_ptr,
                                      int32_t* __restrict__ grp_start, int32_t* __restrict__ counts,
                                      int32_t* __restrict__ ent_dst, int32_t* __restrict__ ent_tok) {
    __shared__ int e_s[128], first_s[128], size_s[128], gidx_s[128], gstart_s[129];
    const int i = threadIdx.x;
    const int e = i < n ? ids[i] : -1;
    e_s[i] = e;
    __syncthreads();
    int first = i, rank = 0, size = 0;
    if (i < n) {
        for (int j = 0; j < i; ++j)
            if (e_s[j] == e) { if (first == i) first = j; ++rank; }
        if (first == i)
            for (int j = i; j < n; ++j) size += e_s[j] == e;
    }
    first_s[i] = first;
    size_s[i] = (i < n && first == i) ? size : 0;
    __syncthreads();
    if (i == 0) {
        int gi = 0, acc = 0;
        for (int j = 0; j < n; ++j)
            if (first_s[j] == j) {
                gidx_s[j] = gi;
                gstart_s[gi] = acc;
                grp_ptr[gi] = (unsigned long long) (base + (size_t) e_s[j] * (size_t) blob);
                grp_start[gi] = acc;
                acc += size_s[j];
                ++gi;
            }
        grp_start[gi] = acc;
        counts[0] = gi;
        counts[1] = acc;
    }
    __syncthreads();
    if (i < n) {
        const int at = gstart_s[gidx_s[first]] + rank;
        ent_dst[at] = i;
        ent_tok[at] = i / k_per_tok;
    }
}
}  // namespace

void moe_group_resident(const int32_t* ids, int n, int k_per_tok, const uint8_t* base, int64_t blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* counts, int32_t* ent_dst,
                        int32_t* ent_tok, void* stream) {
    if (n < 1 || n > 128) { std::fprintf(stderr, "moe_group_resident: n must be 1..128\n"); std::exit(1); }
    group_resident_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(ids, n, k_per_tok, base, (long long) blob, grp_ptr,
                                                               grp_start, counts, ent_dst, ent_tok);
    check("moe_group_resident", stream);
}

void moe_grouped_s2(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                    const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                    const uint8_t* x_q8_0, const float* x_scales, void* scratch, float* out, void* stream) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    cudaStream_t cs = (cudaStream_t) stream;
    const uint64_t gu_bytes = ((uint64_t) cap_entries * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap_entries * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    const bool fast = new_grouped(x_q8_0, scratch);
    {
        const dim3 grid((unsigned) (2 * FF / GU_ROWS), (unsigned) cap_groups);
        if (fast)
            gu_grouped_t_kernel<<<grid, 256, 0, cs>>>(grp_ptr, grp_start, n_groups, ent_tok, x_q8_0, x_scales,
                                                      gate_up, (int) cap_entries);
        else
            gu_grouped_kernel<<<grid, 256, 0, cs>>>(grp_ptr, grp_start, n_groups, ent_tok, x_q8_0, x_scales, gate_up,
                                                    (int) cap_entries);
        check("moe_grouped_s2/gu", stream);
    }
    // #783 PR-k (stuchapin909): SwiGLU and the scaled Q8_0 quantize of its result in one kernel (gate_up still gets the
    // activation written back, so the parity checks read the same buffer); LAMINA_S2_SWIGLU_Q8=0 keeps the two launches
    static const bool swiglu_q8 = [] {
        const char* e = std::getenv("LAMINA_S2_SWIGLU_Q8");
        return e == nullptr || e[0] != '0';
    }();
    static_assert(FF % 32 == 0, "the fused SwiGLU + Q8_0 kernel works in whole 32-value blocks");
    if (fast && x_scales != nullptr && swiglu_q8) {
        const long long pairs = cap_entries * (long long) FF;
        const long long n_blocks = pairs >> 5;
        const int warps = THREADS / 32;
        swiglu_quantize_q8_0_scaled_kernel<<<(unsigned) ((n_blocks + warps - 1) / warps), THREADS, 0, cs>>>(
            gate_up, pairs, h_q8_0, h_scales);
        check("moe_grouped_s2/swiglu_q8", stream);
    } else {
        const long long pairs = cap_entries * (long long) FF;
        {
        swiglu_kernel<<<(unsigned) ((pairs + THREADS - 1) / THREADS), THREADS, 0, cs>>>(gate_up, pairs);
        check("moe_grouped_s2/swiglu", stream);
        if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap_entries * (int64_t) FF, stream);
        else quantize_q8_0(gate_up, h_q8_0, cap_entries * (int64_t) FF, stream);
        }
    }
    {
        const dim3 grid((unsigned) (H / D_ROWS), (unsigned) cap_groups);
        const float* hs = x_scales != nullptr ? h_scales : nullptr;
        if (fast)
            down_grouped_t_kernel<<<grid, 256, 0, cs>>>(grp_ptr, grp_start, n_groups, ent_dst, h_q8_0, hs, out);
        else
            down_grouped_kernel<<<grid, 256, 0, cs>>>(grp_ptr, grp_start, n_groups, ent_dst, h_q8_0, hs, out);
        check("moe_grouped_s2/down", stream);
    }
}

void moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                 int64_t n_embd, void* stream) {
    if (cap <= 0) return;
    const dim3 grid((unsigned) ((n_embd + 255) / 256 < 8 ? (n_embd + 255) / 256 : 8), (unsigned) cap);
    add_hits_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(parts, hit_out, dst, count, (int) n_embd);
    check("moe_hit_add", stream);
}

}  // namespace lamina::kernels
