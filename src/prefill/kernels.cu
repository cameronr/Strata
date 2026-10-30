// src/prefill/kernels.cu - see include/strata/prefill/kernels.hpp.
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/warp.cuh"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {

using strata::kernels::warp_max;
using strata::kernels::warp_sum;
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int S = 128, HK = 16, HV = 48, C = 10240;

__device__ __forceinline__ uint16_t bf(float f) { return strata::kernels::bf16_from_f32(f); }
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ uint16_t hf(float f) { return __half_as_ushort(__float2half_rn(f)); }
// A SwiGLU product for an FP16 GEMM, saturated: a token with a massive activation would turn into inf there and NaN
// after the down projection (decode's q8_1 has room to ~8e6). A NaN stays NaN; finite values round as before.
__device__ __forceinline__ uint16_t hf_sat(float f) { return hf(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f)); }
// block-wide sum for blockDim.x <= 1024, result broadcast
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float t = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (w == 0) t = warp_sum(t);
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    return sh[0];
}
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "prefill %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

// ---------------------------------------------------------------- hyper-connection
__global__ void gr_norm_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                               float* __restrict__ xn, uint16_t* __restrict__ xn16) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    for (int d = threadIdx.x; d < N; d += blockDim.x) {
        const float v = r[d] * rs * w[c * N + d];
        xn[row * N + d] = v;
        xn16[row * N + d] = bf(v);
    }
}
// gr_norm_kernel's row scale and BF16 image; gr_mix_r_kernel recomputes r * rs * w in the same order, so the FP32
// copy of the normalized rows is neither written nor read
__global__ void gr_norm_rs_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                                  float* __restrict__ rs_out, uint16_t* __restrict__ xn16) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
    for (int d = threadIdx.x; d < N; d += blockDim.x) xn16[row * N + d] = bf(r[d] * rs * w[c * N + d]);
}
__global__ void gr_mix_r_kernel(const float* __restrict__ R, const float* __restrict__ rs, const float* __restrict__ w,
                                const float* __restrict__ g, float* __restrict__ mixed, uint16_t* __restrict__ mixed16,
                                int64_t T, uint16_t* __restrict__ mixed_h) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        const float x = R[j] * rs[t * HC + c] * w[c * N + d];   // gr_norm_kernel's value, bit for bit
        s = fmaf(x, sigm(g[j]), s);
    }
    s /= (float) HC;
    if (mixed) mixed[i] = s;
    if (mixed16) mixed16[i] = bf(s);
    if (mixed_h) mixed_h[i] = hf(s);
}
// The read's up projection (lo16 . w_up^T, BF16 in, FP32 sums over K = 320) with gr_mix_r_kernel as its epilogue
// (upstream's gr_upmix_cuda_kernel, cffe31d4): `gated` (T x 10240 FP32) is neither written nor read back.  A block: 128
// tokens x 32 columns x the 4 streams, K in slices of 32 (cp.async, double-buffered); 8 warps of 32 tokens x 16 columns
// x 4 streams on mma.m16n8k16 (ldmatrix), K summed in order as cuBLAS's kernels do (gr_upmix_parity checks the T where
// cuBLAS takes such a kernel); a lane then mixes its (token, column) pairs of all 4 streams in registers in
// gr_mix_r_kernel's order.  PF: the epilogue's R rows are asked into L2 at the start, where the L2 holds those of the
// blocks in flight (the 5070 Ti's 48 MB: 1.71x the pair at 2048 tokens against 1.41x without; on the 3090's 6 MB it
// costs 8%).  The last tile's missing rows are zeros (a cp.async of 0 bytes): read from one row, they ran 2-3x slower.
constexpr int UM_BM = 128, UM_BD = 32, UM_KC = 32, UM_LD = UM_KC + 8;   // tokens, columns a stream, k a slice, a row
__device__ __forceinline__ void um_cp16(void* s, const void* g, unsigned bytes = 16) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)), "l"(g),
                 "r"(bytes));
}
__device__ __forceinline__ void um_ldm4(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void um_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <bool PF>
__global__ void __launch_bounds__(256, 2) gr_upmix_kernel(const uint16_t* __restrict__ lo16, const uint16_t* __restrict__ wu,
                                                          const float* __restrict__ R, const float* __restrict__ rs,
                                                          const float* __restrict__ w, float* __restrict__ mixed,
                                                          uint16_t* __restrict__ mixed16, uint16_t* __restrict__ mixed_h,
                                                          int64_t T) {
    __shared__ __align__(16) uint16_t sA[2][UM_BM][UM_LD];
    __shared__ __align__(16) uint16_t sB[2][HC * UM_BD][UM_LD];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 1, wn = warp & 1;
    // column tiles fastest: the blocks in flight share a token tile (its lo16 rows stay in L2)
    const int64_t t0 = (int64_t) blockIdx.y * UM_BM;
    const int d0 = blockIdx.x * UM_BD;
    if (PF)
        for (int i = tid; i < UM_BM * HC; i += 256) {
            const int64_t t = t0 + i / HC;
            if (t < T) asm volatile("prefetch.global.L2 [%0];" ::"l"(R + t * D + (i % HC) * N + d0));
        }
    auto stage = [&](int kc, int buf) {
        for (int i = tid; i < UM_BM * (UM_KC / 8); i += 256) {
            const int r = i / (UM_KC / 8), q = i % (UM_KC / 8);
            const bool in = t0 + r < T;
            um_cp16(&sA[buf][r][q * 8], lo16 + (in ? t0 + r : 0) * LR + kc * UM_KC + q * 8, in ? 16 : 0);
        }
        for (int i = tid; i < HC * UM_BD * (UM_KC / 8); i += 256) {
            const int r = i / (UM_KC / 8), q = i % (UM_KC / 8);
            um_cp16(&sB[buf][r][q * 8], wu + ((int64_t) (r / UM_BD) * N + d0 + r % UM_BD) * LR + kc * UM_KC + q * 8);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[2][HC][2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int c = 0; c < HC; ++c)
#pragma unroll
            for (int s = 0; s < 2; ++s)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[mt][c][s][i] = 0.0f;
    constexpr int NK = LR / UM_KC;
    stage(0, 0);
    for (int kc = 0; kc < NK; ++kc) {
        const int buf = kc & 1;
        if (kc + 1 < NK) {
            stage(kc + 1, buf ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < UM_KC / 16; ++ks) {
            unsigned a[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) um_ldm4(a[mt], &sA[buf][wm * 32 + mt * 16 + (lane & 15)][ks * 16 + (lane >> 4) * 8]);
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                unsigned b[4];   // b[0..1]: columns 0-7 of the warp's 16, b[2..3]: 8-15 (k 0-7, 8-15 each)
                um_ldm4(b, &sB[buf][c * UM_BD + wn * 16 + (lane & 7) + ((lane >> 4) << 3)][ks * 16 + ((lane >> 3) & 1) * 8]);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    um_mma(acc[mt][c][0], a[mt], b[0], b[1]);
                    um_mma(acc[mt][c][1], a[mt], b[2], b[3]);
                }
            }
        }
        __syncthreads();   // every read of `buf` before stage(kc + 2) refills it
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int64_t t = t0 + wm * 32 + mt * 16 + (lane >> 2) + h * 8;
            if (t >= T) continue;
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int d = d0 + wn * 16 + s * 8 + (lane & 3) * 2;
                float sm[2] = {0.0f, 0.0f};
#pragma unroll
                for (int c = 0; c < HC; ++c) {
                    const float2 r2 = *reinterpret_cast<const float2*>(R + t * D + c * N + d);
                    const float2 w2 = *reinterpret_cast<const float2*>(w + c * N + d);
                    const float rsc = rs[t * HC + c];
                    const float x0 = r2.x * rsc * w2.x, x1 = r2.y * rsc * w2.y;   // gr_mix_r_kernel's order
                    sm[0] = fmaf(x0, sigm(acc[mt][c][s][h * 2 + 0]), sm[0]);
                    sm[1] = fmaf(x1, sigm(acc[mt][c][s][h * 2 + 1]), sm[1]);
                }
                sm[0] /= (float) HC;
                sm[1] /= (float) HC;
                if (mixed) *reinterpret_cast<float2*>(mixed + t * N + d) = make_float2(sm[0], sm[1]);
                if (mixed16)
                    *reinterpret_cast<uint32_t*>(mixed16 + t * N + d) = (uint32_t) bf(sm[0]) | ((uint32_t) bf(sm[1]) << 16);
                if (mixed_h)
                    *reinterpret_cast<uint32_t*>(mixed_h + t * N + d) = (uint32_t) hf(sm[0]) | ((uint32_t) hf(sm[1]) << 16);
            }
        }
}
// gr_write_kernel for one row (t, c), then gr_norm_rs_kernel's reduction over it with the next read's norm weights:
// the same thread-to-element mapping (256 threads, stride 256) and block_sum, so rs and the BF16 image are the same
// bits, and R is not read back
constexpr int GRW_PER = N / 256;
__global__ void __launch_bounds__(256)
gr_write_norm_rs_kernel(float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj,
                        int64_t inj_ld, const float* __restrict__ w, float eps, float* __restrict__ rs_out,
                        uint16_t* __restrict__ xn16, const uint16_t* __restrict__ partial) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int64_t t = row / HC;
    const int c = (int) (row % HC);
    float* r = R + row * N;
    const float sc = 2.0f * sigm(inj[t * inj_ld + c] / (float) HC);
    float v[GRW_PER];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < GRW_PER; ++k) {
        const int d = threadIdx.x + 256 * k;
        const float b = partial ? bo[t * N + d] + __half2float(__ushort_as_half(partial[t * N + d])) : bo[t * N + d];
        const float x = fmaf(b, sc, r[d]);
        r[d] = x;
        v[k] = x;
        ss += x * x;
    }
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
#pragma unroll
    for (int k = 0; k < GRW_PER; ++k) {
        const int d = threadIdx.x + 256 * k;
        xn16[row * N + d] = bf(v[k] * rs * w[c * N + d]);
    }
}
__global__ void gr_silu_kernel(const float* __restrict__ lo, uint16_t* __restrict__ lo16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = lo[i] / (float) HC;
    lo16[i] = bf(x / (1.0f + __expf(-x)));
}
__global__ void gr_mix_kernel(const float* __restrict__ xn, const float* __restrict__ g, float* __restrict__ mixed,
                              uint16_t* __restrict__ mixed16, int64_t T, uint16_t* __restrict__ mixed_h) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        s = fmaf(xn[j], sigm(g[j]), s);
    }
    s /= (float) HC;
    if (mixed) mixed[i] = s;
    if (mixed16) mixed16[i] = bf(s);
    if (mixed_h) mixed_h[i] = hf(s);
}
__global__ void gr_write_kernel(float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj,
                                int64_t inj_ld, int64_t T, const uint16_t* __restrict__ partial) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, c = (i % D) / N, d = i % N;
    const float b = partial ? bo[t * N + d] + __half2float(__ushort_as_half(partial[t * N + d])) : bo[t * N + d];
    R[i] = fmaf(b, 2.0f * sigm(inj[t * inj_ld + c] / (float) HC), R[i]);
}
__global__ void gr_broadcast_kernel(const float* __restrict__ e, float* __restrict__ R, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, d = i % N;
    R[i] = e[t * N + d];
}

// ---------------------------------------------------------------- GDN
__global__ void gdn_gates_kernel(const float* __restrict__ ab, const float* __restrict__ dt,
                                 const float* __restrict__ ssm_a, float* __restrict__ gate, float* __restrict__ beta,
                                 int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * HV) return;
    const int64_t t = i / HV, h = i % HV;
    const float v = ab[t * 2 * HV + h] + dt[h];
    gate[i] = (v > 20.0f ? v : log1pf(__expf(v))) * ssm_a[h];
    beta[i] = sigm(ab[t * 2 * HV + HV + h]);
}
// The conv history after a chunk: its last three inputs, older ones moved up when the chunk is shorter.
__global__ void gdn_conv_hist_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float old[3] = {hist[c * 3], hist[c * 3 + 1], hist[c * 3 + 2]};
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        const int64_t j = T - 3 + r;
        hist[c * 3 + r] = j >= 0 ? qkv[j * C + c] : old[r + T];
    }
}
// y16 = rmsnorm(o) * gamma * sigmoid(z) per (token, head) in FP16 (the out projection's input), from o in y; with
// Y32 its FP32 value too, in place over o
template <bool Y32>
__global__ void __launch_bounds__(S) gdn_norm_kernel(float* __restrict__ y, const float* __restrict__ z,
                                                     const float* __restrict__ gamma, float eps,
                                                     uint16_t* __restrict__ y16) {
    __shared__ float wsum[S / 32];
    const size_t i = ((size_t) blockIdx.y * HV + blockIdx.x) * S + threadIdx.x;
    const float oc = y[i];
    const float sp = warp_sum(oc * oc);
    if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = sp;
    __syncthreads();
    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float v = oc * rsqrtf(ss / (float) S + eps) * gamma[threadIdx.x] * sigm(z[i]);
    if constexpr (Y32) y[i] = v;
    y16[i] = hf(v);
}

// ---------------------------------------------------------------- MoE
__global__ void route_kernel(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ wout,
                             int64_t T) {
    const int64_t t = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (t >= T) return;
    const int lane = threadIdx.x & 31;
    const float* lg = logits + t * 512;
    float v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = lg[lane + i * 32];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) mx = fmaxf(mx, v[i]);
    mx = warp_max(mx);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) { v[i] = expf(v[i] - mx); sum += v[i]; }
    const float rcp = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) { v[i] *= rcp; if (isnan(v[i])) v[i] = -FLT_MAX; }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = v[0];
        int ex = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) if (v[i] > best) { best = v[i]; ex = lane + i * 32; }
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, m);
            const int oi = __shfl_xor_sync(0xffffffffu, ex, m);
            if (ob > best || (ob == best && oi < ex)) { best = ob; ex = oi; }
        }
        if ((ex & 31) == lane) { v[ex / 32] = -INFINITY; selected_sum += best; }
        if (lane == 0) ids[t * 10 + rank] = ex;
        if (rank == lane) selected = best;
    }
    selected_sum = fmaxf(warp_sum(selected_sum), 6.103515625e-5f);
    if (lane < 10) wout[t * 10 + lane] = selected / selected_sum;
}
// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down scales [2560][10] f16
template <bool HALF>
__global__ void blob_dequant_kernel(const uint8_t* __restrict__ blob, uint16_t* __restrict__ gu16,
                                    uint16_t* __restrict__ d16) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                     O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one thread per 4 weights (one code byte)
    const int64_t n_gu = 1280LL * 640, n_d = 2560LL * 160;
    if (i < n_gu) {
        const int64_t row = i / 640, byte = i % 640;
        const uint8_t c = blob[row * 640 + byte];
        const uint8_t* sp = blob + O_GU_SC + (size_t) (row * 40 + (byte * 4) / 64) * 2;
        const float d = __half2float(__ushort_as_half((uint16_t) (sp[0] | (sp[1] << 8))));
        uint16_t* o = gu16 + row * 2560 + byte * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float v = (float) (((c >> (2 * k)) & 3) - 1) * d; o[k] = HALF ? hf(v) : bf(v); }
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / 160, byte = j % 160;
        const uint8_t c = blob[O_D_CODES + row * 160 + byte];
        const uint8_t* sp = blob + O_D_SC + (size_t) (row * 10 + (byte * 4) / 64) * 2;
        const float d = __half2float(__ushort_as_half((uint16_t) (sp[0] | (sp[1] << 8))));
        uint16_t* o = d16 + row * 640 + byte * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float v = (float) (((c >> (2 * k)) & 3) - 1) * d; o[k] = HALF ? hf(v) : bf(v); }
    }
}
__global__ void swiglu_il_kernel(const float* __restrict__ gu, uint16_t* __restrict__ h16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const int64_t r = i / 640, k = i % 640;
    const float g = gu[r * 1280 + 2 * k], u = gu[r * 1280 + 2 * k + 1];
    h16[i] = hf_sat(g / (1.0f + __expf(-g)) * u);
}
__global__ void swiglu_pair_kernel(const float* __restrict__ g, const float* __restrict__ u, uint16_t* __restrict__ h16,
                                   int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const float a = g[i];
    h16[i] = hf_sat(a / (1.0f + __expf(-a)) * u[i]);
}
__global__ void gather_rows16_kernel(const uint16_t* __restrict__ x, const int32_t* __restrict__ src,
                                     uint16_t* __restrict__ dst, int64_t n, int64_t width) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one uint4 (8 bf16)
    const int64_t per = width / 8;
    if (i >= n * per) return;
    const int64_t r = i / per, j = i % per;
    reinterpret_cast<uint4*>(dst)[r * per + j] = reinterpret_cast<const uint4*>(x)[(int64_t) src[r] * per + j];
}
__global__ void gather_rows16_f32_kernel(const uint16_t* __restrict__ x, const int32_t* __restrict__ src,
                                         float* __restrict__ dst, int64_t n, int64_t width) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // 8 values
    const int64_t per = width / 8;
    if (i >= n * per) return;
    const int64_t r = i / per, j = i % per;
    const uint4 v = reinterpret_cast<const uint4*>(x)[(int64_t) src[r] * per + j];
    const __half2* h = reinterpret_cast<const __half2*>(&v);
    float4* o = reinterpret_cast<float4*>(dst + r * width + j * 8);
    const float2 a = __half22float2(h[0]), b = __half22float2(h[1]), c = __half22float2(h[2]), d = __half22float2(h[3]);
    o[0] = make_float4(a.x, a.y, b.x, b.y);
    o[1] = make_float4(c.x, c.y, d.x, d.y);
}
__global__ void moe_shared_gate_kernel(float* __restrict__ x, const float* __restrict__ sg, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    x[i] = x[i] * sigm(sg[i / N]);
}
__global__ void moe_scatter_add_kernel(float* __restrict__ sum, const float* __restrict__ rows,
                                       const float* __restrict__ w, const int32_t* __restrict__ src, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * N) return;
    const int64_t r = i / N, d = i % N;
    float* o = sum + (int64_t) src[r] * N + d;
    *o = fmaf(w[r], rows[i], *o);
}

// ---------------------------------------------------------------- QSA helpers
__global__ void rms_rows_kernel(float* __restrict__ x, const float* __restrict__ w, int64_t cols, int64_t ld, float eps) {
    __shared__ float sh[32];
    float* r = x + (int64_t) blockIdx.x * ld;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) ss += r[c] * r[c];
    const float s = rsqrtf(block_sum(ss, sh) / (float) cols + eps);
    __syncthreads();
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) r[c] = s * r[c] * w[c];
}
template <int MODE>
__global__ void rope_kernel(float* __restrict__ x, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
                            float theta_scale, const int32_t* __restrict__ mtab, strata::kernels::RopeKernelArgs k,
                            strata::kernels::RopeTab rt) {
    const int64_t row = blockIdx.x;             // t * heads + h
    const int pair = threadIdx.x;               // 0..31
    const int64_t t = row / heads, h = row % heads;
    float* p = x + t * ld + h * dim;
    float c, s;
    strata::kernels::rope_cs<MODE>(rt, k, theta_scale, strata::kernels::mrope_pos(mtab, (int) (pos0 + t), pair), pair, c,
                                   s);
    const float a = p[pair], b = p[pair + 32];
    p[pair] = a * c - b * s;
    p[pair + 32] = a * s + b * c;
}
__global__ void split_q_kernel(const float* __restrict__ qf, float* __restrict__ q, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    q[i] = qf[t * 24 * 512 + h * 512 + d];
}
__global__ void gate_attn_kernel(const float* __restrict__ a, const float* __restrict__ qf, uint16_t* __restrict__ o16,
                                 int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    o16[i] = hf(a[i] * (1.0f / (1.0f + expf(-qf[t * 24 * 512 + h * 512 + 256 + d]))));
}

// one block per (token, kv head, 64-value group[, side]); 64 threads
__global__ void kv_append_kernel(const float* __restrict__ K, const float* __restrict__ V, int64_t pos0,
                                 const int32_t* __restrict__ table, int64_t page_size, uint16_t* k_pool,
                                 uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
                                 int sides) {
    const int64_t t = blockIdx.x;
    const int kvh = blockIdx.y, g = sides == 3 ? blockIdx.z >> 1 : blockIdx.z;
    const bool is_v = sides == 3 ? (blockIdx.z & 1) != 0 : sides == 2;
    const int d = g * 64 + threadIdx.x;
    const float x = (is_v ? V : K)[t * 512 + kvh * 256 + d];
    const int64_t pos = pos0 + t;
    const int64_t page = table[pos / page_size];
    const int64_t row = (page * 2 + kvh) * page_size + pos % page_size;
    if (k_pool != nullptr) {
        (is_v ? v_pool : k_pool)[row * 256 + d] = hf(x);
        return;
    }
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float wm[2];
    if ((threadIdx.x & 31) == 0) wm[threadIdx.x >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(wm[0], wm[1]);
    const uint16_t sb = hf(amax / 127.0f);
    const float sf = __half2float(__ushort_as_half(sb));
    int q = 0;
    if (sf > 0.0f) { q = __float2int_rn(x / sf); q = q < -127 ? -127 : (q > 127 ? 127 : q); }
    (is_v ? v_q : k_q)[row * 256 + d] = (int8_t) q;
    if (threadIdx.x == 0) (is_v ? v_scale : k_scale)[row * 4 + g] = sb;
}
__global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = hf(x[i]);
}
__global__ void q8_to_bf16_kernel(const int8_t* __restrict__ q, const uint16_t* __restrict__ s, int64_t n,
                                  uint16_t* __restrict__ y) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = bf((float) q[i] * __half2float(__ushort_as_half(s[i / 32])));
}
__global__ void to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = bf(x[i]);
}

// ---------------------------------------------------------------- GDN, the chunkwise form
// Per value head, a chunk of GCL tokens from state S0 (rows: the key dimension, columns: the value dimension), with
// g_t = exp(gate_t) and gamma_t the gates' sum from the chunk's start to t:
//   A[t][i] = beta_t exp(gamma_t - gamma_i) k_t.k_i (i < t),   T = (I + A)^-1,
//   B_t = beta_t (v_t - exp(gamma_t) S0^T k_t),   D = T B (the delta rule's corrections),
//   o_t = exp(gamma_t) S0^T q_t + sum_{i <= t} exp(gamma_t - gamma_i) (q_t.k_i) D_i,
//   S = exp(gamma_L) S0 + sum_t exp(gamma_L - gamma_t) k_t D_t^T.
// gdn_chunk_kernel: a block per key head and GCW value columns, chunk after chunk.  Six walk warps, a value head and 16
// columns each: the state slice S^T [16 x 128] in FP32 registers as mma accumulators, every product an m16n8k16 FP16
// mma with FP32 sums (an accumulator tile pair is the next mma's A operand), K and Q through ldmatrix.  Two prep warps
// a chunk ahead: K K^T, Q K^T (from hi + lo FP16 halves: the FP32 dots' precision), T, P and the scales, K and Q in
// FP16.  Two loader warps two chunks ahead (cp.async).  One barrier a chunk: the walk warps only compute.
constexpr int VPK = HV / HK;     // value heads a key head
constexpr int GCL = 16;          // tokens a chunk
constexpr int GCW = 32;          // value columns a block
constexpr int GKP = S + 8;       // padded FP32 K / Q rows: a half-warp's 64-bit loads on distinct banks
constexpr int GKH = S + 8;       // padded FP16 K / Q rows: ldmatrix's 8 rows on distinct 16-byte bank groups
constexpr int GTP = GCL + 8;     // padded FP16 T / P rows
constexpr int GVP = GCW + 4;     // padded V rows
constexpr int GWALK = 32 * 2 * VPK, GPREP = 64, GLOAD = 64;   // the block's threads: walk, prep, loaders
struct alignas(16) GdnIn {       // the loaders' K, Q, gates and betas: chunk c in buffer c % 2
    float k[GCL][GKP], q[GCL][GKP];
    float gate[VPK][GCL], beta[VPK][GCL];
};
struct alignas(16) GdnV {        // V: c % 2, a chunk ahead
    float v[VPK][GCL][GVP];
};
struct alignas(16) GdnTps {      // a value head's chunk
    __half t[GCL][GTP], p[GCL][GTP];
    float s[64];                 // exp(gamma_t), exp(gamma_L - gamma_t), beta_t, exp(gamma_L)
};
struct alignas(16) GdnOut {      // the prep's: c % 3
    __half k[GCL][GKH], q[GCL][GKH];
    GdnTps tp[VPK];
};
struct alignas(16) GdnScratch {  // the prep's working set
    float kk[GCL][GCL + 1], qk[GCL][GCL + 1];
    float a[VPK][GCL][GCL + 1];
    float gm[VPK][GCL];
};
constexpr int GDN_SMEM = 2 * sizeof(GdnIn) + 2 * sizeof(GdnV) + 3 * sizeof(GdnOut) + sizeof(GdnScratch);
__device__ __forceinline__ uint32_t gh2(float lo, float hi) {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ void gsplit(float x, float y, uint32_t& hi, uint32_t& lo) {   // x ~ hi + lo / 2^11
    const __half2 h = __floats2half2_rn(x, y);
    const float2 f = __half22float2(h);
    const __half2 l = __floats2half2_rn((x - f.x) * 2048.0f, (y - f.y) * 2048.0f);
    hi = *reinterpret_cast<const uint32_t*>(&h);
    lo = *reinterpret_cast<const uint32_t*>(&l);
}
__device__ __forceinline__ uint32_t gld32(const __half* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ void gmma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void gmmah(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {   // FP16 sums
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ga_of(const float (&x)[4], const float (&y)[4], uint32_t (&a)[4]) {
    a[0] = gh2(x[0], x[1]);
    a[1] = gh2(x[2], x[3]);
    a[2] = gh2(y[0], y[1]);
    a[3] = gh2(y[2], y[3]);
}
__device__ __forceinline__ void gldsm(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void gldsm_t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void gcp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void gcp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ float gex2(float x) {   // 2^x, subnormal results flushed
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// prep warp 0: K K^T from FP16 K, which it keeps for the walk; prep warp 1: Q K^T as hi(Q) hi(K) with FP32 sums plus
// (hi(Q) lo(K) + lo(Q) hi(K)) / 2^11 with FP16 sums (twice the rate), keeping hi(Q)
template <bool QK>
__device__ __forceinline__ void gdn_dots(const GdnIn& in, GdnOut& out, GdnScratch& x, int lane) {
    const int g = lane >> 2, q = lane & 3;
    const float(*m)[GKP] = QK ? in.q : in.k;
    __half(*m16)[GKH] = QK ? out.q : out.k;
    float acc[2][2][4] = {};   // [n tile][k step parity]
    uint32_t cor[2][2][2] = {};
#pragma unroll
    for (int kt = 0; kt < S / 16; ++kt) {
        const int c = 16 * kt + 2 * q;
        const float2 a0 = *reinterpret_cast<const float2*>(&m[g][c]);
        const float2 a1 = *reinterpret_cast<const float2*>(&m[g + 8][c]);
        const float2 a2 = *reinterpret_cast<const float2*>(&m[g][c + 8]);
        const float2 a3 = *reinterpret_cast<const float2*>(&m[g + 8][c + 8]);
        uint32_t a[4], al[4];
        gsplit(a0.x, a0.y, a[0], al[0]);
        gsplit(a1.x, a1.y, a[1], al[1]);
        gsplit(a2.x, a2.y, a[2], al[2]);
        gsplit(a3.x, a3.y, a[3], al[3]);
        *reinterpret_cast<uint32_t*>(&m16[g][c]) = a[0];
        *reinterpret_cast<uint32_t*>(&m16[g + 8][c]) = a[1];
        *reinterpret_cast<uint32_t*>(&m16[g][c + 8]) = a[2];
        *reinterpret_cast<uint32_t*>(&m16[g + 8][c + 8]) = a[3];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const float2 k0 = *reinterpret_cast<const float2*>(&in.k[8 * nt + g][c]);
            const float2 k1 = *reinterpret_cast<const float2*>(&in.k[8 * nt + g][c + 8]);
            uint32_t b0, b1, l0, l1;
            gsplit(k0.x, k0.y, b0, l0);
            gsplit(k1.x, k1.y, b1, l1);
            gmma(acc[nt][kt & 1], a, b0, b1);
            if (QK) {
                gmmah(cor[nt][kt & 1], al, b0, b1);
                gmmah(cor[nt][kt & 1], a, l0, l1);
            }
        }
    }
    float(*o)[GCL + 1] = QK ? x.qk : x.kk;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {   // accumulator pair h: row g + 8 h, columns 2 q, 2 q + 1
            float r0 = acc[nt][0][2 * h] + acc[nt][1][2 * h], r1 = acc[nt][0][2 * h + 1] + acc[nt][1][2 * h + 1];
            if (QK) {
                const float2 c0 = __half22float2(*reinterpret_cast<const __half2*>(&cor[nt][0][h]));
                const float2 c1 = __half22float2(*reinterpret_cast<const __half2*>(&cor[nt][1][h]));
                r0 += (c0.x + c1.x) * (1.0f / 2048.0f);
                r1 += (c0.y + c1.y) * (1.0f / 2048.0f);
            }
            o[g + 8 * h][8 * nt + 2 * q] = r0;
            o[g + 8 * h][8 * nt + 2 * q + 1] = r1;
        }
}

// a chunk's T, P, scales and FP16 K, Q from its staged inputs, by the prep threads (pt: 0 .. 63)
__device__ __forceinline__ void gdn_prep(const GdnIn& in, GdnOut& out, GdnScratch& x, int pt) {
    const int lane = pt & 31, jj = pt >> 4, tt = pt & 15;   // 48 threads: value head jj, its token or column tt
    {   // gamma: the gates' running sum (0 past the prompt's end)
        float gv = jj < VPK ? in.gate[jj][tt] : 0.0f;
#pragma unroll
        for (int d = 1; d < GCL; d <<= 1) {
            const float y = __shfl_up_sync(0xffffffffu, gv, d, GCL);
            if (tt >= d) gv += y;
        }
        if (jj < VPK) x.gm[jj][tt] = gv;
    }
    if (pt >> 5) gdn_dots<true>(in, out, x, lane);
    else gdn_dots<false>(in, out, x, lane);
    asm volatile("bar.sync 1, 64;\n" ::: "memory");
    if (jj < VPK) {   // row tt: A (FP32), P (FP16 pairs), the scales; branch-free, the row's dots loaded first
        constexpr float LOG2E = 1.4426950408889634f;
        const float gt = x.gm[jj][tt], bt = in.beta[jj][tt], gl = x.gm[jj][GCL - 1];
        float gi[GCL], e[GCL], kr[GCL], qr[GCL];
#pragma unroll
        for (int i = 0; i < GCL; i += 4) *reinterpret_cast<float4*>(&gi[i]) = *reinterpret_cast<const float4*>(&x.gm[jj][i]);
#pragma unroll
        for (int i = 0; i < GCL; ++i) {
            kr[i] = x.kk[tt][i];
            qr[i] = x.qk[tt][i];
        }
#pragma unroll
        for (int i = 0; i < GCL; ++i) e[i] = gex2((i <= tt ? gt - gi[i] : 0.0f) * LOG2E);
        uint32_t* pr = reinterpret_cast<uint32_t*>(out.tp[jj].p[tt]);
#pragma unroll
        for (int i = 0; i < GCL; i += 2) {
            x.a[jj][tt][i] = i < tt ? bt * e[i] * kr[i] : 0.0f;
            x.a[jj][tt][i + 1] = i + 1 < tt ? bt * e[i + 1] * kr[i + 1] : 0.0f;
            pr[i / 2] = gh2(i <= tt ? e[i] * qr[i] : 0.0f, i + 1 <= tt ? e[i + 1] * qr[i + 1] : 0.0f);
        }
        float* s = out.tp[jj].s;
        s[tt] = gex2(gt * LOG2E);
        s[GCL + tt] = gex2((gl - gt) * LOG2E);
        s[2 * GCL + tt] = bt;
        if (tt == 0) s[3 * GCL] = gex2(gl * LOG2E);
    }
    asm volatile("bar.sync 1, 64;\n" ::: "memory");
    if (jj < VPK) {   // T = (I + A)^-1, column tt: x_t = -sum_{m < t} A[t][m] x_m, x_tt = 1
        float xv[GCL];
#pragma unroll
        for (int t = 0; t < GCL; ++t) xv[t] = t == tt ? 1.0f : 0.0f;
        const float(*a)[GCL + 1] = x.a[jj];
#pragma unroll
        for (int m = 0; m < GCL - 1; ++m)
#pragma unroll
            for (int t = m + 1; t < GCL; ++t) xv[t] = fmaf(-a[t][m], xv[m], xv[t]);
#pragma unroll
        for (int t = 0; t < GCL; ++t) out.tp[jj].t[t][tt] = __float2half_rn(xv[t]);
    }
}

__global__ void __launch_bounds__(GWALK + GPREP + GLOAD, 1) gdn_chunk_kernel(float* __restrict__ state,
                                                                             const float* __restrict__ h,
                                                                             const float* __restrict__ gate,
                                                                             const float* __restrict__ beta,
                                                                             float* __restrict__ out, int T) {
    extern __shared__ __align__(16) unsigned char gdn_smem[];
    GdnIn* in = reinterpret_cast<GdnIn*>(gdn_smem);
    GdnV* vb = reinterpret_cast<GdnV*>(gdn_smem + 2 * sizeof(GdnIn));
    GdnOut* ob = reinterpret_cast<GdnOut*>(gdn_smem + 2 * sizeof(GdnIn) + 2 * sizeof(GdnV));
    GdnScratch& px = *reinterpret_cast<GdnScratch*>(gdn_smem + 2 * sizeof(GdnIn) + 2 * sizeof(GdnV) + 3 * sizeof(GdnOut));
    const int kh = blockIdx.x, cs = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nch = (T + GCL - 1) / GCL;
    // a chunk's barrier: its prepared operands, V and the next chunk's inputs are in, the buffers of the chunks before
    // free
    if (tid >= GWALK + GPREP) {   // the loaders: chunk c + 2's K, Q, gates and betas and c + 1's V while c is walked
        const int lt = tid - GWALK - GPREP;
        auto copy = [&](int ckq, int cv) {   // a cp.async group every call, nothing past the last chunk
            if (ckq < nch) {
                GdnIn& b = in[ckq & 1];
                const int t0 = ckq * GCL, n = min(GCL, T - t0);
                const float* hc = h + (size_t) t0 * C;
#pragma unroll
                for (int k = 0; k < 2 * GCL * (S / 4) / GLOAD; ++k) {
                    const int i = lt + k * GLOAD, w = i / (GCL * (S / 4)), t = (i / (S / 4)) % GCL, r4 = (i % (S / 4)) * 4;
                    float* d = w ? &b.k[t][r4] : &b.q[t][r4];
                    if (t < n) gcp16(d, hc + t * C + (w ? HK * S : 0) + kh * S + r4);
                    else *reinterpret_cast<float4*>(d) = make_float4(0.f, 0.f, 0.f, 0.f);
                }
                for (int i = lt; i < 2 * VPK * GCL; i += GLOAD) {
                    const int w = i / (VPK * GCL), jj = (i / GCL) % VPK, t = i % GCL;
                    float* d = w ? &b.beta[jj][t] : &b.gate[jj][t];
                    if (t < n) gcp4(d, (w ? beta : gate) + (size_t) (t0 + t) * HV + kh + jj * HK);
                    else *d = 0.0f;
                }
            }
            if (cv >= 0 && cv < nch) {
                GdnV& v = vb[cv & 1];
                const int t0 = cv * GCL, n = min(GCL, T - t0);
                const float* hc = h + (size_t) t0 * C;
#pragma unroll
                for (int k = 0; k < VPK * GCL * (GCW / 4) / GLOAD; ++k) {
                    const int i = lt + k * GLOAD, jj = i / (GCL * (GCW / 4)), t = (i / (GCW / 4)) % GCL;
                    const int c4 = (i % (GCW / 4)) * 4;
                    float* d = &v.v[jj][t][c4];
                    if (t < n) gcp16(d, hc + t * C + 2 * HK * S + (kh + jj * HK) * S + cs * GCW + c4);
                    else *reinterpret_cast<float4*>(d) = make_float4(0.f, 0.f, 0.f, 0.f);
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };
        copy(0, 0);
        copy(1, -1);
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
        for (int ch = 0; ch < nch; ++ch) {
            asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();
            copy(ch + 2, ch + 1);
        }
        return;
    }
    if (tid >= GWALK) {   // the prep warps: chunk c + 1 while c is walked
        __syncthreads();
        gdn_prep(in[0], ob[0], px, tid - GWALK);
        for (int ch = 0; ch < nch; ++ch) {
            __syncthreads();
            if (ch + 1 < nch) gdn_prep(in[(ch + 1) & 1], ob[(ch + 1) % 3], px, tid - GWALK);
        }
        return;
    }
    const int g = lane >> 2, q = lane & 3, j = warp >> 1, head = kh + j * HK;
    const int cl = (warp & 1) * 16, cb = cs * GCW + cl;   // the warp's first column, in the block and in the head
    float s[S / 8][4];   // S^T: tile nt covers rows 8 nt .. 8 nt + 7
#pragma unroll
    for (int nt = 0; nt < S / 8; ++nt) {
        const int r = 8 * nt + 2 * q;
        s[nt][0] = state[((size_t) r * HV + head) * S + cb + g];
        s[nt][1] = state[((size_t) (r + 1) * HV + head) * S + cb + g];
        s[nt][2] = state[((size_t) r * HV + head) * S + cb + g + 8];
        s[nt][3] = state[((size_t) (r + 1) * HV + head) * S + cb + g + 8];
    }
    __syncthreads();
    const int wr = 8 * (lane >> 4) + (lane & 7), wc = 8 * ((lane >> 3) & 1);   // ldmatrix rows: W^T and Q^T's B
    const int ur = (lane & 7) + 8 * ((lane >> 3) & 1), uc = 8 * (lane >> 4);   // the update's
    const float rs = rsqrtf((float) S);
    for (int ch = 0; ch < nch; ++ch) {
        __syncthreads();
        const GdnOut& b = ob[ch % 3];
        const GdnV& vv = vb[ch & 1];
        const int t0 = ch * GCL, n = min(GCL, T - t0);
        const float* scj = b.tp[j].s;
        // W^T = S^T K^T and S^T Q^T [16 columns x GCL tokens]
        float w[GCL / 8][4] = {}, qs[GCL / 8][4] = {};
#pragma unroll
        for (int kt = 0; kt < S / 16; ++kt) {
            uint32_t a[4], kb[4], qb[4];
            ga_of(s[2 * kt], s[2 * kt + 1], a);
            gldsm(kb, &b.k[wr][16 * kt + wc]);
            gldsm(qb, &b.q[wr][16 * kt + wc]);
            gmma(w[0], a, kb[0], kb[1]);
            gmma(w[1], a, kb[2], kb[3]);
            gmma(qs[0], a, qb[0], qb[1]);
            gmma(qs[1], a, qb[2], qb[3]);
        }
        // B^T = beta (V^T - exp(gamma) W^T); element e of tile nt: column g + 8 (e >> 1), token 8 nt + 2 q + (e & 1)
        float bt[GCL / 8][4];
#pragma unroll
        for (int nt = 0; nt < GCL / 8; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int t = 8 * nt + 2 * q + (e & 1);
                bt[nt][e] = scj[2 * GCL + t] * (vv.v[j][t][cl + g + 8 * (e >> 1)] - scj[t] * w[nt][e]);
            }
        // D^T = B^T T^T, O^T = exp(gamma) S^T Q^T + D^T P^T
        uint32_t ab[4], ad[4];
        ga_of(bt[0], bt[1], ab);
        float d[GCL / 8][4] = {}, o[GCL / 8][4];
#pragma unroll
        for (int nt = 0; nt < GCL / 8; ++nt) {
            const __half* tr = &b.tp[j].t[8 * nt + g][2 * q];
            gmma(d[nt], ab, gld32(tr), gld32(tr + 8));
        }
        ga_of(d[0], d[1], ad);
        float* oc = out + (size_t) t0 * HV * S + head * S + cb + g;   // the chunk's first token, the lane's column
#pragma unroll
        for (int nt = 0; nt < GCL / 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) o[nt][e] = scj[8 * nt + 2 * q + (e & 1)] * qs[nt][e];
            const __half* pr = &b.tp[j].p[8 * nt + g][2 * q];
            gmma(o[nt], ad, gld32(pr), gld32(pr + 8));
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int t = 8 * nt + 2 * q + (e & 1);
                if (t < n) oc[t * HV * S + 8 * (e >> 1)] = o[nt][e] * rs;
            }
        }
        // S^T = exp(gamma_L) S^T + (exp(gamma_L - gamma) D^T) K
#pragma unroll
        for (int nt = 0; nt < GCL / 8; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) d[nt][e] *= scj[GCL + 8 * nt + 2 * q + (e & 1)];
        ga_of(d[0], d[1], ad);
        const float eL = scj[3 * GCL];
#pragma unroll
        for (int nt = 0; nt < S / 8; nt += 2) {
            uint32_t kb[4];
            gldsm_t(kb, &b.k[ur][8 * nt + uc]);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                s[nt][e] *= eL;
                s[nt + 1][e] *= eL;
            }
            gmma(s[nt], ad, kb[0], kb[1]);
            gmma(s[nt + 1], ad, kb[2], kb[3]);
        }
    }
#pragma unroll
    for (int nt = 0; nt < S / 8; ++nt) {
        const int r = 8 * nt + 2 * q;
        state[((size_t) r * HV + head) * S + cb + g] = s[nt][0];
        state[((size_t) (r + 1) * HV + head) * S + cb + g] = s[nt][1];
        state[((size_t) r * HV + head) * S + cb + g + 8] = s[nt][2];
        state[((size_t) (r + 1) * HV + head) * S + cb + g + 8] = s[nt][3];
    }
}

}  // namespace

void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream, int sides) {
    if (T <= 0 || sides < 1 || sides > 3) return;
    kv_append_kernel<<<dim3((unsigned) T, 2, sides == 3 ? 8 : 4), 64, 0, (cudaStream_t) stream>>>(
        K, V, pos0, page_table, page_size, k_pool, v_pool, k_q, v_q, k_scale, v_scale, sides);
    check("kv_append");
}
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("to_f16");
}
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_bf16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("to_bf16");
}

void q8_to_bf16(const int8_t* q, const uint16_t* scales, int64_t n, uint16_t* y, void* stream) {
    if (n <= 0) return;
    q8_to_bf16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(
        q, scales, n, y);
    check("q8_to_bf16");
}

void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream) {
    gr_norm_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, xn, xn16);
    check("gr_norm");
}
void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream) {
    gr_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, rs, xn16);
    check("gr_norm_rs");
}
void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
              int64_t T, void* stream, uint16_t* mixed_h) {
    gr_mix_r_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(R, rs, w_norm, gated, mixed, mixed16, T,
                                                                          mixed_h);
    check("gr_mix_r");
}
void gr_upmix(const uint16_t* lo16, const uint16_t* w_up, const float* R, const float* rs, const float* w_norm,
              float* mixed, uint16_t* mixed16, int64_t T, void* stream, uint16_t* mixed_h) {
    if (T <= 0) return;
    // per device: whether half its L2 holds the R tiles of two blocks an SM (0: not asked yet, 1: yes, 2: no)
    static std::atomic<int> pf_of[64];
    int dev = 0;
    cudaGetDevice(&dev);
    std::atomic<int>& pf = pf_of[dev & 63];
    if (pf.load() == 0) {
        int sms = 0, l2 = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
        pf.store((int64_t) sms * 2 * UM_BM * HC * UM_BD * 4 <= (int64_t) l2 / 2 ? 1 : 2);
    }
    const dim3 grid(N / UM_BD, (unsigned) ((T + UM_BM - 1) / UM_BM));
    const cudaStream_t s = (cudaStream_t) stream;
    if (pf.load() == 1) gr_upmix_kernel<true><<<grid, 256, 0, s>>>(lo16, w_up, R, rs, w_norm, mixed, mixed16, mixed_h, T);
    else gr_upmix_kernel<false><<<grid, 256, 0, s>>>(lo16, w_up, R, rs, w_norm, mixed, mixed16, mixed_h, T);
    check("gr_upmix");
}
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream, const uint16_t* partial) {
    gr_write_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, w_norm, eps,
                                                                                     rs, xn16, partial);
    check("gr_write_norm_rs");
}
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream) {
    gr_silu_kernel<<<blocks_for(T * LR), 256, 0, (cudaStream_t) stream>>>(lo, lo16, T * LR);
    check("gr_silu");
}
void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h) {
    gr_mix_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(xn, gated, mixed, mixed16, T, mixed_h);
    check("gr_mix");
}
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream,
              const uint16_t* partial) {
    gr_write_kernel<<<blocks_for(T * D), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, T, partial);
    check("gr_write");
}
void gr_broadcast(const float* e, float* R, int64_t T, void* stream) {
    gr_broadcast_kernel<<<blocks_for(T * D), 256, 0, (cudaStream_t) stream>>>(e, R, T);
    check("gr_broadcast");
}
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream) {
    gdn_gates_kernel<<<blocks_for(T * HV), 256, 0, (cudaStream_t) stream>>>(ab, dt, ssm_a, gate, beta, T);
    check("gdn_gates");
}
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream) {
    for (int64_t t0 = 0; t0 < T; t0 += 65535)
        strata::kernels::gdn_conv_l2_multi(history, qkv, conv_w, h, C, 2 * HK, eps, (int) std::min<int64_t>(65535, T - t0),
                                           stream, (int) t0);
    gdn_conv_hist_kernel<<<C / 256, 256, 0, (cudaStream_t) stream>>>(history, qkv, T);
    check("gdn_conv");
}
void gdn_scan(float* state, const float* h, const float* gate, const float* beta, float* y, int64_t T, void* stream) {
    if (T <= 0) return;
    static std::atomic<bool> smem_set[64] = {};   // 97 KB, opted in on each device (a split prompt path runs on two)
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64 || (!smem_set[dev].load() && cudaFuncSetAttribute(gdn_chunk_kernel,
                                                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                                                GDN_SMEM) != cudaSuccess)) {
        std::fprintf(stderr, "gdn_scan: the kernel's shared memory\n");
        std::exit(1);
    }
    smem_set[dev].store(true);
    gdn_chunk_kernel<<<dim3(HK, S / GCW), GWALK + GPREP + GLOAD, GDN_SMEM, (cudaStream_t) stream>>>(state, h, gate, beta,
                                                                                                   y, (int) T);
    check("gdn_scan");
}
void gdn_out_norm(float* y, const float* z, const float* gamma, float eps, uint16_t* y16, int64_t T, void* stream,
                  bool y32) {
    for (int64_t t0 = 0; t0 < T; t0 += 65535) {
        const int64_t n = std::min<int64_t>(65535, T - t0);
        const dim3 grid(HV, (unsigned) n);
        if (y32)
            gdn_norm_kernel<true><<<grid, S, 0, (cudaStream_t) stream>>>(y + t0 * HV * S, z + t0 * HV * S, gamma, eps,
                                                                         y16 + t0 * HV * S);
        else
            gdn_norm_kernel<false><<<grid, S, 0, (cudaStream_t) stream>>>(y + t0 * HV * S, z + t0 * HV * S, gamma, eps,
                                                                          y16 + t0 * HV * S);
    }
    check("gdn_out_norm");
}
void route(const float* logits, int32_t* ids, float* weights, int64_t T, void* stream) {
    route_kernel<<<(unsigned) ((T + 7) / 8), 256, 0, (cudaStream_t) stream>>>(logits, ids, weights, T);
    check("route");
}
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    swiglu_il_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(gu, h16, n);
    check("swiglu_interleaved");
}
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    swiglu_pair_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(g, u, h16, n);
    check("swiglu_pair");
}
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    gather_rows16_kernel<<<blocks_for(n * (width / 8)), 256, 0, (cudaStream_t) stream>>>(x16, src, dst16, n, width);
    check("gather_rows16");
}
void gather_rows16_f32(const uint16_t* x16, const int32_t* src, float* dst, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    gather_rows16_f32_kernel<<<blocks_for(n * (width / 8)), 256, 0, (cudaStream_t) stream>>>(x16, src, dst, n, width);
    check("gather_rows16_f32");
}
__global__ void sums_to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = hf(fminf(fmaxf(x[i], -65504.0f), 65504.0f));
}
void sums_to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    sums_to_f16_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("sums_to_f16");
}
void moe_shared_gate(float* x, const float* sg, int64_t T, void* stream) {
    moe_shared_gate_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(x, sg, T);
    check("moe_shared_gate");
}
__global__ void moe_gather_add_kernel(float* __restrict__ sum, const float* __restrict__ rows, int64_t r0,
                                      const float* __restrict__ w, const int32_t* __restrict__ tok,
                                      const int32_t* __restrict__ start, const int32_t* __restrict__ list) {
    const int64_t d = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= N) return;
    const int b = blockIdx.y;
    float* o = sum + (int64_t) tok[b] * N + d;
    float s = *o;
    for (int i = start[b]; i < start[b + 1]; ++i) {
        const int r = list[i];
        s = fmaf(w[r], rows[(r - r0) * N + d], s);
    }
    *o = s;
}
void moe_gather_add(float* sum, const float* rows, int64_t r0, const float* w, const int32_t* tok, const int32_t* start,
                    const int32_t* list, int64_t n_tok, void* stream) {
    if (n_tok <= 0) return;
    moe_gather_add_kernel<<<dim3((unsigned) ((N + 255) / 256), (unsigned) n_tok), 256, 0, (cudaStream_t) stream>>>(
        sum, rows, r0, w, tok, start, list);
    check("moe_gather_add");
}
void moe_scatter_add(float* sum, const float* rows, const float* w, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    moe_scatter_add_kernel<<<blocks_for(n * N), 256, 0, (cudaStream_t) stream>>>(sum, rows, w, src, n);
    check("moe_scatter_add");
}
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream) {
    if (rows <= 0) return;
    rms_rows_kernel<<<(unsigned) rows, 256, 0, (cudaStream_t) stream>>>(x, w, cols, ld, eps);
    check("rms_rows");
}
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
          const strata::kernels::RopeScaling& scaling, void* stream) {
    using namespace strata::kernels;
    const float theta_scale = powf((float) scaling.freq_base, -2.0f / 64.0f);
    const RopeKernelArgs k = scaling.kernel_args(64);
    RopeTab rt;
    const int mode = rope_mode(scaling, rt);
    const unsigned grid = (unsigned) (T * heads);
    const cudaStream_t st = (cudaStream_t) stream;
    if (mode == ROPE_TABLE)
        rope_kernel<ROPE_TABLE><<<grid, 32, 0, st>>>(x, heads, dim, ld, pos0, theta_scale, mrope_table(), k, rt);
    else if (mode == ROPE_SCALED)
        rope_kernel<ROPE_SCALED><<<grid, 32, 0, st>>>(x, heads, dim, ld, pos0, theta_scale, mrope_table(), k, rt);
    else
        rope_kernel<ROPE_PLAIN><<<grid, 32, 0, st>>>(x, heads, dim, ld, pos0, theta_scale, mrope_table(), k, rt);
    check("rope");
}
void split_q(const float* q_full, float* q, int64_t T, void* stream) {
    split_q_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(q_full, q, T);
    check("split_q");
}
void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream) {
    gate_attn_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(attn, q_full, out16, T);
    check("gate_attn");
}

}  // namespace strata::prefill
