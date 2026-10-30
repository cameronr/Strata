// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// src/models/qwen4exp.cpp; ggml/src/ggml-cuda/{set-rows.cu,norm.cu,rope.cu}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lamina/kernels/native_qsa_indexer.hpp"
#include "lamina/kernels/warp.cuh"
#include "lamina/kernels/mrope.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace lamina::kernels {
namespace {
constexpr int D = 128, R = 4, ROT = 64, THREADS = 256;
// One token's append; every thread of the block calls it (it synchronizes the block), a later call sees this one's
// tail writes.
template <int MODE>
__device__ void append_one(const float* __restrict__ raw, const int pos,
                           int pos_base, const float* __restrict__ gamma, float epsilon,
                           float* __restrict__ tail, float* __restrict__ dead,
                           float* __restrict__ pooled, int32_t* __restrict__ block_pos,
                           int max_cells, float theta_scale, const int32_t* __restrict__ mtab,
                           const RopeKernelArgs& k, const RopeTab& rt) {
    const int d = threadIdx.x;
    if (pos < 0 || pos >= max_cells) return;
    const int slot = pos % R;
    float incoming = 0.0f;
    if (d < D) {
        // SET_ROWS stores F16; GET_ROWS expands those exact values to F32.
        incoming = __half2float(__float2half_rn(raw[d]));
        if (slot < R - 1) tail[slot * D + d] = incoming;
    }
    if (pos != 0 && slot != R - 1) return;
    __shared__ float values[D];
    __shared__ float partials[32];
    float mean = 0.0f;
    if (d < D) {
        // The spare's four gather indices all name cell zero. Completed blocks
        // use chronological slices; each graph ADD materializes an F32 sum.
        float sum = pos == 0 ? incoming : tail[d];
#pragma unroll
        for (int j = 1; j < R; ++j)
            sum = __fadd_rn(sum, pos == 0 || j == R - 1 ? incoming : tail[j * D + d]);
        mean = __fmaf_rn(0.25f, sum, 0.0f); // SCALE includes a zero bias.
    }
    float square_sum = 0.0f;
    if (d < D) square_sum += mean * mean;
    square_sum = warp_sum(square_sum);
    const int lane = d % 32;
    if (lane == 0) partials[d / 32] = square_sum;
    __syncthreads();
    square_sum = lane < THREADS / 32 ? partials[lane] : 0.0f;
    square_sum = warp_sum(square_sum);
    const float scale = rsqrtf(square_sum / D + epsilon);
    if (d < D) values[d] = scale * mean * gamma[d];
    __syncthreads();
    if (d < D) {
        const int b = pos / R;
        const int rope_pos = pos == 0 ? 0 : pos_base + R * b;
        float y = values[d];
        if (d < ROT) {
            const int pair = d % (ROT / 2);
            float c, s;
            rope_cs<MODE>(rt, k, theta_scale, pos == 0 ? 0 : mrope_pos(mtab, rope_pos, pair), pair, c, s);
            const float a = values[pair], z = values[pair + ROT / 2];
            y = d < ROT / 2 ? a * c - z * s : a * s + z * c;
        }
        pooled[std::size_t(b) * D + d] = y;
        if (pos == 0) dead[d] = y;
        else pooled[std::size_t(b + 1) * D + d] = dead[d];
        if (d == 0 && pos != 0) *block_pos = rope_pos;
    }
    __syncthreads();   // values and partials are reused by the next call
}
// With `tail_snap`, the tail as it was is copied there first.
template <int MODE>
__global__ void append(const float* __restrict__ raw, const int32_t* __restrict__ pos_dev, int pos_stride,
                       int n_tok, int pos_base, const float* __restrict__ gamma, float epsilon,
                       float* __restrict__ tail, float* __restrict__ dead,
                       float* __restrict__ pooled, int32_t* __restrict__ block_pos,
                       int max_cells, float theta_scale, const int32_t* __restrict__ mtab,
                       float* __restrict__ tail_snap, RopeKernelArgs k, RopeTab rt) {
    if (tail_snap != nullptr) {
        for (int i = threadIdx.x; i < (R - 1) * D; i += blockDim.x) tail_snap[i] = tail[i];
        __syncthreads();
    }
    for (int t = 0; t < n_tok; ++t) {
        append_one<MODE>(raw + std::size_t(t) * D, pos_dev[std::size_t(t) * pos_stride], pos_base, gamma, epsilon,
                         tail, dead, pooled, block_pos, max_cells, theta_scale, mtab, k, rt);
        __syncthreads();   // this token's tail and pooled writes before the next token reads them
    }
}
// ---- C-2: the batched append.  The pooled key of a completed block b (its last cell pos = 4b+3), computed with
// the single append's arithmetic: keys rounded through F16, summed tail[0]+tail[1]+tail[2]+incoming in that order,
// the RMS norm, gamma, the rotation at pos_base + 4b.  `k(j)` is the key of the block's cell j: a row of this batch
// (cell >= p0) or the tail the previous batch left (cell < p0).  Only the batch's last completed block writes the
// spare after it (every earlier one's is overwritten by the next block in order).
template <int MODE>
__device__ __forceinline__ float pooled_value(const float* values, int d, int rope_pos, float theta_scale,
                                              const int32_t* mtab, bool zero_pos, const RopeKernelArgs& k,
                                              const RopeTab& rt) {
    float y = values[d];
    if (d < ROT) {
        const int pair = d % (ROT / 2);
        float c, s;
        rope_cs<MODE>(rt, k, theta_scale, zero_pos ? 0 : mrope_pos(mtab, rope_pos, pair), pair, c, s);
        const float a = values[pair], z = values[pair + ROT / 2];
        y = d < ROT / 2 ? a * c - z * s : a * s + z * c;
    }
    return y;
}
struct Span { const void* p; std::size_t n; };
void validate(Span s) {
    const auto p = reinterpret_cast<std::uintptr_t>(s.p);
    if (!p || p % 4 || s.n > UINTPTR_MAX - p)
        throw std::invalid_argument("native QSA indexer requires aligned bounded spans");
}
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<std::uintptr_t>(a.p), y = reinterpret_cast<std::uintptr_t>(b.p);
    return x < y + b.n && y < x + a.n;
}
} // namespace

void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device, int32_t pos_base,
                               const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                               const QsaShapes& s, int64_t max_cells, const RopeScaling& scaling, void* stream) {
    native_qsa_indexer_append_multi(raw, relative_pos_device, 0, 1, pos_base, gamma, epsilon, b, s, max_cells,
                                    scaling, stream);
}
void native_qsa_indexer_append_multi(const float* raw, const int32_t* relative_pos_device, int pos_stride,
                                     int n_tok, int32_t pos_base, const float* gamma, float epsilon,
                                     const QsaIndexerBuffers& b, const QsaShapes& s, int64_t max_cells,
                                     const RopeScaling& scaling, void* stream, float* tail_snap) {
    if (n_tok < 1 || pos_stride < 0 || (n_tok > 1 && pos_stride < 1))
        throw std::invalid_argument("native QSA indexer: n_tok >= 1 appends with a positive position stride");
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT ||
        max_cells < 1 || max_cells > INT32_MAX || pos_base < 0 || pos_base % R ||
        int64_t(pos_base) + max_cells > INT32_MAX || !std::isfinite(epsilon) || epsilon <= 0.0f ||
        rope_scaling_invalid(scaling) != nullptr)
        throw std::invalid_argument("native QSA indexer requires fixed geometry, aligned position base, positive capacity/epsilon, valid frequency and explicit stream");
    const Span spans[] = {{raw,std::size_t(n_tok)*D*4},{relative_pos_device,(std::size_t(n_tok-1)*pos_stride+1)*4},
        {gamma,D*4},{b.tail,(R-1)*D*4},
        {b.dead,D*4},{b.pooled,std::size_t(max_cells/R+1)*D*4},{b.block_pos,4}};
    for (const auto& span : spans) validate(span);
    for (int i = 0; i < 7; ++i) for (int j = i + 1; j < 7; ++j)
        if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA indexer buffers overlap");
    if (tail_snap != nullptr) {
        const Span snap{tail_snap, (R - 1) * D * 4};
        validate(snap);
        for (const auto& span : spans)
            if (overlaps(snap, span)) throw std::invalid_argument("native QSA indexer: the tail snapshot overlaps");
    }
    const float theta_scale = powf((float) scaling.freq_base, -2.0f / ROT);
    const RopeKernelArgs k = scaling.kernel_args(ROT);
    RopeTab rt;
    const int mode = rope_mode(scaling, rt);
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    if (mode == ROPE_TABLE)
        append<ROPE_TABLE><<<1,THREADS,0,st>>>(raw,relative_pos_device,pos_stride,n_tok,pos_base,gamma,
            epsilon,b.tail,b.dead,b.pooled,b.block_pos,int(max_cells),theta_scale,mrope_table(),tail_snap,k,rt);
    else if (mode == ROPE_SCALED)
        append<ROPE_SCALED><<<1,THREADS,0,st>>>(raw,relative_pos_device,pos_stride,n_tok,pos_base,gamma,
            epsilon,b.tail,b.dead,b.pooled,b.block_pos,int(max_cells),theta_scale,mrope_table(),tail_snap,k,rt);
    else
        append<ROPE_PLAIN><<<1,THREADS,0,st>>>(raw,relative_pos_device,pos_stride,n_tok,pos_base,gamma,
            epsilon,b.tail,b.dead,b.pooled,b.block_pos,int(max_cells),theta_scale,mrope_table(),tail_snap,k,rt);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace lamina::kernels
