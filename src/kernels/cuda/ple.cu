// src/kernels/cuda/ple.cu - P2.S4: the PLE block's GPU half.
//
// See include/lamina/kernels/ple.hpp for the structure and for the `normalized`-is-the-conv-input finding.
//
// The legacy arithmetic below follows the captured CPU ggml graph
// (`bench/micro/ple_in.bin` / `ple_out.bin`, produced by `ple_layer_xcheck.cpp`):
//
//   * `rms_norm` accumulates `(double)(x*x)` with the PRODUCT rounded in f32 first - that is literally what
//     `ggml_compute_forward_rms_norm_f32` does (`sum += (ggml_float)(x[i00]*x[i00])` with `ggml_float` =
//     double), and it is not the same as accumulating in f32 or as widening before the multiply.
//   * `silu` is `x / (1 + expf(-x))` in f32, which is `ggml_silu_f32`.
//   * the gate's `sqrt` and `sigmoid` are f32.
//
// The opt-in native BF16 path replaces only ple_value with the pinned CUDA single-token BF16/F32 MMVF.
// A separate opt-in native postops path follows the pinned CUDA arithmetic after both projections.
#include "lamina/kernels/ple.hpp"
#include "lamina/kernels/bf16_bits.hpp"
#include "lamina/kernels/bf16_gemv.hpp"
#include "lamina/kernels/f16_bits.hpp"
#include "lamina/kernels/ngram.hpp"
#include "lamina/kernels/quantize_act.hpp"
#include "lamina/kernels/s2_gemv_q8.hpp"
#include "lamina/kernels/native_mmvq.hpp"
#include "lamina/kernels/native_ple_postops.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace lamina::kernels {
namespace {

constexpr int THREADS = 256;

bool overlap(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
    if (a == nullptr || b == nullptr || a_bytes == 0 || b_bytes == 0) return false;
    const uintptr_t aa = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
    return aa < bb ? bb - aa < a_bytes : aa - bb < b_bytes;
}

__global__ void history_advance_kernel(float* __restrict__ history, const float* __restrict__ normalized) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= NG_HC_DIM) return;
    float* column = history + (size_t) channel * NG_HIST;
    for (int row = 0; row + 1 < NG_HIST; ++row) column[row] = column[row + 1];
    column[NG_HIST - 1] = normalized[channel];
}











void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ple_block: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace


void ple_history_advance(float* hist, const float* normalized, void* stream) {
    if (hist == nullptr || normalized == nullptr)
        throw std::invalid_argument("ple_history_advance: null history or normalized input");
    if (overlap(hist, (size_t) NG_HIST * NG_HC_DIM * sizeof(float),
                normalized, (size_t) NG_HC_DIM * sizeof(float)))
        throw std::invalid_argument("ple_history_advance: history and normalized input overlap");
    history_advance_kernel<<<(NG_HC_DIM + THREADS - 1) / THREADS, THREADS, 0, (cudaStream_t) stream>>>(hist, normalized);
    ck(cudaGetLastError(), "history advance launch");
}




}  // namespace lamina::kernels
