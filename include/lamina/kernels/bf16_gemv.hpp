// include/lamina/kernels/bf16_gemv.hpp - the GEMV for a BF16 weight against FP32 activations: llama.cpp's CUDA
// MMVF, one row or several (native_bf16.cu).  docs/activation-contract.md: why the activation is FP32.
#pragma once

#include <cstdint>

namespace lamina::kernels {

/// Single-token, contiguous 2D BF16-weight/F32-activation MMVF from the pinned llama.cpp CUDA path.
/// Preserves pair accumulation, adaptive block size and warp-XOR reduction order.
/// `w[o*n_in+i]` is BF16 bits; x/y are F32. n_in must be positive and even, n_out positive, both <= INT_MAX.
/// x must be 8-byte aligned, w 4-byte aligned, and y 4-byte aligned. Throws on invalid geometry/pointers.
/// The device buffers must not overlap; allocation bounds remain the caller's responsibility.
/// No allocations or synchronization, including when stream is null (the CUDA default stream).
void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y,
                         int64_t n_in, int64_t n_out, void* stream);
/// `bf16_gemv_fp32_mmvf` for n_tok (1..8) activation rows x (n_tok, n_in) -> y (n_tok, n_out), each weight read once;
/// every row bitwise what the single-row call gives.  With `w2` the same launch also gives y2 (n_tok, n_out2) from a
/// second matrix of the same input.
void bf16_gemv_fp32_mmvf_multi(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, int n_tok,
                               void* stream, const uint16_t* w2 = nullptr, float* y2 = nullptr, int64_t n_out2 = 0);

}  // namespace lamina::kernels
