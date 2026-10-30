// include/lamina/kernels/native_mma.hpp - the verify window's dense projections of 2-8 tokens on the int8 tensor
// cores (src/kernels/cuda/native_mma.cu): a CTA's 16-row tiles of the GGUF blocks (Q6_K, Q5_K, Q4_K, IQ4_XS) copied
// into shared memory a stage ahead and decoded once into mma's A, the window's tokens its N.
#pragma once

namespace lamina::kernels {

// y = W x for `ncols` (2-8) columns of n_in quantized by native_quantize_q8_1_il (x_il, its interleaved copy), where a
// table measured on the RTX 3090 puts that format, row count and column count on the tensor cores (each format from a
// column count by its rows); false (nothing launched): the dp4a kernels take it (native_mmvq_il). A column's values
// do not depend on ncols; they are native_mmvq's to float rounding (the integer sums exact, the scales applied per q8_1
// block in another order). Weights: GGUF blocks, Q6_K's rows at 4-byte and the others' at 16-byte multiples (else
// false); y: column j at y + j * n_out.
bool native_mma(int ggml_type, const void* weights, const void* x_il, float* y, int n_in, int n_out, int ncols,
                void* stream);

}  // namespace lamina::kernels
