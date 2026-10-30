// include/lamina/kernels/draft_proxy.hpp - the drafter's head argmax through a 2-bit copy of its rows (basalt 29e7176's
// two-stage argmax, for the Q6_K and Q8_0 heads): every row is scored from the copy (32-weight groups of four levels
// and a half scale: 800 bytes a row of 2560 against Q6_K's 2100 and Q8_0's 2720), the rows whose score is near the
// best are scored exactly (bitwise native_mmvq's logits), and the exact best is the draft.  Its softmax probability
// takes the rescored rows' exact terms and the others' approximate ones.
#pragma once

#include <cstddef>
#include <cstdint>

namespace lamina::kernels {

/// The copy's rows: n_in 2560 (the model's width) and the head types 14 (Q6_K) and 8 (Q8_0).
bool draft_proxy_supported(int ggml_type, int n_in) noexcept;
std::size_t draft_proxy_bytes(int64_t n_rows);
/// Rows [0, n_rows) of `rows` (native_mmvq's layout of `ggml_type`) into the copy.
void draft_proxy_pack(int ggml_type, const void* rows, int64_t n_rows, void* proxy, void* stream);

/// The argmax row of rows . x (one q8_1 column), as `map[row]` (the token id), and its softmax probability over the
/// rows (up to INT32_MAX of them).  `scratch`: draft_proxy_scratch_bytes(n_rows), zeroed once before the first launch.
/// Graph-capturable.
std::size_t draft_proxy_scratch_bytes(int64_t n_rows);
void draft_proxy_top1(int ggml_type, const void* proxy, const void* rows, const void* x_q8_1, int64_t n_rows,
                      const int32_t* map, int32_t* out_id, float* out_prob, void* scratch, void* stream);

}  // namespace lamina::kernels
