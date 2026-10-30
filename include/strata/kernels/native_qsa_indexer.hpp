#pragma once

#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/qsa.hpp"

namespace strata::kernels {

void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device,
                               int32_t pos_base, const float* gamma, float epsilon,
                               const QsaIndexerBuffers& buffers, const QsaShapes& shapes,
                               int64_t max_cells, const RopeScaling& scaling, void* stream);
// n_tok appends in one launch, in order: token t's raw key at raw + t * 128, its position at
// relative_pos_device[t * pos_stride].  Bitwise n_tok calls of native_qsa_indexer_append.  With `tail_snap`
// (3 * 128 floats) the key tail as it was before them is copied there first.
void native_qsa_indexer_append_multi(const float* raw, const int32_t* relative_pos_device, int pos_stride,
                                     int n_tok, int32_t pos_base, const float* gamma, float epsilon,
                                     const QsaIndexerBuffers& buffers, const QsaShapes& shapes,
                                     int64_t max_cells, const RopeScaling& scaling, void* stream,
                                     float* tail_snap = nullptr);

} // namespace strata::kernels
