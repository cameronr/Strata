// include/strata/kernels/qsa_select.hpp - plan v0.3 P5/P7: the QSA indexer's block scores and top-k selection,
// for many queries at once and at long context.
//
// The per-token path scores every pooled block with one FP64 block per row (qsa_index_kernel) and selects with a
// single-block kernel that makes 32 bit-serial passes over every CELL (topk_kernel): ~0.1-0.2 ms per query at 32K,
// once per QSA layer and token - 1-2 ms of a decode token and most of a 32K prompt's time.  Here:
//
//   qsa_block_scores : FP32; relu per indexer head, summed; block n_bid (the incomplete tail) scores the `dead` key
//                      and gets +1e9 when it has cells - which is exactly what the per-token path reads there,
//                      because pooled[n_bid] holds `dead` at that time (in a chunk it may already hold a block
//                      completed later, so it is not read).  Each score sums in the order one warp per
//                      (query, block) did.  Decode windows (up to 8 queries): a warp per key block, the queries'
//                      leaves in registers, the warp's sums by recursive halving (9-11 queries: an 8-lane group per
//                      key block, the queries through shared memory).  Batches: a per-thread GEMM form, a lane's 2
//                      queries x 4 key blocks over all heads, the tree on a stack of partial sums, the queries read
//                      by the whole warp at once; one persistent block an SM with an equal share of the (key tile,
//                      query chunk) units.
//   qsa_block_topk   : a 4-pass radix select over the query's n_bid + 1 blocks, each weighted by its cell count, then
//                      the cells emitted in ascending order with ties to the lowest index - the same selection as
//                      topk_kernel, over a quarter of the elements.  Up to 8 queries (decode windows) in one launch
//                      over many 256-thread blocks, a query's blocks in chunks that meet at barriers in device
//                      memory (calls on a GPU must not overlap); more queries: one 1024-thread block per query.
//
// Queries carry their own step record (pos, n_kv, n_bid, width) as everywhere else in QSA.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

struct SegRow;   // the batch window's segment table entry (seg_rows.hpp)

/// scores [nq, max_blocks]; q_idx [nq, idx_n_head, idx_dim] (normed and rotated); steps [nq, kStepCount].
/// The grid covers the first `grid_blocks` blocks and goes round for the rest: a captured graph passes a fixed size
/// (qsa_score_grid_blocks), the prompt path the batch's largest n_bid + 1.
void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t grid_blocks);

/// The block-score grid of the decode paths, whatever the context: 164 thread blocks of 64 key blocks, one wave on
/// the RTX 3090 (two per SM); longer contexts go round.
constexpr int64_t qsa_score_grid_blocks = 10496;

/// The batch window's scores in ONE launch (seg_rows.hpp): row r of [0, n) scores the keys of its own slot, read
/// as `seg[row_slot[r]]` - the resident segment table and the window's row -> slot index, both at fixed addresses.
/// Row r's program is the per-row `qsa_block_scores(nq = 1)` call's exactly (same body, same grid.x): the per-row
/// launcher always picks block_scores_kernel at nq=1, so there is no decline path. `row_slot` entries are
/// validated by the caller (Verifier::run_slot_rows).
void qsa_block_scores_seg(const SegRow* seg, const int32_t* row_slot, const float* q_idx, const int32_t* steps,
                          int64_t n, int64_t max_blocks, const QsaShapes& s, float* scores, void* stream,
                          int64_t grid_blocks);

/// The most cells qsa_block_topk selects over on the current device, --max-context's limit: its tables take 512 bytes
/// of shared memory per 4,096 cells (1,466,368 on the RTX 3090's 99 KiB, 630,784 on a 48 KiB card).
int64_t qsa_select_max_cells(const QsaShapes& s);

/// ids [nq, cap] (cells, ascending); `cap` >= the largest selection width.
void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream);

}  // namespace strata::kernels
