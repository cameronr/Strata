# Phase 4, segment-table kernels

Back-link: [overview.md](overview.md).

## Goal

One launch covers all rows of a multi-sequence window where the per-segment loop of phase 3 was slow: QSA block scores, top-k and attention; GDN conv and recurrence; MoE row processing. Bitwise equal to the per-segment path.

## Changes

Port the design from Niko's `STRATA_QSA_SLOTS` single-launch work (`src/kernels/qsa_slots_parity.cpp` and the verify.cpp call sites on niko/main as reference), writing our own sm_86-compatible kernels from our existing single-sequence kernels plus the segment table.

- `src/kernels/cuda/qsa_select.cu`, `qsa_decode_attn.cu`, decode indexer: read `SegRow` so queries, top-k and attention resolve each row's sequence, page tables per segment.
- GDN conv/recurrence kernels: per-segment state pointers from the table.
- MoE decode path: rows already carry their plan; segment ids ride along.
- Parity suites in the style of Niko's `qsa_slots_parity`: single-launch multi-seq bitwise equals per-segment runs, on both cards, skipped where a launcher declines by design.

## Data structures

No new contract, the phase-3 `SegRow` table. Kernels index it; no per-kernel hidden copies of it.

## Verification

Static: build, ctest, the new parity suites, phase-1 digest gate.
Runtime: two-sequence window microbench (per-segment loop vs single launch) on both cards; the phase-3 token-stream test still passes.
