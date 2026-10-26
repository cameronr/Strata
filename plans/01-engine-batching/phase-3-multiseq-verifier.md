# Phase 3, multi-sequence verify windows

Back-link: [overview.md](overview.md).

## Goal

The verifier runs and commits windows whose rows are segments of different sequences, correctly before fast. Kernels may run per segment in a loop; the single-launch fusion is phase 4.

## Changes

- `include/strata/core/verify.hpp` + `src/core/verify.cpp`: add the batch API in Niko's shape (`init_slots`, `run_slot_segs(slots, lens, S, ...)`, `commit_slots(n_keep[])`), re-derived against our window code. A window is S row segments; graph keying stays by row count, the segment split is staged data.
- Commit becomes per-segment with exact rollback to the previous window's state (our async commit beside the draft must order against per-segment keeps).
- `include/strata/core/batch_rows.hpp` (orphaned, has an `init_slots` sketch) is either extended to match or deleted in favor of the real thing. Do not leave both.
- Parity test: a two-sequence window equals two solo runs token-for-token, driven by a test harness, not the server.

## Data structures

`SegRow { slot, row_in_slot }` staged table; per-segment commit records (positions, KV lengths, GDN state snapshots or redo info). Name the shape before coding: the window row buffer plus the segment table is the whole contract between scheduler and kernels.

## Verification

Static: build, ctest, the new two-sequence parity test, phase-1 digest gate unchanged for solo.
Runtime: a test program feeds two prompts' tokens as interleaved windows and diffs the token streams against two solo runs. Run it on both cards.

## Risk note from phase 1 (measured 2026-10-26, resolved)

Greedy output on this branch's engine is not reproducible run to run with default flags: expert-cache adaptation
re-tiers experts mid-run and the tiers round differently (bisect: adapt default FAIL x2, `--adapt-every 0` PASS x2,
plus `STRATA_IQ_MT_MIN=1` throughout). Resolved for gating: every parity and digest gate in this plan runs under
`--adapt-every 0` and `STRATA_IQ_MT_MIN=1`, where the solo path is bitwise stable. The multi-sequence equality gate
in this phase is therefore meaningful as written, in deterministic mode.
