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

## As built (commits `1075d204a` `8cc45020f` `70b286fa0` `484e3f511` `7f25b92d3`)

The design arena (`.p4-arena/synthesis.md`, scratch) ported the resident-table shape of Niko's `STRATA_QSA_SLOTS` work onto our kernels:

- `include/strata/kernels/seg_rows.hpp`: the `SegRow` device table (per-row state pointers + per-slot pools, `SegWritePools`
  contract) and the staging rule.  Tables are resident per (layer, slot) from `init_slots`; a window stages only an int32
  row -> slot index (`h_seg_idx_` -> `m_seg_idx_` -> `seg_idx_`), never pointers, so no graph bakes slot identity.
- One-launch batch forms: `qsa_block_scores_seg` + `qsa_block_topk`, `qsa_kv_append_steps_seg`, `native_qsa_indexer_append_multi_seg`,
  `qsa_decode_attn_seg`, GDN conv/step over rows, `native_ple_postops_tokens_seg`.  Each is the one-token window's own
  `__device__` body with a row index: bitwise equality is by program identity (same body, same grid dims, same derived scalars),
  never by cross-kernel numeric comparison.  Top-k reuses the per-row launcher but pins its chunk count to the nq = 1 value
  (`qsa_block_topk_nch(1, ...)`, an override the fused call passes): letting nq = S derive nch would shrink the threshold
  histogram's granularity whenever `resident/S < want`, which is selection-visible at ties.
- PLE: every batch row is its own chunk start, so conv history taps read at `3k`, not `row + 3k`; the fixed seg kernel passes
  per-row bases plus the local index into the shared body.
- MoE: no change needed.  Rows already resolve experts through the drive plan (phase 3); no segment tables enter the expert path.
- `init_slots` refuses F16 KV pools for batch slots (the seg append has no F16 form) and `qsa_kv_append_steps_seg` aborts if one
  ever reaches it; the attention call site probes `qsa_attn_pools(st)` for the format only (init_slots enforces one format across slots).
- Graph keying: fused batch graphs key on S alone (`exec_bs_[9]`, captured once per window size).  `STRATA_BATCH_FUSE=0`
  (file-static `kBatchFuse`, read once) records the phase-3 per-row launches at the six batch call sites instead; those bake
  the layout at capture, so the fallback keeps the layout-keyed `exec_bm_`/`brow_`/`kMaxBatchGraphs` path.  A bisection and
  microbench tool, not a production mode; solo windows never reach either branch.
- Parity suites (`src/kernels/`): `qsa_sel_seg_parity`, `qsa_kv_seg_parity` (three quantized formats), `gdn_seg_parity`
  (indexer, GDN, PLE).  Each drives permuted row -> slot index tables against per-row runs on both cards, exits 77 where a
  launcher declines by design, and honors a `STRATA_P4_ONLY` filter.
- Parity script mode (`STRATA_BATCH_PARITY`) gained `SHUFFLE 1`: rotate the row -> slot map every window, feed included.
  `tools/batch_parity_test.py --shuffle` drives it and `--extra` passes sizing overrides.

## Validation (2026-10-27, branch `batch-phase1-harness`, 4090 + 3090 rig)

| check | result |
| --- | --- |
| three seg parity suites, both cards (sm_86 + sm_89) | passed (permuted index tables, per-device loop) |
| SASS identity, cuobjdump per-function diff, sm_86 + sm_89 | passed - 14/14 file x arch comparisons, 46 functions in 7 files byte-identical to the pre-phase-4 baselines |
| solo digest gate, fresh processes A + B, deterministic config + frozen profile | passed - `DIGEST-A-IDENTICAL`, `DIGEST-B-IDENTICAL` against `digests-golden.txt` |
| `batch_parity_test.py --n 2 --max-new 48 --context 8192` (identity map) | passed - both slots equal, 48 tokens |
| same, `--shuffle` (rotated row -> slot map), S=2 | passed - `SHUFFLE-ENGAGED` (engine log confirms rotation ran) |
| same, `--shuffle`, S=8 (`--max-context 2048 --expert-cache 4000`) | passed - all 8 slots equal, eight rotations exercised, one S=8 graph |
| microbench, wall time of the parity leg, deterministic config | fused +0.3% at S=2 ctx 8192 (27.2 s vs 27.1 s, noise) and +1.5% at S=8 ctx 2048 (54.8 s vs 54.0 s) - matches the synthesis threshold: no regression at S=2, win at S=8 |
| audit round (architect + correctness review of `4a17e225f..0e41a9ebc`), then re-gate after the hardening commit | passed - suites 3/3 (indexer case now at the engine's pos_stride, strict skip/empty semantics), SASS identity re-run, digests identical, identity + shuffled S=2 and S=8 parity with the rotation marker enforced in the committed harness |
| ctest | not run (tree builds `STRATA_BUILD_TESTS=OFF`, as phase 3) |

## Findings

- **Program identity is the only workable bitwise gate.**  Body extraction into shared `__device__` functions must not add
  `__restrict__` the old kernel lacked: it changes LDG `.CONSTANT` hints and the SASS differs (`conv_residual_tokens_kernel`,
  caught by the gate, reverted).  The SASS gate compares HEAD baselines compiled under the same base filenames.
- **GDN state is context-independent** (~0.11 GiB/slot): eight slots fit at ctx 2048 only by dropping the expert cache
  4800 -> 4000; the interleave-style three-slot run at 8192 needed 4400.  Slot sizing talks should quote both numbers.
- **The layout-varying runtime check needed a new knob**: `batch_interleave_test.py` drives upstream's BGEN slot server,
  which is not in this lineage, and the parity script was identity-only.  `SHUFFLE 1` is the direct test of the S-only key:
  with rotation on, a stale layout-baked graph cannot survive 48 windows, and the streams must (and do) equal solo.
- **Microbench is wall-time A/B** (no BGEN slot server here): window GPU time is dominated by expert reads, so launch-count
  savings are noise at S=2 and ~1.5% of wall at S=8.  The constant-node-count win (graph replay CPU) is hidden in the GPU
  pipeline at these sizes; it matters to the phase-5 scheduler, which pays replay CPU per window across churn.
- **A passing shuffle leg must prove it shuffled**: the gate greps the engine log for the rotation marker, so a silently
  unparsed `SHUFFLE` directive fails the gate instead of degrading to an identity run.  The check lives in the committed
  harness (`--shuffle`), not only in the scratch gate script.
- **The audit round caught one site where "program identity" was empirical**: top-k derived its chunk count from nq, so
  the fused nq = S call could histogram at a different granularity than the per-row calls it replaces (selection-visible
  at ties; equal on the rig's cards only because resident >> want there).  Pinned to the nq = 1 value, the same override
  the suite now uses - the taxonomy is structural again at every fused site.
