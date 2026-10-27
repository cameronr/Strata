# Phase 3, multi-sequence verify windows

Back-link: [overview.md](overview.md).

## Prerequisite, done: the deterministic gate config (2026-10-27)

The pinned `--expert-cache` alone did NOT make solo runs reproducible across processes: golden-vs-golden2
drifted at req 4. Root cause (audit of `generate.cpp`/`expert_source.cpp`): `--expert-cache N` is a byte
BUDGET, capped at startup by live free VRAM (`cap = min(budget, free_room)`, generate.cpp:2227-2241), the
second GPU's expert tier is sized from free VRAM unless `--second-gpu-gib` pins it, the GPU2 prefetch tier
(default 12 MiB/layer) is fed only when a prediction lands inside a 50 ms wall-clock wait (verify.cpp:1318
-1332; `expert_source.cpp:324-331` routes prefetched experts through GPU2 instead of the CPU, rounding
differently), and the prompt path's per-prompt slot loan inherits all of it.

The gate config (`.ph3-det.json`, scratch, from `.ph1-e2e.json`): `--expert-cache 4800` (below the free-room
cap so the budget always binds and residency is arg-determined), `--second-gpu-gib 2`, `--second-gpu-prefetch-mb 0`,
`--no-prefill-borrow`, plus the existing `--adapt-every 0`, `--pcie-frac 0`, `STRATA_IQ_MT_MIN=1`.
Measured: 6 requests x 4 fresh processes (two `window_digest_test` runs), token streams identical across all
of them (digests-detA/B). Upstream's docs/BATCHING.md Exactness section lists the same shape of settings; the
VRAM-budget cap and the wall-clock prefetch are the parts our rig hits.

Second hole (found when the phase-3 binary's digest drifted from the detA golden at req 1 win 5, though the
solo code path was byte-identical): **the `--expert-profile` file is a mutable input**. The prod server
(`eddoursul.json`) sets `expert_profile_save: /data/unsloth-ud-iq4_xs-learned.bin` with
`expert_profile_save_every: 1`, so every minute of prod traffic - which is my own agent inference between
gate sessions - rewrites the profile the gate loads, and the profile decides residency order. detA/detB
agreed because they ran back-to-back inside one GPU session with prod stopped; the next session loaded a
newer profile. Fix: the gate loads a frozen copy (`/work/.ph3-profile.bin` in the gate config; the golden
was re-recorded against it, since the 07:38 contents are gone).

## Goal

The verifier runs and commits windows whose rows are segments of different sequences, correctly before fast. Kernels may run per segment in a loop; the single-launch fusion is phase 4.

## Changes

- `include/strata/core/verify.hpp` + `src/core/verify.cpp`: add the batch API in Niko's shape (upstream names:
  `init_slots`, `run_slots`/`run_slot_rows`, `commit_slots`/`commit_slot_prefixes`), re-derived against our window
  code. A window is S row segments; **the batch graphs are captured per row layout** (Niko keys by `(rows, S, hbase)`
  with an LRU cap, baking each slot's session pointers in at capture), the segment split is staged data.
- Commit becomes per-segment with exact rollback to the previous window's state (our async commit beside the draft must order against per-segment keeps).
  **As built:** a batch window is one token per row, so like the one-token window it SELF-COMMITS (the window advances each slot's state, GDN
  keep=true, indexer append, PLE history) and there is nothing to roll back: `commit_slots` is host-side `ple_prev` bookkeeping only, no batch
  commit graph. Rollback-shaped keeps (`commit_slot_prefixes`, multi-row groups per slot) arrive with MTP-in-batch (phase 7). The draft/commit
  ordering worry was moot: batch windows carry no drafts.
- `include/strata/core/batch_rows.hpp` (orphaned, has an `init_slots` sketch) is either extended to match or deleted in favor of the real thing. Do not leave both.
  **As built:** deleted, with its unregistered test (`tests/core/batch_rows_test.cpp`), its unbuildable parity file (`src/kernels/batch_rows_parity.cpp`),
  and the two metal stub includes.
- Parity test: a two-sequence window equals two solo runs token-for-token, driven by a test harness, not the server.
  **As built:** `tools/batch_parity_test.py` drives one engine per leg: the solo legs through the ordinary serve `GEN` protocol (WITH MTP + lookup
  drafts - the strongest form of the claim), the batch leg through a `STRATA_BATCH_PARITY=<script>` mode in the engine's serve block (reuses the whole
  serve setup; allocates the registry slot sessions, feeds prompts and decodes greedily, one window per step, writes the streams, exits).

## Data structures

`SegRow { slot, row_in_slot }` staged table; per-segment commit records (positions, KV lengths, GDN state snapshots or redo info). Name the shape before coding: the window row buffer plus the segment table is the whole contract between scheduler and kernels.
**As built:** the contract is `run_slot_rows(rows, S, tokens, pos, ...)` - row t is slot `rows[t]`, token `tokens[t]`, position `pos[t]`; the layout
`rows` IS the graph key (`std::map<vector<int8_t>, cudaGraphExec_t>`, cap 64; a layout-churning scheduler hits the cap, phase 5 keeps layouts stable),
and capture-time `batch_rec_`/`brow_` steer `record_window`'s state-touching launches to each row's session. Positions/steps/tokens/PLE rows were
already per-row staging; nothing new needed there. No per-segment commit records (self-committing windows, see Changes).

## Verification

Static: build, ctest, the new two-sequence parity test, phase-1 digest gate unchanged for solo.
Runtime: a test program feeds two prompts' tokens as interleaved windows and diffs the token streams against two solo runs. Run it on both cards.

## Validation (2026-10-27, branch `batch-phase1-harness`, 4090 + 3090 rig)

| check | result |
| --- | --- |
| `verify.cpp` + `generate.cpp` syntax check, `-Wall -Wextra` | clean (only the pre-existing `raise_flag` unused-`f` warning) |
| source build, archs 80;86;89;120 | passed |
| solo digest phase-2 binary vs phase-3 binary, deterministic config + frozen profile, 6 requests x 2 fresh processes each | passed - `digests-cmpA.txt` and `digests-cmpB.txt` byte-identical, window shapes included; new golden `digests-golden.txt` = cmpB |
| `batch_parity_test.py --n 2 --max-new 48 --context 8192` (solo GEN with MTP+lookup drafts vs batch windows) | passed - both slots token-for-token equal |
| `batch_test` clean skip | passed |
| ctest | not run (this tree builds with `STRATA_BUILD_TESTS=OFF`; the deleted `batch_rows` test/parity files were never registered in CMake) |
| HIP / SYCL / Metal builds | not run (metal stubs: only the two dead `batch_rows.hpp` includes removed) |

## Findings

- **The `--expert-profile` file is a mutable gate input** (see the prerequisite section): prod rewrites it every minute of traffic, and the gate
  loads it. Freeze a copy for the gate; the golden is recorded against `/work/.ph3-profile.bin`.
- **Self-committing batch windows made the plan's rollback machinery unnecessary in phase 3**: one token per row means every row is kept, so the
  window commits itself exactly like the one-token window and `commit_slots` is host bookkeeping. Upstream agrees structurally (their
  `batch_self_commits` fast path skips the commit graph for one-row windows).
- **`--no-prefill-borrow` reserves prompt-path bytes that the phase-2 sizing numbers didn't include**: the parity batch leg was refused at
  `--context 16384` (rc=2) with the deterministic config; 8192 fits two slot sessions comfortably.
- **`groups_[T]` collision avoided**: solo split windows (`--spec-split`, G=2) and batch windows (always G=1) would fight over `groups_[rowcount]`
  at drive time; `run_slot_rows` refuses `split_` and `drive_window` takes a `batched` flag that pins G=1.
- Batch legs are fast: 2 slots x (prompt feed + 48 decode tokens) ran in ~20 s once the process was up (per-row launches are ~2x one-row launches
  per layer, the weights-read-once win arrives with phase 4's fused kernels).

## Risk note from phase 1 (measured 2026-10-26, resolved)

Greedy output on this branch's engine is not reproducible run to run with default flags: expert-cache adaptation
re-tiers experts mid-run and the tiers round differently (bisect: adapt default FAIL x2, `--adapt-every 0` PASS x2,
plus `STRATA_IQ_MT_MIN=1` throughout). Resolved for gating: every parity and digest gate in this plan runs under
`--adapt-every 0` and `STRATA_IQ_MT_MIN=1`, where the solo path is bitwise stable. The multi-sequence equality gate
in this phase is therefore meaningful as written, in deterministic mode.
