# Phase 7, MTP drafts per slot

Back-link: [overview.md](overview.md).

## Goal

Speculative decoding inside batch windows. Today `--batch-mtp` is documented for one slot's MTP proposals; this makes every slot draft.

## Changes

Two designs exist. Niko runs per-slot drafters with their own K/V rings, opt-in via `--batch-mtp`, with layer-split placement rules we do not need. Basalt keeps one drafter with per-slot K/V stores (`MtpDrafter::set_stores`) and drafts in every window. Basalt's shape fits our single drafter (`src/core/mtp.cpp` is host-driven with one shared store) and is the smaller change; take it, keep Niko's opt-in flag name for compatibility.

- `src/core/mtp.cpp` / `include/strata/core/mtp.hpp`: per-slot draft K/V stores, swapped into the working state by device copies between draft and verify (Basalt `ca533a53` as reference design; our unbounded `draft_proxy` plugs in unchanged since it already rescoring any row count).
- Window budget: drafts from several slots share the 8-row window; trim the least-likely-accepted drafts when the total exceeds it (Basalt's trim-to-fit rule).
- Per-slot n-gram lookup rows (Niko's `ngram_rows` per segment, ours is `src/kernels/ngram.cpp`).

## Data structures

`DraftStore { kv arena, lookup rows }` per slot, owned by the phase-2 `Slot`.

## Verification

Static: build, ctest, existing `draft_proxy` parity tests, phase-1 digest gate (solo MTP untouched).
Runtime: batch_test.py with `--mtp --batch-mtp` at N=2, token streams equal solo-with-MTP replays; acceptance rate and tok/s reported for 1/2/4 slots against the baseline table.
