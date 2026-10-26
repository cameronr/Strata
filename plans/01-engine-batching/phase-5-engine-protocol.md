# Phase 5, the BGEN protocol and scheduler

Back-link: [overview.md](overview.md).

## Changes

- `src/program/generate.cpp`: accept `BGEN`/`BT`/`BDONE`/`BSTOP` (wire format per Niko's series; `include/strata/program/batch_read.hpp` is merged scaffolding to plug in). The decode loop becomes a slot scheduler: admit between windows, run one window over the admitted slots' rows, retire finished slots.
- Admission rules copy Niko's current shape: FCFS between windows, solo path per slot when only one is active (`STRATA_PARALLEL_SOLO` style fallback), `--parallel N` / `--slots N` alias, refuse above 8 (our verifier row cap).
- `docs/BATCHING.md` already describes the user-visible behavior; correct it where our branch differs instead of writing new docs.

## Data structures

The scheduler consumes phase-2's `SlotRegistry`; no new types beyond the protocol parser.

## Verification

Static: build, ctest, phase-1 digest gate (solo untouched when no BGEN was ever sent).
Runtime: `tools/batch_test.py` now actually runs and passes, GEN-vs-BGEN greedy equality at N=2 and N=4 on the dev box; two `curl` streams against the raw engine protocol interleave.
