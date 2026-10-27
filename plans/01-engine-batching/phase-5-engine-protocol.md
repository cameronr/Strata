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

## Scheduler obligations the phase-4 audit moved onto this phase

- **Register once, at the maximum.** `init_slots` fixes the slot count and the table sizes for the Verifier's life
  (the device tables live at baked addresses; a count change fails stop). The scheduler registers the full
  `--parallel` set at startup and keeps every session allocated; the memory floor is `N x (KV + indexer + GDN)`
  at full context, and GDN state (~0.11 GiB/slot) does not shrink with context.
- **Homogeneous slots.** Every slot must equal the verifier's own session's `max_cells` (init_slots enforces it):
  admitting a short prompt pays the largest configured context's state. Sizing talk must quote both numbers.
- **Eviction is same-count replacement.** Re-`init_slots` with the same count re-uploads the tables (serialized:
  init_slots syncs `cs_` itself); there is no shrink path.
- **Slot 0 aliases `ss_` in today's harness.** Until solo and slot 0 stop sharing a session, solo GEN and batch
  rows must not run on one Verifier.
- **Gate list grows one item:** the seg kernels compile for sm_120 but the rig has no card, so the SASS-identity
  and parity suites gate sm_86 + sm_89 only; add an sm_120 compile check (and, when hardware allows, a suite run)
  to this phase's gates.
