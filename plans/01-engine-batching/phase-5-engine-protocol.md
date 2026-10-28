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

## As built (commits `c4a4228c1` `baf2b7349` + the exactness fixes below)

- Step 1 (`c4a4228c1`): `--parallel N` registers N dedicated batch sessions at `registry.at(1..N)`; degrade with a
  line instead of refusing when the slots do not fit.
- Steps 2+3+4 (`baf2b7349`): the BGEN protocol (`BGEN`/`BADM`/`BT`/`BDONE`/`BSTOP`, `ERR BGEN:` wording per the
  wire contract), the FCFS slot scheduler between windows, admission as `GEN 1` + `session_prefix_copy`, and the
  interleave harness against it.
- Step 5a: `BYIELD` is dropped silently (upstream drops it for a pipelined batch too); real give-way is 5b.

## Findings

- **Two window-shape-dependent programs survived into the batch path**, both in the same family as phase 4's
  nch finding, and both only bite where a shape-dependent choice changes a row's bits:
  - **The second GPU's share gate.** `expert_pool_dispatch_multi` let GPU2 take a layer's entries only when the
    window's total miss bytes crossed `--second-gpu-min-mb` (4 default). A batch window (S x 10 ids) crosses it
    where the same conversation's solo window does not, and with `--expert-math fast` the GPU's experts differ
    from the CPU's at ULP, so a token's tokens depended on which rows shared its window. The split is now a
    function of the entry alone (GPU2 takes what its tier holds); the flag and the gate are gone.
  - **Decode attention's split-K count.** `qsa_decode_attn_batch` derived `splits` from `n_q` (occupancy), so a
    drafted solo window (T >= 2 rows) reduced the same row's attention in a different order than the one-token
    windows the slot runs - visible exactly where selection fills every split, i.e. long contexts. It is pinned
    to the per-row call (`decode_splits(cap, 1)`), which `qsa_decode_attn_seg` already documented as
    load-bearing. This is the mechanism behind upstream's notes that a long read beside a decode "was not
    repeatable": both solo draft windows and batch windows were nondeterministically-shaped programs.
- **Debug route that found them**: token-level bisection (partner-dependence, prompt-length threshold), then a
  temporary state-hash dump (`session_state_dump`, since removed) that named the first diverging array - the
  first QSA layer's state, with everything before it bit-equal - after `session_prefix_copy` itself had been
  proven faithful by hashing every copied range on both sides.
- **The golden digests moved** with the gate fix: the old split made accepted tokens depend on co-scheduled
  draft ids, so removing it changed solo's window structure (and one request's text of six) at ULP branch
  points. Regenerated from the fixed binary, cross-process stable; the cost is documented here rather than
  hidden by keeping a window-dependent split.
- **Gate** (`.p5-gate.sh`, scratch): digest A/B, parity 48 + S=2 at 200, SHUFFLE S=2/S=8, batch_test 48/200,
  same-prompt S=2, admission-delay, long-prompt S=1/S=2 probes, and the full interleave (`--yield skip`)
  including the long reads C and D - all green on the 4090 rig with the deterministic config.
