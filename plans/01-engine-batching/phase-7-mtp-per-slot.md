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

## As built

Three commits on `batch-phase1-harness`: `5fd5afb0e` (per-slot draft K/V stores, `--batch-mtp`, the store copies
that follow a conversation in and out of a slot), `c2f044954` (the drafted window: assembly, acceptance, commit),
and the harness/docs commit with these numbers.

Shape as planned (Basalt's: one drafter, per-slot stores), with three deviations:

- **MTP-only drafts.** The batch windows do not feed the prompt n-gram lookup drafter (the per-slot lookup rows
  were not built); a slot's window carries its MTP proposals, up to `--spec-1`, gated by `--spec-min-p` like the
  solo loop's. Token identity holds anyway: verification output is draft-shape-independent (`STRATA_IQ_MT_MIN=1`),
  which is what every digest gate here rides.
- **The store contract is the phase-4 seg table.** Draft graphs bake only the table address; `draft()` stamps the
  mapped row -> store index per round (store 0 is the solo store), so one graph serves every slot. No per-slot
  drafter objects, no graph captures per slot.
- **Window assembly lives in the driver** (`batch_step` in `generate.cpp`), mirroring the solo loop: per-slot row
  counts from the draft odds with the eos-draft, length and context clamps, then trim-to-8 dropping the
  least-likely last draft (`dprob`) first. The commit (`Verifier::commit_slots(keep)`) snapshots each slot's GDN
  state and indexer tails before the window and replays each group's accepted prefix through the solo commit
  kernels as plain launches; the PLE history rolls back to the per-row snapshots the window already writes. With
  slot drafts on, the fused graphs advance the state-touching kernels one row at a time (a group's rows chain one
  slot's state; grouped parallel launches would race it), KV append and attention stay grouped.

Three bugs the gates caught, worth remembering:

- The drafter zero-fills `drafts`/`probs` to `max_window()-1`, not to `--spec-1`: per-slot arrays sized `spec-1`
  overflowed and zeroed the slot's `produced` counter every window (the length cap never fired). Size them
  `kVerifyMaxT`.
- `keep[b]` must be capped at the slot's end (the eos pick, a stop, the length and context caps) **before** the
  commit: committing rows past the last emitted token leaves the slot's state and ids past the conversation and
  breaks the give-back's prefix match.
- `commit_slots` ends with a `cs_` sync. Its replay kernels run after the window (phase 5's state kernels lived
  inside the graph and settled before the picks), and a give-back's `session_prefix_copy` runs on the main stream,
  which has no ordering with `cs_`: the PLE history restore raced the copy and the continuation decoded from a
  history three tokens past the frontier. Found by checksumming each state part at give-back against a
  drafts-off run of the same conversation - the parts that matched pointed at the one that didn't.

Verification (deterministic config: frozen profile, `--adapt-every 0`, `STRATA_IQ_MT_MIN=1`, `--pcie-frac 0`):

- Digests: flag-off == golden; `--batch-mtp` on == plain solo. ctest green (`draft_proxy_parity` among them; the
  fixture-dependent `iq_parity`/`expert_parity`/`pool_test` skip on this box for missing data, as before).
- `batch_test --mtp --batch-mtp`: N=2 @96 and N=1..4 @128 IDENTICAL per slot; a 2000-token prompt read beside two
  drafting slots IDENTICAL. Interleave/give-back: 10/10 with `--batch-mtp`, including the solo continuation after a
  mid-stream `BSTOP` and give-backs after eos, length and cancel endings.
- Acceptance and rate (greedy @128, aggregate tok/s incl. admissions; the baseline is the same config with the
  flag off):

| slots | drafts off | drafts on | gain | batch acceptance (solo same run) |
| --- | --- | --- | --- | --- |
| 1 | 44.9 | 49.5 | +10.2% | 0.565 (0.569) |
| 2 | 47.2 | 50.7 | +7.4% | 0.676 (0.679) |
| 4 | 52.8 | 53.7 | +1.7% | 0.747 (0.629) |

- Batch acceptance sits at the solo rate; at 4 slots the 8-row window is full at two rows per slot, so drafts are
  trimmed to the most likely pair and the gain shrinks (trim-to-fit costs the drafter its long tail - the
  acceptance rate rises while the per-round gain falls). The deterministic config (MT_MIN=1, no prefetch, frozen
  profile) is the slow one; these are the comparison numbers, not the production rate.
