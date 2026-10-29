# Phase 9, peer-tier windows and per-slot sizing (stretch)

Back-link: [overview.md](overview.md).

## Goal

Make batched windows good on this rig specifically: the 5070 Ti peer tier serving multi-sequence windows, and memory that actually fits a display-attached 16 GB card.

## Changes

- Peer-tier window integration: Basalt's peer GDN key-head split, peer-first expert refill and device-planned MoE tail run under batch windows on their rig and exist nowhere in Niko for our topology. Port the designs that measure out on the 3090 + 5070 Ti pair; skip what only helps NVLink-fast peers. Start-time cost pricing of mechanisms (their `multi_gpu_plan.cpp` idea) decides per-PC defaults here instead of hand-tuned `--prompt-shift` style knobs.
- Per-slot memory: our `--kv-grow` elastic K/V already exists; make it per-slot and account GDN state, draft stores and PLE history per slot so `--parallel N` sizing from phase 2 is exact on the 16 GB peer card. Basalt's partial-slot start (start with the slots that fit, grow later) is the UX to copy.
- Re-measure the RTX 5070 12 GB table from upstream's BATCHING.md on our pair and publish the numbers in our docs/BATCHING.md. Use the real workload shapes, not synthetic identical prompts: two same-domain coding sessions, one coding plus one Chinese chat (less expert overlap, the pessimistic pair), and a 4-5 conversation subagent swarm with shared prefixes (prefill-heavy, KV-hungry). Expert overlap across the window's rows is what sets the spill penalty, so the table must name the domain mix per row.

## Data structures

The cost model is a pure function over card properties plus the model's byte profile; keep it out of the engine loop.

## Verification

Static: build, ctest, unit tests for the pricing function, phase-1 digest gate.
Runtime: tok/s and TTFT at 1/2/4 slots with the peer tier on and off; the published table lands in docs/BATCHING.md with our hardware named.

## As built (phase 9, measured-gaps scope)

The scope was re-cut against what the code maps showed (user-approved): the peer serving path was already
identical under batch windows (`expert_pool_dispatch_multi`, bit-identical peer rows), and Basalt's three named
mechanisms exist nowhere as code with a reference on this box. Shipped instead:

- **Adaptation rhythm inside batch windows** (`generate.cpp` batch_step): batch windows advance the shared
  `rounds` counter, set `adapt_next`/`flush_next` on the same modulo, and run `drive_window(S)` (admission of
  landed moves + shape-aware staging) before `run_slot_rows`; the ranking thread joins before every window and
  before every request starts (closing an in-flight-round race the frozen shape never had). A `tiers_still`
  flag keeps the tiers frozen through phase-8 read-boundary windows (the read may loan cache slots).
  `STRATA_BATCH_ADAPT=0` restores the pre-phase-9 shape. Measured: +25% aggregate tok/s at 2 slots on the
  production config (64.5 against 51.1-52.0); with the rhythm, batching wins on this rig, frozen it does not.
- **Exact per-slot accounting + partial-slot start**: `slots_that_fit()` (`include/strata/program/slot_fit.hpp`,
  pure, `slot_fit_test`) carves the slots that fit instead of zero; the fit counts `--batch-mtp`'s draft store
  and commit snapshots (0.67 against 0.52 GiB per slot at 30K here); INFO/`bs`/BGEN/parity-leg all speak the
  carved `n_slots`. Verified live: `--parallel 8 --max-context 30000` -> "starting with 4 slots", slot 3 BADM,
  slot 4 ERR.
- **Measurement tables** (docs/BATCHING.md): domain-mix shapes (coding pair / coding + Chinese / four-task swarm
  on a shared 1.2K prefix) at 1/2/4 slots, peer tier on and off; and the rhythm A/B. Peer tier is worth 2.4-3.7x
  batched decode on this pair.

Skipped, with reasons: peer GDN key-head split (distributes GDN compute = layer-split territory, out of scope);
device-planned MoE tail (deep graph surgery, no reference, no bottleneck evidence); replacing hand-tuned knobs
with a full `multi_gpu_plan` (the pricing seed exists as `slots_that_fit`; replacing prod defaults without a
reference rig to validate against is guessing); runtime slot growth ("grow later" of Basalt's partial start -
registry and captured graphs are fixed at startup; a phase of its own).

Gates (all green): G1/G2 digests byte-identical to the phase-7 streams (the rhythm code is inert under
`--adapt-every 0`), G3 batch parity N=2@96 IDENTICAL, G4a interleave 10/10, G5 startup scenarios, G6 rhythm
smoke both switch positions, ctest 47/47 (three fixture-missing tests of this box excluded).
