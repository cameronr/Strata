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
