# Batch slots for the main engine

Back-link: this file. Phases are linked at the bottom.

## Context

The server serializes requests: `serve/server.py` holds a FIFO lock (`self.fifo`, line 3422) across a whole request, and the main engine (`src/program/generate.cpp`) accepts only `GEN`/`GENI`, one session per pipe. Two chat users means one waits. The root project (Niko1221/Strata) shipped exactly this feature (batch slots, PR #559 + the #2364 multi-user series, docs/BATCHING.md is already in our tree), and Basalt runs a rebased copy of it. Our branch merged those histories twice and resolved the engine as "stays this branch's", so the engine-side code never entered our tree while the docs, the server handshake (`batch_slots`, `generate_batched`), the harness (`tools/batch_test.py`), and scaffolding headers (`batch_read.hpp`, `batch_rows.hpp`, `conversation_snapshot.cpp`) did.

Worth having on this rig, with honest expectations. Niko's closest measured card (RTX 5070 12 GB, experts spill): slots buy waiting time, aggregate throughput drops 11-24% at 2+ slots but TTFT goes 11.2 s to 1.8 s. On a busy server that is the right trade. The read-beside-decode half is a separate win: Basalt holds running streams at 85-92 tok/s during a 64K prompt read where they used to fall to 2-3.

## Scope

In: multi-sequence verify windows, per-slot session state and drafts, the `BGEN` protocol and between-window admission, server slot routing, prompt reads beside decode, memory sizing for the 3090 + 5070 Ti peer-tier config, up to 8 slots.

Out: layer-split pipelining and `--batch-groups` (this branch tiers experts, it does not split layers), more than 8 slots, per-slot grammar (no grammar decoder here), relaxed drafting in windows, Windows-specific work, the Qwen3.6 path (already batched).

## Constraints

- Every phase must keep solo runs token-identical. The digest harness (phase 1) is the gate for every later phase.
- Our engine files are heavily rewritten versus both sources: `generate.cpp` differs from the batching patch base by ~7,400 lines and has 202 of our commits since the 2026-10-19 merge-base; `verify.cpp` has 71. Batch behavior must be re-derived against our verify window, draft_proxy, PLE, and peer-tier code, not pasted.
- sm_86 and sm_120 both in scope; no TMA, clusters, or PDL assumptions in new kernels.
- Kernels stay bitwise-parity tested: single-launch multi-sequence must equal per-sequence runs.

## Alternatives considered

1. Cherry-pick Niko's series. Rejected: the first commit alone conflicts in 13 hunks of our `verify.cpp`, and the auto-merged hunks across a 7,380-line base gap are Frankenstein merges, not signal.
2. Cherry-pick Basalt's series. Strictly worse: their first batching commit is a byte-identical replay of Niko's, every follow-up pays a `basalt`/`strata` path-rename tax, and their series churns `verify.cpp`/`mtp.cpp`/`ngram.cpp` (170/59/19 commits), the exact files our 407 engine commits rewrote.
3. **Chosen: re-implement from the design, Niko's series as the reference implementation.** The API (`init_slots`/`run_slot_segs`/`commit_slots`), the protocol (`BGEN`/`BT`/`BDONE`/`BSTOP`), the admission rules, and the harness all exist upstream to copy designs and test expectations from. Basalt-only extensions (per-slot draft stores, GDN commit pipelining, peer-tier window integration, admission overtaking on prefix hits) port on top afterward, where they beat Niko's version.

## Applicable skills

Implementers must load per phase: **how** (Explain mode) over any subsystem before changing it, especially the verify window and the generate loop. **interrogate** before shipping phases 3 and 4 (the contested designs). **typescript-best-practices** is not relevant; C++ conventions follow the nearest file. **unslop** over every diff and every prose surface. **babysit** after opening each PR.

## Phases

1. [phase-1-exactness-harness.md](phase-1-exactness-harness.md), wake the token-exactness harness and lay digest gates.
2. [phase-2-slot-sessions.md](phase-2-slot-sessions.md), the slot state machine and per-slot session ownership.
3. [phase-3-multiseq-verifier.md](phase-3-multiseq-verifier.md), multi-sequence windows, naive per-segment kernel runs first.
4. [phase-4-segment-kernels.md](phase-4-segment-kernels.md), segment-table kernels, one launch over rows.
5. [phase-5-engine-protocol.md](phase-5-engine-protocol.md), BGEN protocol, scheduler, admission.
6. [phase-6-server-slots.md](phase-6-server-slots.md), server slot routing over the existing handshake.
7. [phase-7-mtp-per-slot.md](phase-7-mtp-per-slot.md), draft state per slot.
8. [phase-8-read-beside-decode.md](phase-8-read-beside-decode.md), prompt reads beside running windows.
9. [phase-9-peer-tier-sizing.md](phase-9-peer-tier-sizing.md), stretch: peer-tier windows and per-slot memory sizing for this rig.

Order is deliberate: the safety net first, then the shared types, then the verifier, then kernels, then the loop, then the server. Each phase ships alone and behind flags; absent flags change nothing, mirroring upstream's "opt-in and changes nothing when the options are absent".

## Verification (project level)

Static: full build, `ctest` (including the new parity tests), solo digest gate.
Runtime on the 3090 + 5070 Ti box: `bin/strata` with a real pack; `tools/batch_test.py` greedy GEN-vs-BGEN equality; two concurrent streaming `curl` calls against the server with `"parallel": 2`; tok/s and TTFT recorded before/after each user-visible phase. No phase is done on unit tests alone.

## Implementation guidance

- Build the harness (phase 1) before touching any engine behavior. Every later phase ends in a digest or parity gate it must pass.
- Re-derive, don't paste: when Niko's code and our code disagree, the design goes in Niko's shape but our file's local rules win.
- Keep solo-path code paths free of batch branches at the kernel level; batch behavior arrives through the segment table, so solo is the one-segment case.
- Land each phase as its own PR through `wt switch --create`, squash on merge.
