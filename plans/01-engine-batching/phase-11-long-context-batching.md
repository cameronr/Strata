# Phase 11, making batching safe and honest at long context

Back-link: [overview.md](overview.md).

## Why (measured, 2026-10-30/31)

The traffic-shape sweeps added with `--client-warm`/`--client-tokens` put the batch arm
against real agent traffic (llama-swap stats over 27 days: 89% of prompt tokens served
from cache, p50 resident context 82K, new tails p50 ~350 tokens, outputs p50 ~400,
44% of requests arriving while another runs):

| sweep (`bench-batch-ab.py --clients 4`) | solo | batch | delta |
| --- | ---: | ---: | ---: |
| 64K cold prompts, 128 out, total tok/s | 9.0 | 8.4 | -7% |
| 64K warm sessions, 1000 out, total tok/s | 39.8 | 33.6 | -16% |
| same, TTFT p50/p95 s | 26.5 / 52.3 | 27.8 / 58.5 | worse |
| short prompts, 128 out (the phase-10 shape) | 88 | 105 | +19% |

Batching wins where the window cost is weight-dominated (short context) and loses where
it is per-slot-state-dominated (long KV reads, four divergent contexts thrashing the
expert tier: expert-cache hit 85% in the warm sweep vs 96% solo). Upstream's own doc
reaches the same law ("the gain is in latency... on a card whose experts mostly run
from the CPU"), and Niko keeps CUDA at 8-row windows for the same reason we measured
(phase 10). That physics is accepted: this phase does not chase long-context
throughput. It removes the two things that make batching *unsafe* at long context and
the one thing that makes it *impolite*, and adds the sampling-path win:

1. **No admission control.** An over-full K/V pool exits the engine ("the K/V cannot
   grow"). A 4 x 100K experiment can kill the server; production must never be one
   long fourth request away from that.
2. **Expert residency collapses under concurrency** (85% vs 96% hit), which is most of
   the -16%.
3. **Draft depth ignores context size**: at 64K context every verified-rejected draft
   row costs a full 4-slot KV read; the trim caps rows, not their price.
4. **Sampling traffic gets no speculative win**: prod runs temperature 0.6, where
   exactness keeps acceptance at the model's own draw; Basalt's `--relaxed-drafting`
   (keep any draft the model gives probability >= P) is its own original, not upstream,
   and is exact for greedy/tools by construction.

Reference designs (do-not-rebase analysis, see phase 10's as-built note): Niko
`docs/BATCHING.md` (admission shape, slot cost rules), Basalt `docs/running.md`
(relaxed drafting, refuse-with-400 wording). This fork is +53.6% over Niko main
(`~/strata-results.md`, 2026-10-31); features are ported as designs onto the fast base,
never by rebase. Revisit if the nightly ratio crosses ~1.15 (it moved 1.99 -> 1.54 in
one week; the watch is the instrument).

## Scope

1. **Admission control** (`generate.cpp` kvg path + `serve/server.py`).
   - Engine: a request whose prompt + growth margin cannot be admitted even after all
     give-backable cells (idle slots, parkable conversations) are returned answers the
     protocol with a refusal line (`REFUSED <needed_cells>`); it never reaches
     `kvg_ensure` failure. The existing `kvg_trim` gives back slots first; refusal is
     only when that cannot reach the need.
   - Server: `REFUSED` becomes a 400 naming the limit (Basalt's wording: a length the
     slot cannot hold), and the queue keeps FIFO order. `/metrics` gains a refusal
     count.
   - Startup: per-slot fit warnings in the engine's INFO block (Niko's shape: name the
     slots that do not fit, start with the ones that do; fewer than two fitting turns
     batching off with a log line).
   - The batch arm config raises `--cache-ram` 8192 -> 16384 (the warm sweep showed the
     8 GiB stash cannot hold 4 x 64K sessions; the container has room: 55.4 GiB experts
     against `--memory=88g`).
2. **Expert residency under concurrency** (audit first).
   - Measure per-window VRAM-tier hit with 1 vs 4 active slots at equal total context
     (the phase-9 telemetry gives routed-entry counts; add a per-slot breakdown).
   - If the rhythmic round is the loss (one session's heat evicting another's every
     round): try decay tuning (`--adapt-decay`) and a min-gain floor
     (`--adapt-min-gain`, Niko default 1.5) before any new mechanism; only then consider
     per-slot routing heat feeding the round's ranking.
   - Deliverable includes the negative-result path: if residency cannot be saved at 4
     long sessions, the honest lever is item 3 + a slot-count recommendation in the
     batch config's comment.
3. **Context-priced drafts** (`batch_step` trim).
   - Price a draft row as roughly the window's mean slot length (KV read grows with it):
     extend the trim's least-likely-last drop with a context term, so at 64K contexts a
     slot keeps ~1 draft even when row count would allow more, and the 8-row default
     cap stays for short contexts. Env knob mirroring `STRATA_BATCH_WINDOW`
     (`STRATA_BATCH_DRAFT_PRICE`, default on; 0 = today's behavior) for the A/B.
4. **Relaxed drafting** (`src/core/verify.cpp` accept path, `serve/server.py` request
   parsing).
   - Engine: accept a drafted token whose model probability >= P (request
     `relaxed_drafting`, else config `--relaxed-drafting [P]`, default off; `true` =
     0.5) instead of only the model's argmax draw. Greedy (temperature 0), logit_bias,
     grammar/tool requests, and control tokens stay exact, as Basalt's.
   - The digest gates run greedy, so they must be bit-identical with the flag off - and
     they also prove the exactness carve-outs.
   - Per-request opt-out stays (a client can demand exactness).
5. **Bench: the warm sweep becomes a standing arm.**
   - `bench-batch-ab.py` gains a third scenario row set: `--client-warm 65536
     --client-tokens 1000` (the measured shape above) beside the existing default and
     2x/4x rows, so every future phase is judged on the traffic shape, not only the
     short-prompt one. Documented in `testing.md`.

## Data structures

No new engine types. `BRequest`/`BSlot` gain a refusal path (no slot state changes);
the trim gains one pricing term; the verifier accept path gains one comparison.

## Verification

- Static: build, ctest, solo digest gate (item 4 off must be bit-identical; items 1-3
  must not move greedy solo tokens).
- Item 1: a scripted over-subscription (4 requests of 120K into the 4-slot arm) ends in
  four 400s and a live server, `/metrics` showing refusals; the same load today exits
  the engine. Add to the gate runner as G7.
- Item 2: per-slot hit telemetry committed with the sweep logs; the round's swaps and
  the warm-sweep expert hit before/after.
- Item 3: warm sweep (the new standing arm) before/after; the short-prompt +19% row must
  not regress (pricing only bites above the context threshold).
- Item 4: greedy digests unchanged with the flag off; with the flag on, sampled streams
  keep tool-call and control-token text exact (extend the interleave harness with a
  sampled tool-call case); record tokens/window and acceptance as Basalt's done lines
  do (`acc`, `dl`).

## Non-goals

- Long-context aggregate-throughput wins from batching (accepted physics; latency and
  refusal are the goals).
- Concurrent prefill threads (phase 8's as-built stands: single-thread invariants;
  chunk interleave + BYIELD already landed there).
- Per-slot KV streaming (`--kv-resident` shape): bigger surgery, separate phase if the
  warm sweep still demands it after items 1-3.
- Rebase onto Niko main (see the phase-10 as-built note; ratio-watch rule).
