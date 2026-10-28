# Phase 6, server slot routing

Back-link: [overview.md](overview.md).

## Goal

The OpenAI-compatible server runs concurrent requests through engine slots instead of the FIFO lock.

## Changes

- `serve/server.py`: the `batch_slots` handshake (line ~1083) and `generate_batched` (~1735) already exist for the Qwen3.6 engine. Make the main engine satisfy them. When `batch_slots > 0`, requests route to free slots and the FIFO lock stops guarding the inference path; when 0, today's serialization is untouched.
- Queue and admission at the server: FCFS with bounded overtaking on prefix-cache hits is Basalt's improvement (`serve/engine/slots.py`), worth copying now that the protocol exists. This one earns its keep for parallel-subagent traffic, where several requests share one long prefix.
- `/slots` and `/metrics` exposure of live slots and queue depth, matching upstream's contract.
- Cancellation and disconnect free the slot; a slot keeps its conversation cache between turns (slot-as-conversation-cache routing, upstream).

## Data structures

Server-side slot table keyed by conversation; no engine changes.

## Verification

Static: server test suite, phase-1 digest gate.
Runtime: two concurrent streaming `curl` calls finish in overlapping time with correct per-request tokens (compare against solo replays); `"parallel": 2` vs FIFO recorded tok/s and TTFT on the dev box, appended to the phase-1 baseline table.

## As built (validation 2026-10-28, branch `batch-phase1-harness`, 4090 + 3090 rig)

The server side arrived almost entirely with the merged Qwen3.6 scaffolding, and phase 5's engine satisfies it:
`"parallel": N` -> `--batch N` (`parallel_args`), the `INFO batch_slots=N` handshake, `generate_batched`'s slot
table (admission under `ctl`, `BT`/`BDONE` routing in `_pump`, `BSTOP` cancellation, refusal retry #1994,
conversation affinity in `pick_slot`, orphan reaper), `/slots` + `/metrics` slot views, and the FIFO lock already
skipped when slots are live.  What the plan listed as to-build was checked as existing, not rebuilt:

- **Bounded overtaking** exists as `STRATA_ADMIT_SHORT_FIRST` + `STRATA_ADMIT_MAX_WAIT_S` (`_take_control`'s
  `new` ordering + `_admit_next`): short reads go before long ones, a long one is never held back past the max
  wait.  `serve/engine/slots.py` as a separate module was not created: the logic lives with the locks it guards.

Static: `serve/test_parallel.py` + `test_slot_alloc` + `test_admit_order` + `test_slots` + `test_prometheus` +
`test_queue_keepalive` - 92 tests / 4011 subtests, all pass (host python, no GPU).  Engine: no code changed in
this phase, so the phase-1 digest gate stands from phase 5's final run.

Runtime (`.p6-bench.py`, scratch; deterministic config + `STRATA_IQ_MT_MIN=1`, real server + real engine, two
distinct ~400-token greedy thinking prompts streamed concurrently; reference texts from fresh processes, one per
prompt - the settled gate rule):

| arm | pair wall | TTFT req-b | per-stream tok/s | aggregate tok/s | texts |
| --- | --- | --- | --- | --- | --- |
| `parallel: 2` | 16.0 s | **1.4 s** | 116 + 116 | 232 | = reference |
| `parallel: 1` (FIFO) | 15.8 s | **9.1 s** | 117 then 219 | 233 | = reference |

- **Correctness:** both concurrent streams' texts equal their fresh-process solo replays, on both arms.
- **The gain is latency, not throughput:** the second request's TTFT drops 9.1 s -> 1.4 s (admitted beside the
  first instead of queued behind it); decode splits the same total (this MoE's decode is PCIe/expert-bound, so
  two rows cost about the same as one - aggregate holds at ~232 tok/s).  Wall time of the pair is unchanged.
- **Gate-method note:** in-process replays of the same prompts differ from the concurrent texts even on the FIFO
  arm - the prompt-lookup drafter's cross-request carry, exactly what phase 1 excluded from determinism gates.
  A phase-6 gate that compares texts must use fresh processes, or it measures the wrong thing.
