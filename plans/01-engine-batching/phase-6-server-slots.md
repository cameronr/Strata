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
