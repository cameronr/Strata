# Phase 2, the slot state machine

Back-link: [overview.md](overview.md).

## Goal

Own per-conversation state explicitly so the scheduler and kernels can address several sequences. Types first, no behavior change; the solo path becomes slot 0.

## Changes

- New `include/strata/program/slots.hpp`: the slot registry. States free, admitted, running, finishing, freed. One full session per slot: GDN state, QSA K/V arena handle, PLE history, conversation snapshot handle (our `src/core/conversation_snapshot.cpp` exists and is unused by the loop, wire it to the type).
- `src/program/generate.cpp`: move the solo session's state behind slot 0 of the registry. Mechanical, no scheduling yet.
- Memory accounting: at startup with `--parallel N` (parsed, stored, still unused beyond sizing), print per-slot VRAM/RAM needs for GDN state, K/V at context size, draft state, and refuse to start when N slots cannot fit, in the style of our existing free-memory logs.

## Data structures

`Slot { id, state, SessionState, draft handles, kv arena, ple history }` and a `SlotRegistry` with the five-state transition table as the only mutator. The engine's single session becomes `registry.solo()`.

## Verification

Static: build, ctest, digest gate from phase 1 unchanged.
Runtime: solo run on the dev box, digests equal the phase-1 golden file; `--parallel 2` prints the sizing table and still serves one request at a time.

## Validation (2026-10-26, branch `batch-phase1-harness`, 4090 + 3090 rig)

| check | result |
| --- | --- |
| `slots_test` (registry states, table, solo, drafter), `-Wall -Wextra` | passed, 0 failures |
| source build, archs 80;86;89;120 | passed |
| `window_digest_test` on the final binary, 2 fresh processes | passed, cross-process stable |
| `batch_test` clean skip | passed |
| `--parallel 9` | refused at argument validation, rc=2 |
| `--parallel 2` at 32K (prod memory settings) | sizing table printed, clean fit refusal rc=2 (0.56 GiB/slot vs 0.72 GiB free; matches the 0.56 GiB docs/BATCHING.md quotes) |
| `--parallel 2` at 16K | sizing printed, served one request (24 tokens), clean rc=0 exit |
| `--parallel 8` | clean fit refusal rc=2 |
| ctest | not run (this tree registers no ctest; `slots_test` compiles and runs standalone, see its header comment) |
| HIP / SYCL / Metal builds | not run (the shim for the stdin read is `_WIN32`/POSIX only; no other new platform surface) |

## Findings

- **Cross-run golden gates need a fixed `--expert-cache`.** `auto` sizes from free VRAM at startup; two runs of the SAME build drift at requests 4-5 (measured: digests-phase2 vs phase2b, cache 5160 slots both times within runs, token streams differ between runs). The phase-1 golden comparison is therefore only meaningful with a pinned cache size; re-record the golden that way before phase 3 uses it.
- **The serve loop deadlocked on exit** (pre-existing, not phase 2): the detached stdin thread blocked in `getline(std::cin)` holds stdin's FILE lock, and `exit()`'s `_IO_flush_all` waits on it forever. QUIT never returned from exit; every tool hid it behind kill-timeouts, production only stopped cleanly because SIGTERM skips `exit`. Fixed: the thread reads fd 0 raw (`STRATA_FD_READ`, `_read` on Windows). Backtrace kept at `.exit-bt.txt`. Candidate upstream report.
- `--batch`/`--slots` are aliases of `--parallel` (the server's `parallel_args` sends those names); `batch_test`'s clean skip now comes after a full model load, since the engine no longer dies on the flag.
