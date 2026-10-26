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
