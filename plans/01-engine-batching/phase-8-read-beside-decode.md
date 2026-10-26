# Phase 8, prompt reads beside running windows

Back-link: [overview.md](overview.md).

## Goal

A new prompt is read on its own thread while running windows keep decoding, with cooperative preemption. This is the phase users feel: Basalt holds streams at 85-92 tok/s during a 64K read where they used to fall to 2-3.

## Changes

- `src/program/generate.cpp`: admission starts the prompt read on a dedicated thread; decode kernels launch at high stream priority, the reader at low (Basalt `set_main_priority`, Niko's chunked hand-off pieces with event gates as references).
- `BYIELD` cooperative preemption: the reader stops at a chunk boundary when a window needs the cards, resumes after; the admit-short-first switch (`STRATA_ADMIT_SHORT_FIRST` shape) keeps long prompts from starving the queue.
- Expert-cache and PLE access from the reader thread must stay race-free against window kernels; start reads only for layers the running windows have passed, which is what upstream's chunk gating does.

## Data structures

No new engine types; the reader thread owns a phase-2 `Slot` in admitted state until its read completes.

## Verification

Static: build, ctest, phase-1 digest gate (window tokens must be unaffected by a concurrent read).
Runtime on the dev box: two streams running, start a 64K prompt, record stream tok/s during the read and the reader's TTFT; compare against the FIFO baseline where streams stall for the whole read.
