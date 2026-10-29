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

## As built

The reader thread and stream priorities were not built: the engine has one Verifier, one ExpertPool, one PLE table
and one residency table, all single-thread by design, and a second thread issuing through them is a rewrite of the
phase-2..6 machinery, not of this phase. What was built instead is upstream's own one-GPU shape, which this branch
had lost: **chunk-granularity cooperative interleave**. The read stays on the request thread; the slots' windows run
between its chunks, and `BYIELD` preempts at a chunk boundary. This delivers the phase's user-visible property
(streams do not stall behind a read) without touching any of the single-thread invariants.

- `generate.cpp` (`read_part` + `read_boundary` in the serve request loop): with slots on, a segment is read one
  chunk at a time (`batch_read_piece`, header wired in for real now, `batch_read_test` registered in ctest); after
  each piece the boundary consumes `BSTOP`/`BYIELD`, parks the part read on a taken `BYIELD` (`session_prefix_copy`
  + draft `copy_store` + `YIELDED <slot> <tokens>` + `DONE cancel`; the resend goes on through the existing
  slot-gives-back path), and runs windows for `STRATA_BATCH_DECODE_SHARE` x the piece's time (default 0.5, 0 = the
  old FIFO behavior).
- Windows run only where the sampling matches them: `Verifier::windows_allowed()` (greedy, no penalty history).
- `--prompt-split` reads go whole (the stages pipeline over one run's chunks; a piece would drain it) but get the
  windows at their segment boundaries; their `BYIELD` is dropped (the second stage may still be writing the session
  a park would copy), as upstream drops the BYIELD of a pipelined read. Picture reads go whole.
- The harness's long prompts (C, D) are now soloed after their slot reads: a solo reference sent first leaves the
  prompt checkpointed, and the slot phase silently restored it instead of reading - the interleave this test is
  named for had never actually run beside a window on any earlier run of it (upstream papered over this with a
  stash small enough to evict; this box's stash never does).

Verification (deterministic config; digests compared against the phase-7 final streams, `.p7-dig-off2.txt` /
`.p7-dig-g2.txt` - the older golden files predate phase 4's window-shape change):

- Solo digests unchanged, flag-off and `--batch-mtp`; `batch_test` N=2 @96 IDENTICAL.
- `batch_interleave_test.py` with the pieces live: 10/10 IDENTICAL, `YIELDED 2 1024`, the resume reusing exactly
  `1024 cached`, 143 slot tokens arriving during the admissions' reads, "the prompt was read in" x 2. With
  `--prompt-split` (production shape): 10/10 IDENTICAL, no pieces, `--yield skip`.
- ctest 50/51: `expert_parity`/`pool_test` (missing `pack/full/experts.bin`) and `ple_parity` (missing the Q2_0
  RCO GGUF) fail on fixtures this box does not have, as in phase 7; the hard error on a missing fixture is that
  test's own documented design.
- Bench (two slots streaming while a 6.4K prompt reads): FIFO 0.26 tok/s vs interleave 23.98 tok/s during the
  read; the read pays 10.7 -> 15.5 s, about the decode share x its time.

Finding (pre-existing, not this phase): a read cancelled mid-way leaves its completed pieces as live prompt-cache
tokens, and the next same-prompt request reuses them without a checkpoint (`PromptCache::reusable`, live full
match). That reuse changes the later read's chunk shapes, and a greedy resend can drift from a fresh-solo run
(measured: first difference at token 22 of 40; with `--prompt-cache 0` the resend equals solo, and a cancel after
the read completes never drifts). The old code's cancel path set the same live tokens, so this is upstream's
semantics, not a regression; the digest gates (fresh processes) are unaffected.
