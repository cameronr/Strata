# Phase 1, the exactness harness

Back-link: [overview.md](overview.md).

## Goal

A rerunnable gate that proves future phases keep solo runs token-identical, and a GEN-vs-BGEN equality test that will start passing in phase 5. Nothing in the engine changes behavior this phase.

## Changes

- `tools/batch_test.py`: our merged copy already does GEN-vs-BGEN greedy equality over up to 32 prompts, but it targets an engine that speaks BGEN. Make it skip cleanly with a clear "engine has no batch slots" result today, and pin its protocol expectations to Niko's (reference: `9461f5e3a` and `tools/batch_test.py` on niko/main).
- `src/program/generate.cpp`: add a debug-only digest line per decode window (window rows, accepted count, sampled output-token hash chain), behind an env switch in the style of Niko's `--exp concur5_digest`. This is the solo baseline recorder.
- `tests/`: a golden-file test that runs the tiny test pack solo and records the digest stream; reruns must match.

## Data structures

One digest record per window: sequence id, window index, row count, token-hash chain value. Text output, no new engine types.

## Verification

Static: build, ctest.
Runtime: run the golden test on the dev box twice, digests equal; run `batch_test.py` and see the clean skip. Record the solo tok/s/TTFT baseline table for the phases ahead.

## Findings (measured 2026-10-25, 4090 + 3090 rig, IQ4_XS pack)

- In-process repeats are not a determinism test: the prompt-lookup drafter carries text across requests, and greedy tokens can move with the resulting window shapes (upstream #152: expert rows round differently alone than in a group unless `STRATA_IQ_MT_MIN=1`).
- Cross-process repeats with default flags also diverged: fresh processes chose different draft counts by request 1 window 21, then token streams moved. Suspect sub-threshold nondeterminism (GPU accumulation order) flipping near-threshold drafts, amplified by the default MT_MIN=2 rounding.
- Gate definition under test now: fresh process per repeat, `STRATA_IQ_MT_MIN=1` (drafting-independent rounding, same as batch_test). If this still diverges, the solo path has genuine run-to-run nondeterminism and it gets its own investigation before any phase gates on token equality.
- **Root cause found (2026-10-26, bisect arms, 2 rounds each):** expert-cache adaptation. With defaults (adapt every 4 rounds) fresh processes diverge at the same window with row counts flipped in opposite directions, a scheduling coin flip; with `--adapt-every 0` two fresh processes produce identical 171-window digest streams and token streams, twice. Mechanism: adaptation re-tiers the most-routed experts mid-run, and CPU / local-GPU / peer-GPU expert kernels round differently, so logits depend on swap timing.
- Gate definition, settled: fresh process per repeat, `STRATA_IQ_MT_MIN=1`, `--adapt-every 0`.
- Production consequence: the daily server runs `--adapt-every 1`, so its greedy output is not reproducible run to run. A candidate upstream report (the #152 docs cover CPU-internal rounding; the cross-tier variant via adaptation appears undocumented).

## Validation (2026-10-26, branch `batch-phase1-harness`)

| check | result |
| --- | --- |
| `py_compile` both tools files | passed |
| source build (`docker build`, archs 80;86;89;120) | passed |
| `window_digest_test` e2e, 2 fresh processes x 6 prompts, deterministic mode | passed, 170 windows stable |
| `batch_test` clean skip against the main engine | passed |
| `ctest` | not run (no test touches the digest path) |
| HIP / SYCL / Metal builds of generate.cpp | not run (change is an env-gated stderr line) |
| q36 batch path against a batching engine | not run (logic preserved; INFO parsing is additive) |

Development is on fork branches; upstream's PR conventions apply only when the batching work is eventually submitted upstream.
