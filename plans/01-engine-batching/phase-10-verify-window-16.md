# Phase 10, the 16-row verify window

Back-link: [overview.md](overview.md).

## Why (measured, 2026-10-29)

The first real A/B (`~/strata-data/bench/batch-ab/2026-10-29`, 2 rounds, 262K config,
4 slots + `--batch-mtp`, tiers rhythmic):

| | solo | batch | gain |
| --- | --- | --- | --- |
| chat decode tok/s | 126.3 | 122.3 | -3% |
| 2x total tok/s | 87.1 | 91.4 | +5% |
| 4x total tok/s | 90.5 | 105.1 | +16% |
| 4x TTFT p50/p95 s | 2.4 / 4.7 | 1.0 / 1.8 | -60% |

4 streams produce less than one solo stream. The binding constraint is the window:
`kVerifyMaxT = 8` rows, while each drafting slot wants `1 + spec = 5`. `batch_step`
trims drafts greedily to fit (`generate.cpp`, "trimmed to the 8-row window by dropping
the least likely last draft"), so at 4 slots every slot verifies exactly **1 draft
against a spec depth of 4**, and at 2 slots 3 of 4. Measured solo acceptance ~55%
(`drafts 65-77 offered / 29-40 accepted`), so the windows commit roughly half the
tokens a full-depth window would. The trim math matches the flat 2x row and the
modest 4x row; the -3% solo row is the 3 GiB slot reserve shrinking the expert cache
(6393 vs 7332 slots), not code overhead.

Prerequisite landed: `74936a6ea` (server counts a slot's drafts from the BDONE tail -
batched requests showed `drafts 0/0` before it, hiding all of this). The next sweep
shows per-request batch acceptance directly.

## Scope

`kVerifyMaxT` 8 -> 16 (`include/strata/kernels/verify_kernels.hpp:23`). The kernels
do NOT template per T at the call sites that matter (T is a runtime bound with
fixed-size arrays), so this is a constant bump plus a bounded audit:

1. `src/kernels/cuda/verify_kernels.cu` - `float acc[kVerifyMaxT]` register arrays in
   the GDN segment kernels. THE risk: doubling per-thread accumulators may spill and
   slow solo decode (currently 126 tok/s). Measure; if it spills, restructure the
   loop (two 8-row passes inside the kernel over the same window) instead of reverting.
2. `src/kernels/cuda/s2_expert_grouped.cu` - `GMAX = 8` (entries per expert group)
   with `static_assert(GMAX >= kVerifyMaxT)`. Raise GMAX to 16; audit its smem/register
   sizing at the grouped-expert launch.
3. `src/kernels/cuda/fused_gr.cu` - `STRATA_UP(1..8)` / `STRATA_DOWN(1..4)` switch
   instantiations. Add 9..16 to the UP switches (DOWN chunks by DT and needs nothing).
4. `src/kernels/cuda/qsa_select.cu:930` - comment asserts n <= 8 "validated by
   run_slot_rows"; audit the kernel body for the 8-row assumption (warp layout).
5. `src/kernels/cpu/kq_avx2.cpp` - `constexpr int MAXT = 8` (CPU reference); raise and
   extend the parity tests to windows of 9..16 rows (gdn_seg_parity, gr_multi_parity,
   qsa select parity - whatever covers 1-8 today covers 1-16 after).
6. `src/core/verify.cpp` - window capture loop (buffers ~99 MiB at T=8, ~2x after;
   fits the 3072 reserve), `init_slots` bound, stamps allocation already uses the
   constant. Startup capture time roughly doubles for the extra T shapes - record it.
7. `serve/server.py` - the BSTAT parser guard `rows > windows * 8` must read 16 (or
   drop the hardcoded bound); `PARALLEL_MAX = 8` STAYS (it caps slot count, and the
   one-row-per-slot floor keeps 8 slots inside a 16-row window).
8. `batch_step` (generate.cpp) - trim loop unchanged (it is bound-generic); with 4
   slots x 5 = 20 it still trims, now to 16: 4 slots x 4 rows. Slots 5-8 become
   practical later; nothing here blocks them.

Out of scope: multi-wave windows (trim-to-fit is strictly better once the window
fits full drafts), per-slot grammar, >8 slots.

## Gates

- Solo digest byte-identical (the solo window is T <= 5; values cannot change - spills
  would only cost speed, and speed is the measured risk).
- Solo decode tok/s within noise of 126 (the register-pressure check, run in the same
  window as the sweep).
- Parity: single-launch 16-row window == per-sequence runs (extend the phase 4 tests).
- ctest 47/47 + the extended parity tests; `tools/batch_test.py` greedy GEN-vs-BGEN.
- The A/B sweep: expect the 2x row to move most (2 slots x 5 = 10 fits entirely:
  full draft depth for the first time), then 4x (4 x 4 rows).

## Cost and payoff

Kernel audit + tests: the bulk. Expected: 2x toward +20-30% (full drafts, no trim),
4x up from +16% by draft depth 1 -> 4; solo unchanged if no spills. If spills eat it,
the fallback (two-pass kernel) keeps the value with more work.
