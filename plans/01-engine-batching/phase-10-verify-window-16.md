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

## As built (2026-10-29, a12958b8a)

Landed as scoped, with one addition: `gu_grouped_kernel` (the plain variant) needed
the EMAX=8 sub-batch staging too - at GMAX 16 its 45 KB of static smem would have
halved occupancy for EVERY window including solo's 5-row ones (static smem is sized
by the constant, not by the group's actual entry count). Both gate/up kernels hold
23/25 KB; `gr_up_multi_kernel<16>` 106 REG / 32 KB; `gdn_ab_multi_kernel` 48 REG;
no spills anywhere (cuobjdump). Solo never instantiates a >8 kernel: each T is its
own template instantiation and the solo window is T <= 5.

CPU gates pass (full build 49/49 targets; batch_read/slot_fit/iq_avx2/bf16_bits/
platform_memory green - the last needs `--ulimit memlock=-1`, unrelated).
The GPU gates (digest, parity suite, decode speed, sweep) run in the next window:
the same `docker stop strata-eddoursul; docker run --gpus all ... strata:src`
invocation as the phase 9 gates, then `./bench-batch-ab.py --yes --rounds 2`.

## As built (2026-10-30, the GPU gates) - three latent bugs surfaced by >8

The first gate run NaN'd (`non-finite logits`) at feed widths >= 11 and the
interleave gate died with a SIGSEGV inside `cuStreamSynchronize`. Both were
pre-existing contract breaks of the 8 -> 16 bump, found with a per-layer NaN
probe inside the captured window (env-gated, scratch) and a gdb replay of the
interleave sequence:

- **The interleaved q8_1 copy is 8 columns by contract** (`native_q8_1_il_cp`
  caps at 8). The `ncols` 9..16 instantiations quantized columns 8.. past each
  slot (stomping the pm/dl regions) and read `o[8..]` uninitialized: NaN at 11+,
  silently-corrupted-but-finite values at 9..10. Fixed by routing `ncols > 8`
  to `native_mmvq`'s multi kernels (46d552afd), guards + `static_assert`
  enforcing the cap, and the window/split-head/shared-expert quantizing plain
  above 8. `mmvq_multi_parity` now covers 10/12/16.
- **`ahead_tok_` stayed `int32_t[8]`** while `ple_ahead` guards against
  `max_t_` (16): windows >8 overflowed the stack-allocated Verifier - the
  silent SIGSEGV (208e48ad9). The rest of the host-side widening (graph-exec
  and groups arrays, MTP draft steps, CPU pool multi kernels chunking nt > 4
  by 4) landed with it (fdafefb83).
- **The verify router routed one row per warp**, so rows 8..15 of a wide
  window were never routed (1e47e71dc).

Gates (deterministic config, frozen profile): solo digests byte-identical with
MTP off and on, GEN-vs-BGEN identical, the interleave gate 10/10 IDENTICAL,
startup scenarios and the adapt-rhythm smoke green, ctest 47/47 with the
extended parity suites.

Sweep (medians, 2 rounds, 2026-10-30): solo 126.5 / batch 120.2 chat tok/s
(-5%, within the old baseline's noise); 2x 82.9 -> 89.8 (+8%), 4x 86.4 -> 96.5
(+12%); 4x TTFT p50/p95 2.5/4.8 -> 1.0/1.7 s. The 2x row did not reach the
+20-30% the plan hoped for: 2 slots x 5 rows fit the window even before this
phase (the trim only bit at 3+ slots), and at 4 slots the draft depth, not the
window, is the binding constraint. The window itself costs nothing on solo.

## As built (2026-10-30, 48ddce97c) - the 16-row window regressed 4x; default capped at 8

A user-reported 4x regression confirmed against the pre-p10 baseline: old 4x
medians 101.8/104.3/96.8/107.2/105.9 vs 93.7-97.6 after this phase. A
same-binary alternating A/B pinned the cause on the window itself:
`STRATA_BATCH_WINDOW=8` -> 104.7/105.5, `=16` -> 98.0/97.4. 2x stayed flat
(88.1 vs 88.3), which is why the sweep's two-round medians missed it.

This refines the "draft depth binds" note above. At 4 slots the 16-row window
is not free: windows over 8 columns lose the interleaved q8_1 fast path (this
phase routed `ncols > 8` to `native_mmvq`), and the extra verification rows
outweigh the deeper drafts they let the scheduler batch. The draft depth still
binds the 2x upside; it just no longer explains 4x.

Shipped: the `batch_step` trim defaults to an 8-row window. The
`STRATA_BATCH_WINDOW` env knob raises it, capped at `kVerifyMaxT`. Raising the
default again needs a
>8-column interleaved fast path or drafts cheap enough that the wide-window
verification pays; until then 16 stays an experiment knob, not the default.
Post-cap sanity (default build, no env, 2 rounds): 4x median 104.6 tok/s
(98.3-106.2), back on the pre-p10 distribution.
