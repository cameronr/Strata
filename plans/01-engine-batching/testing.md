# Testing strategy

Back-link: [overview.md](overview.md).

Three gates, every phase, in this order.

1. **Solo digests unchanged.** The phase-1 golden digest stream must match any build. A phase that shifts solo tokens fails, whatever else it gains.
2. **Batch parity.** Two-sequence windows equal two solo runs token-for-token (phase 3 harness), single-launch kernels bitwise equal per-segment runs (phase 4 suites), `tools/batch_test.py` GEN-vs-BGEN greedy equality from phase 5 onward, MTP parity from phase 7.
3. **The rig.** Everything runs on the 3090 + 5070 Ti box, not just CI. Baseline table (solo tok/s, TTFT, then 1/2/4-slot numbers) grows one row per user-visible phase and is the record for the docs work in phase 9.

Failure policy: a gate that cannot pass stops the phase; the phase branch stays open with the failing evidence recorded. No gate gets weakened to make a phase pass, and no parity skip gets added without a hardware reason named in the test.

Reference material in the scratch clone of Basalt (`/tmp/opencode/fork-analysis/basalt`, disposable) and Niko1221/Strata main: the harness (`tools/batch_test.py`, batch_interleave tests), digests (`--exp concur5_digest`, `concur9_logit_hash`), parity suites (`qsa_slots_parity`), and the protocol wire format.
