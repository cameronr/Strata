#!/usr/bin/env python3
"""Solo token-exactness baseline: the engine's per-window digests (STRATA_WINDOW_DIGEST) over a fixed greedy
prompt set, compared across fresh engine processes.  A build whose solo decode shifts a token shifts the chains,
and this exits 1 at the first divergent line.

Each repeat runs in its own process on purpose.  In one process the prompt-lookup drafter and the conversation
state carry across requests, so a prompt sent twice gets different drafts the second time; window shapes then
differ and (this engine's numerics being window-shape dependent) even greedy tokens can move.  Cross-process is
the determinism a build gate needs; in-process repeats test nothing.

  python3 tools/window_digest_test.py --exe build/bin/strata --config strata-<model>.json [--repeats 2] \
      [--digest-out digests.txt]

--digest-out records the digest stream for comparison across engine builds (phases of the batching plan).  The
file is model- and flags-specific: record it, do not commit it.  Chains restart per request, so only the request
counter differs between repeats of the same prompt.
"""
import argparse, json, os, sys, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_test as BT  # noqa: E402

N_PROMPTS = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.environ.get("STRATA_EXE", ""), help="the engine binary (or STRATA_EXE)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--digest-out", default="")
    ap.add_argument("--mt-min", default="1",
                    help="STRATA_IQ_MT_MIN for the engine (1: exact, output independent of the drafting; empty: the default)")
    a = ap.parse_args()
    if not a.exe:
        ap.error("--exe or STRATA_EXE is required")
    cfg = json.loads(Path(a.config).read_text())
    tok = BT.tokenizer(cfg["tokenizer"])
    prompts = []
    for q in BT.QUESTIONS[:N_PROMPTS]:
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prompts.append(tok.encode(text, parse_special=True))

    n = len(prompts)
    digest, tokens = [], []
    for r in range(a.repeats):
        os.environ["BATCH_TEST_LOG"] = f"/tmp/ph1-digest-rep{r}.log"   # Engine reads it from the environment
        env = {"STRATA_WINDOW_DIGEST": "1"}
        if a.mt_min:
            env["STRATA_IQ_MT_MIN"] = a.mt_min
        eng = BT.Engine(a.exe, cfg, None, env, [])
        out = eng.lines()
        stream = []
        for ids in prompts:
            eng.send(f"GEN {a.max_new} {','.join(map(str, ids))}")
            got = []
            for line in out:
                if line.startswith("T "):
                    got.append(int(line.split()[1]))
                elif line.startswith("DONE") or line.startswith("ERR"):
                    break
            stream.append(got)
        tokens.append(stream)
        eng.send("QUIT")
        try:
            eng.p.wait(timeout=180)
        except subprocess.TimeoutExpired:
            eng.p.kill()
        eng.log.close()
        digest.append([l.rstrip("\n") for l in open(eng.log_path) if l.startswith("strata digest:")])

    ok = True
    if len(digest[0]) == 0:
        print("FAIL: the engine produced no digest lines")
        ok = False
    for r in range(1, a.repeats):
        for i, (d0, dr) in enumerate(zip(digest[0], digest[r])):
            if d0 != dr:
                ok = False
                print(f"FAIL: processes 0 and {r} diverge at digest line {i}:\n  0: {d0}\n  {r}: {dr}")
                break
        if len(digest[0]) != len(digest[r]):
            ok = False
            print(f"FAIL: processes 0 and {r} emitted {len(digest[0])} vs {len(digest[r])} digest lines")
        for i in range(n):
            if tokens[0][i] != tokens[r][i]:
                ok = False
                d = next((k for k in range(min(len(tokens[0][i]), len(tokens[r][i])))
                          if tokens[0][i][k] != tokens[r][i][k]), None)
                print(f"FAIL: processes 0 and {r} token streams differ for prompt {i} at token {d}")
    if a.digest_out:
        Path(a.digest_out).write_text("\n".join(digest[0]) + "\n", encoding="utf-8")
    if ok:
        finals = " ".join(l.split("chain=")[1] for l in digest[0] if "win=0 " not in l)[-64:]
        print(f"ok: {n} requests x {a.repeats} fresh processes, {len(digest[0])} windows, cross-process stable; "
              f"tail chains: ...{finals}")
        print("   ", ascii(tok.decode(tokens[0][0])[:80]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
