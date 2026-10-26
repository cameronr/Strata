#!/usr/bin/env python3
"""Solo token-exactness baseline: the engine's per-window digests (STRATA_WINDOW_DIGEST) over a fixed greedy
prompt set, run twice, compared request by request.  A build whose solo decode shifts a token shifts the chains,
and this exits 1 at the first divergent line.

  python3 tools/window_digest_test.py --exe build/bin/strata --config strata-<model>.json [--repeats 2] \
      [--digest-out digests.txt]

--digest-out records the digest stream for comparison across engine builds (phases of the batching plan).  The
file is model- and flags-specific: record it, do not commit it.  Chains restart per request, so only the request
counter differs between repeats of the same prompt.
"""
import argparse, json, os, re, sys, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_test as BT  # noqa: E402

N_PROMPTS = 6


def groups_of(digest_lines):
    groups = []
    for line in digest_lines:
        req = next(int(t[4:]) for t in line.split() if t.startswith("req="))
        if not groups or groups[-1][0] != req:
            groups.append((req, []))
        groups[-1][1].append(re.sub(r"req=\d+ ", "", line))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.environ.get("STRATA_EXE", ""), help="the engine binary (or STRATA_EXE)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--digest-out", default="")
    a = ap.parse_args()
    if not a.exe:
        ap.error("--exe or STRATA_EXE is required")
    cfg = json.loads(Path(a.config).read_text())
    tok = BT.tokenizer(cfg["tokenizer"])
    prompts = []
    for q in BT.QUESTIONS[:N_PROMPTS]:
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prompts.append(tok.encode(text, parse_special=True))

    eng = BT.Engine(a.exe, cfg, None, {"STRATA_WINDOW_DIGEST": "1"}, [])
    out = eng.lines()
    tokens = []
    for _ in range(a.repeats):
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
    digest = [l.rstrip("\n") for l in open(eng.log_path) if l.startswith("strata digest:")]

    groups = groups_of(digest)
    n = len(prompts)
    ok = True
    if len(groups) != a.repeats * n:
        print(f"FAIL: {len(groups)} digest groups, expected {a.repeats * n} "
              f"(a request produced no windows or the engine died)")
        ok = False
    if ok:
        for r in range(1, a.repeats):
            for i in range(n):
                w0, wr = groups[i][1], groups[r * n + i][1]
                if w0 != wr:
                    ok = False
                    d = next((k for k in range(min(len(w0), len(wr))) if w0[k] != wr[k]), min(len(w0), len(wr)))
                    print(f"FAIL: prompt {i} repeats 0 and {r} diverge at window {d}:\n"
                          f"  repeat 0: {w0[d] if d < len(w0) else '<none>'}\n"
                          f"  repeat {r}: {wr[d] if d < len(wr) else '<none>'}")
                if tokens[0][i] != tokens[r][i]:
                    ok = False
                    d = next((k for k in range(min(len(tokens[0][i]), len(tokens[r][i])))
                              if tokens[0][i][k] != tokens[r][i][k]), None)
                    print(f"FAIL: prompt {i} token streams differ at token {d}")
    if a.digest_out:
        Path(a.digest_out).write_text("\n".join(digest) + "\n", encoding="utf-8")
    if ok:
        finals = " ".join(g[1][-1].split("chain=")[1] for g in groups if g[1])
        print(f"ok: {len(groups)} requests, {len(digest)} windows digested, repeat-stable; final chains: {finals}")
        print("   ", ascii(tok.decode(tokens[0][0])[:80]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
