#!/usr/bin/env python3
"""Phase 3 parity: the same conversations decoded solo (GEN, the usual path with MTP and lookup drafts) and
together in batch windows (the engine's STRATA_BATCH_PARITY script), greedy, token for token equal.

  python3 tools/batch_parity_test.py --exe engine/strata --config .ph3-det.json --n 2 --max-new 48 \
      --context 8192

Two engine processes on the same deterministic config (the tiers must be arg-determined: --adapt-every 0,
--pcie-frac 0, --expert-cache pinned below the free-VRAM cap, --second-gpu-gib fixed, STRATA_IQ_MT_MIN=1):
one serves the solo GEN legs, the other runs the batch script.  A batch row's arithmetic is the one-token
window's, so a slot's batch tokens must equal its solo tokens exactly - that is the claim this checks.
`--context` shrinks the context so the slot sessions fit the VRAM the sizing block left (see the phase 2
gate: at 32K prod settings two slots do not fit this rig).
"""
import argparse, json, os, shlex, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_test as BT  # noqa: E402


def solo_streams(exe, cfg, prompts, max_new, extra_args, log):
    env = {"STRATA_IQ_MT_MIN": "1"}
    eng = BT.Engine(exe, cfg, 0, env, extra_args)
    if not eng.ready:
        raise SystemExit(f"solo engine never became ready - see {eng.log_path}")
    out = eng.lines()
    streams = []
    for ids in prompts:
        eng.send(f"GEN {max_new} " + ",".join(str(i) for i in ids))
        got = []
        for line in out:
            if line.startswith("T "):
                got.append(int(line.split()[1]))
            elif line.startswith("DONE") or line.startswith("ERR"):
                if line.startswith("ERR"):
                    raise SystemExit(f"solo GEN failed: {line} - see {log}")
                break
        streams.append(got)
    eng.send("QUIT")
    try:
        eng.p.stdin.close()
    except OSError:
        pass
    eng.p.wait(timeout=300)
    eng.log.close()
    return streams


def batch_streams(exe, cfg, prompts, max_new, extra_args, work, log, shuffle=False):
    script = Path(work) / "parity-script.txt"
    out_file = Path(work) / "parity-out.txt"
    lines = [f"SLOTS {len(prompts)}", f"N {max_new}"]
    if shuffle:
        lines.append("SHUFFLE 1")   # rotate the row -> slot map every window (phase 4 S-key check)
    for i, ids in enumerate(prompts):
        lines.append(f"PROMPT {i} " + " ".join(str(t) for t in ids))
    lines.append(f"OUT {out_file}")
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    env = {"STRATA_IQ_MT_MIN": "1", "STRATA_BATCH_PARITY": str(script)}
    eng = BT.Engine(exe, cfg, len(prompts), env, extra_args)   # --batch N: the slot sizing (phase 2)
    rc = eng.p.wait(timeout=3600)
    log_tail = Path(log).read_text(errors="replace").splitlines()[-25:] if Path(log).exists() else ["(no log)"]
    if rc != 0:
        print("batch leg engine log tail:\n    " + "\n    ".join(log_tail))
        raise SystemExit(f"the batch script leg exited rc={rc} - see {log}")
    streams = []
    for line in out_file.read_text().splitlines():
        streams.append([int(t) for t in line.split()[1:]])
    if shuffle:
        # gate integrity: a silently unparsed SHUFFLE directive would degrade to an identity-map pass
        if "rows rotated per window" not in Path(log).read_text(errors="replace"):
            raise SystemExit(f"--shuffle was requested but the engine log has no rotation marker - see {log}")
    return streams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=2, help="conversations (batch slots)")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--context", type=int, default=8192, help="--max-context for both legs")
    ap.add_argument("--work-dir", default=".batch-parity")
    ap.add_argument("--shuffle", action="store_true",
                    help="rotate the row -> slot map every window (the phase-4 S-keyed-graph check)")
    ap.add_argument("--extra", default="", help="extra engine args, e.g. a smaller --expert-cache to fit slots")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = BT.tokenizer(cfg["tokenizer"])
    prompts = []
    for q in BT.QUESTIONS[: a.n]:
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prompts.append(tok.encode(text, parse_special=True))
    work = Path(a.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    extra = ["--max-context", str(a.context)] + shlex.split(a.extra)
    t0 = time.time()
    os.environ["BATCH_TEST_LOG"] = str(work / "solo-engine.log")
    solo = solo_streams(a.exe, cfg, prompts, a.max_new, extra, str(work / "solo-engine.log"))
    print(f"solo legs done ({time.time() - t0:.0f}s): " + ", ".join(f"{len(s)} tok" for s in solo))
    t1 = time.time()
    os.environ["BATCH_TEST_LOG"] = str(work / "batch-engine.log")
    batch = batch_streams(a.exe, cfg, prompts, a.max_new, extra, work, str(work / "batch-engine.log"), a.shuffle)
    print(f"batch leg done ({time.time() - t1:.0f}s): " + ", ".join(f"{len(s)} tok" for s in batch))
    ok = True
    if len(solo) != len(batch) or len(batch) != len(prompts):
        raise SystemExit(f"legs disagree on slot count: solo {len(solo)}, batch {len(batch)}, prompts {len(prompts)}")
    if any(len(s) == 0 for s in batch) or any(len(s) == 0 for s in solo):
        raise SystemExit("an empty token stream: the legs produced nothing to compare (engine died mid-run?)")
    for i, (s, b) in enumerate(zip(solo, batch)):
        if s == b:
            print(f"slot {i}: equal ({len(s)} tokens)")
            continue
        ok = False
        d = next((k for k in range(min(len(s), len(b))) if s[k] != b[k]), min(len(s), len(b)))
        print(f"slot {i}: DIFFER at token {d} (solo {len(s)}, batch {len(b)})")
        print("  solo :", ascii(tok.decode(s[max(0, d - 4):d + 8])))
        print("  batch:", ascii(tok.decode(b[max(0, d - 4):d + 8])))
    print("BATCH-PARITY " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
