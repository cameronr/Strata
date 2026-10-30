"""tools/qwen35/kl.py - teacher-forced KL for Qwen3.6-35B-A3B (NEW-MODELS-PLAN.md T1/T2): on one token sequence per
prompt (the prompt + llama.cpp GPU's greedy answer, from a parity.py run), the next-token distributions of
llama.cpp HIP (the reference P), llama.cpp CPU (the noise floor) and lamina-q36ref.  Reported: mean KL(P || Q) and
the top-1 agreement over all steps, per engine; the plan's gate is Lamina's KL within 1.5x of the floor.

    python tools/qwen35/kl.py --ref BIN --dump BIN --model GGUF --parity DIR --out DIR [--lock "..."] [--cpu-lock "..."]
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

V = 248320
EOS = 248046                                    # <|im_end|>


def ids(path: Path, key: str) -> list[int]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(key + ":"):
            return [int(x) for x in line.split(":", 1)[1].split()]
    raise SystemExit(f"no {key}: in {path}")


def run(cmd: list[str], lock: str, log: Path) -> None:
    r = subprocess.run((shlex.split(lock) if lock else []) + cmd, capture_output=True, text=True, timeout=3600)
    log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed (rc {r.returncode}), see {log}")


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--parity", required=True, help="a parity.py output dir (its llama.cpp logs give the sequences)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lock", default="")
    ap.add_argument("--cpu-lock", default="")
    ap.add_argument("--cache", default="", help="where llama.cpp's logits are kept between runs (default --out)")
    ap.add_argument("--ref-args", default="", help="extra lamina-q36ref arguments (e.g. '--act f32')")
    a = ap.parse_args()
    out, par = Path(a.out), Path(a.parity)
    out.mkdir(parents=True, exist_ok=True)
    tags = sorted(p.name.split(".")[0] for p in par.glob("p*.llama.log"))
    kl = {"cpu": [], "lamina": []}
    top1 = {"cpu": [], "lamina": []}
    kl_ans = {"cpu": [], "lamina": []}
    top1_ans = {"cpu": [], "lamina": []}
    for tag in tags:
        prompt, gen = ids(par / f"{tag}.llama.log", "prompt"), ids(par / f"{tag}.llama.log", "gen")
        force = ",".join(map(str, gen))
        toks = ",".join(map(str, prompt))
        cache = Path(a.cache) if a.cache else out
        f = {k: (out if k == "lamina" else cache) / f"{tag}.{k}.full" for k in ("gpu", "cpu", "lamina")}
        # llama.cpp's two runs depend only on the model and the sequence: kept in --out and reused (64 MB each)
        if not f["gpu"].exists():
            run([a.dump, "--model", a.model, "--tokens", toks, "--force", force, "--ngl", "99", "--dump-full-logits",
                 str(f["gpu"])], a.lock, cache / f"{tag}.gpu.log")
        if not f["cpu"].exists():
            run([a.dump, "--model", a.model, "--tokens", toks, "--force", force, "--ngl", "0", "--dump-full-logits",
                 str(f["cpu"])], a.cpu_lock, cache / f"{tag}.cpu.log")
        run([a.ref, "--model", a.model, "--tokens", toks, "--force", force, "--dump-full-logits", str(f["lamina"]),
             *shlex.split(a.ref_args)], a.lock, out / f"{tag}.lamina.log")
        P = log_softmax(np.fromfile(f["gpu"], dtype=np.float32).reshape(-1, V))
        # the answer: the steps up to and including the first <|im_end|> of the forced sequence (past it the context is
        # one no chat reaches, and its distributions are the noisiest); "all" keeps every step
        ans = gen.index(EOS) + 1 if EOS in gen else len(P)
        line = f"{tag}: {len(P)} steps ({ans} answer)"
        per = {}
        for k in ("cpu", "lamina"):
            Q = log_softmax(np.fromfile(f[k], dtype=np.float32).reshape(-1, V))
            n = min(len(P), len(Q))
            d = (np.exp(P[:n]) * (P[:n] - Q[:n])).sum(axis=1)
            t1 = P[:n].argmax(axis=1) == Q[:n].argmax(axis=1)
            kl[k].extend(d.tolist())
            top1[k].extend(t1.tolist())
            kl_ans[k].extend(d[:ans].tolist())
            top1_ans[k].extend(t1[:ans].tolist())
            per[k] = d
            line += f"; {k}: KL {d.mean():.5f} (answer {d[:ans].mean():.5f}) top-1 {t1.mean():.3f}"
        np.savez(out / f"{tag}.kl.npz", answer=ans, **per)
        f["lamina"].unlink(missing_ok=True)
        print(line, flush=True)
    for name, K, T in (("all steps", kl, top1), ("answers only", kl_ans, top1_ans)):
        print(f"-- {name}")
        for k in ("cpu", "lamina"):
            print(f"{k:6s} vs llama.cpp HIP: mean KL {np.mean(K[k]):.5f} (median {np.median(K[k]):.6f}, p99 "
                  f"{np.percentile(K[k], 99):.4f}), top-1 {np.mean(T[k]):.4f} over {len(K[k])} steps")
        print(f"ratio lamina / floor: {np.mean(K['lamina']) / max(np.mean(K['cpu']), 1e-12):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
