#!/usr/bin/env python3
"""Batching test: the same prompts decoded alone (GEN, the usual path with drafts) and together in the batch
windows (BGEN), greedy.  Every slot's tokens must equal its solo tokens; prints the aggregate decode rate.

  python3 tools/batch_test.py --exe engine/strata --config strata-<model>.json --batch 4 --n 4 \
      --extra "--layer-split 12,24,36 --trim-stage-weights --pcie-frac 0 --adapt-every 1000000"
"""
import argparse, json, os, subprocess, sys, tempfile, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_tokenizer as ST  # noqa: E402

QUESTIONS = [
    "Explique en detail le fonctionnement d'un B-tree.",
    "Write a Python function that merges overlapping intervals, then explain it.",
    "Compare TCP and QUIC: handshake, congestion control, multiplexing.",
    "Raconte l'histoire du calcul de pi, d'Archimede a nos jours.",
    "What are the main causes of the French Revolution?",
    "Explain how a transformer's attention works, step by step.",
    "Donne une recette de tarte aux pommes, etape par etape.",
    "Describe the life cycle of a star like the Sun.",
    "Erklaere ausfuehrlich, wie ein Dieselmotor funktioniert.",
    "Write a Rust function that parses a CSV line with quoted fields, then explain it.",
    "What were the main consequences of the invention of the printing press?",
    "Explique le theoreme de Pythagore avec trois demonstrations differentes.",
    "Describe how the immune system responds to a viral infection.",
    "Explain the difference between processes and threads in an operating system.",
    "Beschreibe die Geschichte des Roemischen Reiches in Grundzuegen.",
    "How does public-key cryptography work? Give an example with RSA.",
]
# more than 16 slots: the same questions again, asked to answer in German
QUESTIONS = QUESTIONS + [q + " Answer in German." for q in QUESTIONS]


def tokenizer(path):
    t = Path(path)
    vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for s, i in vocab.items():
        tokens[i] = s
    merges = (t / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((t / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


class Engine:
    def __init__(self, exe, cfg, batch, extra_env, extra_args):
        args = list(cfg["args"])
        if len(cfg.get("gpu") or []) > 1:
            args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
        if batch:
            args += ["--batch", str(batch)]
        args += extra_args
        env = dict(os.environ, **extra_env)
        env["LD_LIBRARY_PATH"] = ":".join(cfg.get("lib_dirs", []) + [env.get("LD_LIBRARY_PATH", "")])
        if os.name == "nt":
            env["PATH"] = os.pathsep.join(cfg.get("lib_dirs", []) + [env.get("PATH", "")])
        self.log_path = os.environ.get("BATCH_TEST_LOG") or os.path.join(tempfile.gettempdir(), "batch_test_engine.log")
        self.log = open(self.log_path, "w")
        self.batch = batch
        self.ready = False
        self.info = {}   # the INFO key=value lines before READY; a batching engine announces batch_slots=N there
        self.p = subprocess.Popen([exe, "--serve", *args], cwd=cfg.get("cwd"), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1, env=env)
        for line in self.p.stdout:
            if line.startswith("READY"):
                self.ready = True
                return
            if line.startswith("INFO"):
                for kv in line.split()[1:]:
                    k, sep, v = kv.partition("=")
                    if sep:
                        self.info[k] = v
        # An engine without batch support dies on the unknown --batch before READY; with batch asked for that is a
        # clean skip (main), not a crash.  Without batch it stays the same hard failure.
        if batch:
            return
        raise SystemExit(f"the engine ended before READY - see {self.log_path}")

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def lines(self):
        for line in self.p.stdout:
            yield line.rstrip("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n", type=int, default=4, help="prompts (<= --batch)")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--skip-solo", action="store_true")
    ap.add_argument("--keys", default="", help='sampling keys for every request, e.g. "temperature=0.7 top_k=20"')
    ap.add_argument("--extra", default="", help='more engine arguments in one string, e.g. "--adapt-every 1000000"')
    ap.add_argument("--dump", default="", help="write every slot's tokens (solo and batch) as JSON: an A/B of two engine builds")
    ap.add_argument("--long-tokens", type=int, default=0,
                    help="the LAST prompt is read from about this many tokens of filler text first: its chunks run "
                         "beside the earlier slots' windows (the prompt path lends experts then)")
    ap.add_argument("--mt-min", default="1", help="STRATA_IQ_MT_MIN for the engine (1: exact; empty: the default)")
    ap.add_argument("--mtp", action="store_true", help="require the engine's MTP drafts (its INFO says spec > 1)")
    ap.add_argument("--batch-mtp", action="store_true",
                    help="the slots draft too (engine --batch-mtp; INFO must confirm batch_mtp=1, else skip): "
                         "BDONE's draft acceptance fields are then printed per slot")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])
    prompts = []
    for q in QUESTIONS[: a.n]:
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        if a.long_tokens and q is QUESTIONS[a.n - 1]:
            para = ("The committee reviewed item %d of the long agenda, noting the budget, the schedule and the open "
                    "risks, and asked the staff to report back next quarter. ")
            per = max(1, len(tok.encode(para % 100)))
            filler = "".join(para % i for i in range(a.long_tokens // per + 1))
            text = f"<|im_start|>user\n{filler}\n\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prompts.append(tok.encode(text, parse_special=True))
    env = {"STRATA_DECODE_TIMING": "1", **({"STRATA_VERIFY_PROFILE": "1"} if os.environ.get("PROF") else {})}
    if a.mt_min:
        env["STRATA_IQ_MT_MIN"] = a.mt_min
    eng = Engine(a.exe, cfg, a.batch, env, a.extra.split() + (["--batch-mtp"] if a.batch_mtp else []))
    if a.batch and not (eng.ready and "batch_slots" in eng.info):
        print(f"skip: engine has no batch slots (INFO: {eng.info})")
        try:
            eng.p.terminate()
        except OSError:
            pass
        return 0
    if a.mtp and int(eng.info.get("spec", "0") or 0) < 2:
        print(f"skip: engine has no MTP drafts (INFO: {eng.info})")
        eng.p.terminate()
        return 0
    if a.batch_mtp and eng.info.get("batch_mtp") != "1":
        print(f"skip: engine has no batch drafts (INFO: {eng.info})")
        eng.p.terminate()
        return 0
    out = eng.lines()

    solo = []
    solo_drafts = []
    if not a.skip_solo:
        for i, ids in enumerate(prompts):
            eng.send(" ".join(x for x in ("GEN", str(a.max_new), a.keys, ",".join(map(str, ids))) if x))
            got, t0 = [], time.time()
            for line in out:
                if line.startswith("T "):
                    got.append(int(line.split()[1]))
                elif line.startswith("DONE") or line.startswith("ERR"):
                    f = line.split()
                    if f[0] == "DONE" and len(f) >= 8:
                        solo_drafts.append((int(f[6]), int(f[7])))
                    print(f"solo {i}: {len(got)} tokens in {time.time() - t0:.1f}s  {line[:60]}", flush=True)
                    break
            solo.append(got)

    # batch: admit every prompt, then the slots decode together
    got = {i: [] for i in range(len(prompts))}
    done = {}
    drafts = {}   # slot -> (accepted, offered) from BDONE's trailing fields (--batch-mtp engines)
    t_admit = time.time()
    first_bt = None
    for i, ids in enumerate(prompts):
        eng.send(" ".join(x for x in ("BGEN", str(i), str(a.max_new), a.keys, ",".join(map(str, ids))) if x))
        for line in out:
            if line.startswith("T "):
                got[i].append(int(line.split()[1]))
            elif line.startswith("BT "):
                _, s, y = line.split()
                got[int(s)].append(int(y))
                first_bt = first_bt or time.time()
            elif line.startswith("BDONE "):
                p = line.split()
                done[int(p[1])] = line
                if len(p) >= 7:
                    drafts[int(p[1])] = (int(p[5]), int(p[6]))
            elif line.startswith("ERR"):
                print("batch:", line)
                return 1
            elif line.startswith("BADM "):
                if line.split()[2] == "0":
                    done[i] = line
                break
    t_admitted = time.time()
    for line in out:
        if line.startswith("BT "):
            _, s, y = line.split()
            got[int(s)].append(int(y))
            first_bt = first_bt or time.time()
        elif line.startswith("BDONE "):
            p = line.split()
            done[int(p[1])] = line
            if len(p) >= 7:
                drafts[int(p[1])] = (int(p[5]), int(p[6]))
            if len(done) == len(prompts):
                break
        elif line.startswith("ERR"):
            print("batch:", line)
            return 1
    t_end = time.time()
    total = sum(len(v) for v in got.values())
    print(f"batch: {len(prompts)} slots, {total} tokens; admissions {t_admitted - t_admit:.1f}s; "
          f"then {t_end - t_admitted:.1f}s -> aggregate {total / max(t_end - t_admit, 1e-9):.1f} tok/s overall, "
          f"{sum(len(v) for v in got.values()) / max(t_end - (first_bt or t_admit), 1e-9):.1f} tok/s from the first batch token",
          flush=True)
    ok = True
    for i in range(len(prompts)):
        b = got[i]
        if solo:
            same = b == solo[i]
            ok &= same
            first_diff = next((k for k in range(min(len(b), len(solo[i]))) if b[k] != solo[i][k]), None)
            print(f"slot {i}: {len(b)} tokens, solo {len(solo[i])}: {'IDENTICAL' if same else f'DIFFERS at {first_diff}'}")
        print("   ", ascii(tok.decode(b)[:160]))
    if solo_drafts:
        o_, t_ = sum(v[0] for v in solo_drafts), sum(v[1] for v in solo_drafts)
        print(f"solo drafts: {o_} of {t_} accepted ({o_ / max(t_, 1):.3f})")
    if drafts:
        for i in sorted(drafts):
            o_, t_ = drafts[i]
            print(f"slot {i} drafts: {o_} of {t_} accepted ({o_ / max(t_, 1):.3f})")
        o_, t_ = sum(v[0] for v in drafts.values()), sum(v[1] for v in drafts.values())
        print(f"batch drafts: {o_} of {t_} accepted ({o_ / max(t_, 1):.3f})", flush=True)
    if a.dump:
        Path(a.dump).write_text(json.dumps({"solo": solo, "batch": [got[i] for i in range(len(prompts))]}), encoding="utf-8")
    eng.send("QUIT")
    eng.p.wait(timeout=180)   # the next run needs the GPUs back
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
