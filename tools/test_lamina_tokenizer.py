"""Tests for tools/lamina_tokenizer.py: llama.cpp's reference vectors, the piece cache (0.1.41: the ids are the ones the
merge loop gives, cache cold, warm or switched off), and the incremental prompt encoder, which must give exactly the
ids of a plain encode.

    python -m unittest tools.test_lamina_tokenizer -v

The real vocabulary comes from $LAMINA_TOKENIZER (a pack's tokenizer/ directory) when set, else from llama.cpp's
`models/ggml-vocab-qwen35.gguf` next to the gguf-py the tools use ($LAMINA_GGUF_PY, third_party/llama.cpp); the
tests that need it are skipped without one.  The chat template's USER_DEFINED tokens (`<think>`, `<tool_call>`,
...) are added to the llama.cpp vocabulary, which does not have them, as the pack's has.
"""
from __future__ import annotations

import json
import os
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import lamina_tokenizer as ST  # noqa: E402

ADDED = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
         "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]


def vocab_gguf() -> Path | None:
    cands = [os.environ.get("LAMINA_GGUF_PY"), ROOT / "third_party" / "llama.cpp" / "gguf-py", "/opt/llama.cpp/gguf-py"]
    for c in cands:
        if c and (Path(c).parent / "models" / "ggml-vocab-qwen35.gguf").is_file():
            return Path(c).parent / "models" / "ggml-vocab-qwen35.gguf"
    return None


def load_tokenizer():
    pack = os.environ.get("LAMINA_TOKENIZER")
    if pack and (Path(pack) / "vocab.json").is_file():
        t = Path(pack)
        vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for s, i in vocab.items():
            tokens[i] = s
        return ST.Tokenizer(tokens, (t / "merges.txt").read_text(encoding="utf-8").split("\n"),
                            json.loads((t / "token_type.json").read_text()))
    path = vocab_gguf()
    return None if path is None else gguf_tokenizer(path)


def gguf_tokenizer(path):
    """llama.cpp's qwen35 vocabulary with the template's added tokens: the one its test vectors were made with."""
    from gguf_reader import GGUFFile
    md = GGUFFile(path).metadata
    tokens, types = list(md["tokenizer.ggml.tokens"]), list(md["tokenizer.ggml.token_type"])
    for s in ADDED:
        if s not in tokens:
            tokens.append(s)
            types.append(4)
    return ST.Tokenizer(tokens, list(md["tokenizer.ggml.merges"]), types)


def byte_tokenizer(specials: list[str]) -> ST.Tokenizer:
    """A vocabulary of the 256 byte tokens and `specials` (CONTROL), no merges: small enough to build per test."""
    tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)] + specials
    return ST.Tokenizer(tokens, [], [1] * 256 + [3] * len(specials))


WORDS = ["the", "tokenizer", " ", "  ", "\n", "\n\n", "\t", "def", "f(x):", "return", "x", "==", "->", "café",
         "你好", "\U0001f600", "é", "1234", "don't", "I'm", "<|im_", "<think", "</", "tool", "_call>",
         " ", "\r\n", "Hello,", "world!", "<", ">", "|", "      ", "x" * 70]


def random_text(rnd: random.Random, n: int, specials=()) -> str:
    parts = []
    for _ in range(n):
        parts.append(rnd.choice(WORDS) if not specials or rnd.random() > 0.05 else rnd.choice(specials))
        if rnd.random() < 0.5:
            parts.append(" ")
    return "".join(parts)


class Oracle(unittest.TestCase):
    """llama.cpp's own tokenizer test vectors for qwen35."""

    def test_llama_cpp_vectors(self):
        path = vocab_gguf()
        if path is None or not path.with_suffix(".gguf.inp").is_file():
            self.skipTest("llama.cpp's ggml-vocab-qwen35.gguf test vectors are not here")
        tk = gguf_tokenizer(path)                  # not $LAMINA_TOKENIZER: the vectors belong to this vocabulary
        inp = path.with_suffix(".gguf.inp").read_text(encoding="utf-8").split("\n__ggml_vocab_test__\n")
        out = path.with_suffix(".gguf.out").read_text(encoding="utf-8").split("\n")
        n = 0
        for text, want in zip(inp, out):
            if text.endswith("\n__ggml_vocab_test__"):
                text = text[:-len("\n__ggml_vocab_test__")]
            self.assertEqual(tk.encode(text), [int(x) for x in want.split()], repr(text[:40]))
            n += 1
        self.assertGreater(n, 40)


class Marks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tk = load_tokenizer()
        if cls.tk is None:
            raise unittest.SkipTest("no tokenizer (set LAMINA_TOKENIZER or LAMINA_GGUF_PY)")

    def test_marks(self):
        text = "a<|im_start|>user\nhi<|im_end|>\n<think>\n"
        ids, marks = self.tk.encode_marked(text, parse_special=True)
        self.assertEqual(ids, self.tk.encode(text, parse_special=True))
        self.assertEqual([m[0] for m in marks], [text.index("user"), text.index("<|im_end|>") + len("<|im_end|>"),
                                                  len(text) - 1])
        for end, n in marks:                            # each mark: the ids of the text up to it
            self.assertEqual(ids[:n], self.tk.encode(text[:end], parse_special=True))


class Margin(unittest.TestCase):
    """A special literal that is a prefix of a longer one: cutting at the shorter one's end without the look-ahead
    margin would give different ids."""

    def test_longer_literal_past_the_common_prefix(self):
        tk = byte_tokenizer(["<|a|>", "<|a|>zz"])
        enc = ST.PromptEncoder(tk)
        first = "x<|a|>y" + "<|a|>" * 3 + "q"
        self.assertEqual(enc.encode(first), tk.encode(first, parse_special=True))
        second = "x<|a|>y" + "<|a|>" * 3 + "zz"             # shares everything up to the last <|a|>
        got = enc.encode(second)
        self.assertEqual(got, tk.encode(second, parse_special=True))
        self.assertEqual(got[-1], tk.special_tokens["<|a|>zz"])
        self.assertGreater(enc.last_reused, 0)              # the earlier boundaries were still reused

    def test_literal_ending_inside_another(self):
        tk = byte_tokenizer(["<|a|>", "b<|a|>c"])
        enc = ST.PromptEncoder(tk)
        for text in ("zzb<|a|>d", "zzb<|a|>c", "zzb<|a|>cz<|a|>", "zzb<|a|>cz<|a|>b<|a|>c"):
            with self.subTest(text=text):
                self.assertEqual(enc.encode(text), tk.encode(text, parse_special=True))

    def test_random_specials(self):
        """Literals that extend or end inside one another, texts that change right after one of them."""
        rnd = random.Random(3)

        def word(lo, hi):
            return "".join(rnd.choice("<|ab>") for _ in range(rnd.randint(lo, hi)))
        for trial in range(60):
            a, b = word(2, 4), word(2, 4)
            lits = sorted({a, a + word(1, 3), b, word(1, 2) + b + word(0, 2)})
            tk = byte_tokenizer(lits)
            enc = ST.PromptEncoder(tk, keep=3)
            pieces = lits + [x[:rnd.randint(1, len(x))] for x in lits] + ["x", "y", "a", "|"]
            base = "".join(rnd.choice(pieces) for _ in range(60))
            for _ in range(30):
                cut = rnd.randint(0, len(base))
                text = base[:cut] + "".join(rnd.choice(pieces) for _ in range(rnd.randint(0, 6)))
                self.assertEqual(enc.encode(text), tk.encode(text, parse_special=True), (lits, text))
                if rnd.random() < 0.3:
                    base = text + base[cut:]


class CommonPrefix(unittest.TestCase):
    def test_against_a_loop(self):
        rnd = random.Random(5)
        for _ in range(500):
            a = "".join(rnd.choice("ab") for _ in range(rnd.randint(0, 9000)))
            k = rnd.randint(0, len(a))
            b = a[:k] + rnd.choice(["", "c", "cd"]) + a[k:k + rnd.randint(0, 50)]
            want = 0
            while want < min(len(a), len(b)) and a[want] == b[want]:
                want += 1
            self.assertEqual(ST.common_prefix_len(a, b), want)
            self.assertEqual(ST.common_prefix_len(b, a), want)



def toy() -> ST.Tokenizer:
    base = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
    merges = ["t h", "th e", "i n", "a n", "an d", "Ġ t", "Ġt h", "Ġth e", "e r", "o n"]
    merges = [m.replace("Ġ", ST.BYTE_TO_UNICODE[32]) for m in merges]
    tokens = list(base)
    for m in merges:
        a, b = m.split(" ")
        tokens.append(a + b)
    return ST.Tokenizer(tokens, merges)


class PieceCache(unittest.TestCase):
    def test_same_ids_cold_warm_and_off(self):
        rng = random.Random(7)
        words = ["the", "and", "in", "on", "there", "other", "x" * 70, "你好", "def", "return", "\n\n", "  "]
        text = "".join(rng.choice(words) + rng.choice([" ", "", "\n", "\t"]) for _ in range(3000))
        a = toy()
        a.PIECE_CACHE_MAX = 0
        want = a.encode(text)
        b = toy()
        cold = b.encode(text)
        warm = b.encode(text)
        self.assertEqual(cold, want)
        self.assertEqual(warm, want)
        self.assertGreater(len(b._piece_ids), 0)
        self.assertEqual(b.decode(warm), text)

    def test_the_cache_is_bounded(self):
        t = toy()
        t.PIECE_CACHE_MAX = 5
        t.encode(" ".join("w%d" % i for i in range(100)))
        self.assertLessEqual(len(t._piece_ids), 5)


class ByteCache(unittest.TestCase):
    def test_decode_before_encode_cold_and_warm(self):
        t = toy()
        text = "café 日本語 🙂\r\n\x00"
        ids = list(text.encode("utf-8"))  # toy's first 256 token IDs are the raw bytes
        self.assertEqual(t.decode(ids, errors="strict"), text)
        self.assertEqual(t.decode(ids, errors="strict"), text)
        self.assertEqual(t.decode([]), "")
        for i in range(256):
            self.assertEqual(t.token_bytes(i), bytes([i]))

    def test_invalid_ids_before_and_after_cache_use(self):
        t = toy()
        for _ in range(2):
            for i in (-1, len(t.tokens), len(t.tokens) + 100):
                with self.assertRaises(IndexError):
                    t.token_bytes(i)
            self.assertEqual(t.token_bytes(65), b"A")

    def test_instances_with_different_vocabularies_stay_independent(self):
        tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
        a, b = ST.Tokenizer(tokens, []), ST.Tokenizer(tokens[::-1], [])
        text = "the café 日本語 🙂\n"
        for t in (a, b, a, b):
            expected = [t.ids[ST.BYTE_TO_UNICODE[byte]] for byte in text.encode("utf-8")]
            self.assertEqual(t.encode(text), expected)
            self.assertEqual(t.decode(expected, errors="strict"), text)
        self.assertEqual(a.token_bytes(0), b"\x00")
        self.assertEqual(b.token_bytes(0), b"\xff")

    def test_shared_tokenizer_concurrent_first_decode(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        t = toy()
        texts = ["café", "日本語", "🙂\r\n", "\x00plain"]
        barrier = threading.Barrier(len(texts))
        def decode(text):
            ids = list(text.encode("utf-8"))
            barrier.wait(timeout=5)
            for _ in range(100):
                if t.decode(ids, errors="strict") != text:
                    raise AssertionError("concurrent decode changed the text")
            return text
        with ThreadPoolExecutor(len(texts)) as pool:
            self.assertEqual(list(pool.map(decode, texts)), texts)


if __name__ == "__main__":
    unittest.main()
