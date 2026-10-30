"""Tests for serve/server.py's effort_end_args (#458): "effort_position": "end" is used when the engine binary knows
--tail-role-token.  Fake files in a temp folder - nothing is built or run.

    python -m unittest tools.test_effort_end_args
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import server as S  # noqa: E402


class Tok:
    def encode(self, text, parse_special=False):
        return [42] if text == "system" else [1, 2]


ENGINE_WITH = b"\x7fELF ... --tail-role-token ... --serve ..."
ENGINE_WITHOUT = b"\x7fELF ... --serve ..."


class EffortEndArgs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, cfg, exe):
        with contextlib.redirect_stdout(io.StringIO()):
            return S.effort_end_args(cfg, str(exe), Tok())

    def test_a_plain_engine_is_read_as_before(self):
        exe = Path(self.tmp.name) / "strata"
        exe.write_bytes(ENGINE_WITH)
        self.assertEqual(self.call({"effort_position": "end"}, exe), ["--tail-role-token", "42"])
        exe.write_bytes(ENGINE_WITHOUT)
        self.assertIsNone(self.call({"effort_position": "end"}, exe))

    def test_start_is_the_default(self):
        exe = Path(self.tmp.name) / "lamina"
        exe.write_bytes(ENGINE_WITH)
        self.assertIsNone(self.call({}, exe))
        self.assertIsNone(self.call({"effort_position": "start"}, exe))


if __name__ == "__main__":
    unittest.main()
