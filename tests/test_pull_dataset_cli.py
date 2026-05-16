"""CLI surface tests for scripts/pull_dataset.py.

Subprocess-based tests verify argparse construction + subcommand --help
work without any of the script's heavy function-level deps (pyarrow,
soundfile, huggingface_hub) being installed. argparse builds the parser
at module import time but the subcommand handlers only import their
deps when called — so --help is safe on a bare clone.

These tests serve as regression guards: if a future edit drops one of
the new Phase 1 flags from argparse, the test fails fast.

Run:
    python -m unittest tests.test_pull_dataset_cli -v
    make test
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "pull_dataset.py"


def _run(*cli_args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *cli_args],
        capture_output=True, text=True, env=env, timeout=15,
    )


class TestKirtanHelp(unittest.TestCase):
    def test_help_exits_zero(self):
        r = _run("kirtan", "--help")
        self.assertEqual(r.returncode, 0, msg=r.stderr)

    def test_help_lists_phase_1_flags(self):
        """Every Phase 1 puller flag must appear in `kirtan --help`. If a
        future edit drops one, this test fails fast — the alternative is
        discovering it at training-machine launch time."""
        r = _run("kirtan", "--help")
        for flag in (
            # 1.A: registry-driven holdout is implicit; no new flag here.
            # 1.B: split flags
            "--split-by", "--split-ratios", "--split-seed",
            # 1.C: duration filter
            "--min-duration-s", "--max-duration-s",
        ):
            self.assertIn(flag, r.stdout, msg=f"--help missing {flag}")

    def test_sehaj_and_sehajpath_have_same_flags(self):
        """All three SURT subcommands share the _add_surt_args helper, so
        they must expose the same Phase 1 flags. Cheap to verify, catches
        accidental divergence."""
        for sub in ("sehaj", "sehajpath"):
            r = _run(sub, "--help")
            self.assertEqual(r.returncode, 0, msg=f"{sub} --help failed: {r.stderr}")
            for flag in ("--split-by", "--min-duration-s", "--max-duration-s"):
                self.assertIn(flag, r.stdout, msg=f"{sub} --help missing {flag}")


class TestRatioValidation(unittest.TestCase):
    """--split-ratios validation fires at argparse time (type=_parse_split_ratios).
    Confirm bad input is rejected with a clear message."""

    def test_rejects_wrong_count(self):
        r = _run("kirtan", "--out-dir", "/tmp/x", "--split-ratios", "0.5,0.5")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("3 comma-separated floats", r.stderr)

    def test_rejects_not_summing_to_one(self):
        r = _run("kirtan", "--out-dir", "/tmp/x", "--split-ratios", "0.5,0.3,0.3")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("must sum to 1.0", r.stderr)


if __name__ == "__main__":
    unittest.main()
