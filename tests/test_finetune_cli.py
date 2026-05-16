"""CLI / argparse tests for scripts/finetune_path_b.py.

Uses subprocess so we exercise the actual CLI surface a user / Makefile
sees, including the YAML config two-pass parsing. No torch/transformers
needed — every test path here exits before the heavy imports (via --help
or argparse.error() before `import torch`).

Run:
    python -m unittest tests.test_finetune_cli -v
    pytest tests/test_finetune_cli.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "finetune_path_b.py"


def _run(*cli_args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(_SCRIPT), *cli_args]
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, env=full_env, timeout=30)


class TestHelp(unittest.TestCase):
    def test_help_exits_zero(self):
        """--help is the smoke test for argparse construction."""
        r = _run("--help")
        self.assertEqual(r.returncode, 0, msg=r.stderr)

    def test_help_lists_new_flags(self):
        """Every new Phase 0 flag must appear in --help output (regression
        guard if a future edit drops one)."""
        r = _run("--help")
        # Spot-check each Phase 0.A addition.
        for flag in ("--seed", "--weight-decay", "--max-grad-norm",
                     "--lr-scheduler-type", "--warmup-ratio",
                     "--gradient-checkpointing", "--eval-strategy",
                     "--eval-steps", "--early-stopping-patience",
                     "--load-best-model-at-end", "--report-to",
                     "--wandb-project", "--run-name"):
            self.assertIn(flag, r.stdout, msg=f"--help missing {flag}")


class TestValidationGuards(unittest.TestCase):
    """The script must reject misconfigured arg combos BEFORE Trainer init,
    with messages the user can act on. We trigger each failure path and
    inspect stderr."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.manifest = self.tmpdir / "manifest.json"
        # Minimal valid manifest — won't be read because validation errors
        # fire before manifest loading.
        self.manifest.write_text("[]")
        self.out = self.tmpdir / "out"

    def tearDown(self):
        self._tmp.cleanup()

    def _base_args(self) -> list[str]:
        return [
            "--manifest", str(self.manifest),
            "--output-dir", str(self.out),
            "--model-id", "openai/whisper-tiny",  # trigger Whisper path
        ]

    def test_load_best_without_eval_strategy_errors(self):
        """load_best_model_at_end=True + eval_strategy=no must error."""
        r = _run(*self._base_args(),
                 "--eval-strategy", "no",
                 "--load-best-model-at-end")
        self.assertNotEqual(r.returncode, 0)
        # The error names the failing knob so the user knows what to fix.
        self.assertIn("load_best_model_at_end", r.stderr)
        self.assertIn("eval_strategy", r.stderr)

    def test_save_steps_must_be_multiple_of_eval_steps(self):
        """When load_best_model_at_end is on, save_steps must be a multiple
        of eval_steps. HF Trainer enforces this but fails late and
        obscurely; we surface it at argparse time."""
        r = _run(*self._base_args(),
                 "--eval-strategy", "steps",
                 "--eval-steps", "250",
                 "--save-steps", "300",  # 300 % 250 != 0
                 "--load-best-model-at-end")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("save_steps", r.stderr)
        self.assertIn("eval_steps", r.stderr)

    def test_save_steps_multiple_passes(self):
        """Counterpart: 500 % 250 == 0 must NOT trigger the guard. The run
        may still fail later (at torch import in a deps-less env, or during
        Trainer init), but the failure must not be ours."""
        r = _run(*self._base_args(),
                 "--eval-strategy", "steps",
                 "--eval-steps", "250",
                 "--save-steps", "500",
                 "--load-best-model-at-end",
                 "--validate-only")
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        # Our guard's error message is the only place that uses this phrase;
        # argparse's usage line in stderr mentions --save-steps regardless,
        # so check for the guard's unique string instead.
        self.assertNotIn("must be a multiple", r.stderr)


if __name__ == "__main__":
    unittest.main()
