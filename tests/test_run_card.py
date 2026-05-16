"""Unit tests for src/run_card.py — the lineage-emission module.

Designed to run with either pytest or stdlib unittest:

    python -m unittest tests.test_run_card -v
    pytest tests/test_run_card.py -v

No heavy deps required (no torch, no transformers, no peft). The module
under test degrades gracefully when those aren't importable.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Repo root on sys.path so `from src.run_card import ...` works under both
# pytest (which loads conftest.py first) and plain `python -m unittest`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.run_card import (  # noqa: E402
    config_hash,
    data_hash,
    git_sha,
    peak_memory_gb,
    write_run_card,
)


class _Args:
    """Minimal argparse.Namespace stand-in for tests."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


class TestConfigHash(unittest.TestCase):
    def test_order_independent(self):
        """Same key-value set in different attribute order hashes the same."""
        a = _Args(lr=1e-5, seed=42, epochs=3)
        b = _Args(seed=42, epochs=3, lr=1e-5)
        self.assertEqual(config_hash(a), config_hash(b))

    def test_value_sensitive(self):
        """Changing one value changes the hash."""
        a = _Args(lr=1e-5, seed=42)
        b = _Args(lr=3e-5, seed=42)
        self.assertNotEqual(config_hash(a), config_hash(b))

    def test_path_values_canonicalized(self):
        """Path attrs must hash by their string form so re-runs aren't
        falsely flagged as different config."""
        a = _Args(output_dir=Path("/tmp/x"))
        b = _Args(output_dir="/tmp/x")
        self.assertEqual(config_hash(a), config_hash(b))

    def test_ignores_private_attrs(self):
        """Attrs starting with _ shouldn't affect the hash (internal state)."""
        a = _Args(lr=1e-5)
        b = _Args(lr=1e-5)
        b._internal_cache = {"junk": True}
        self.assertEqual(config_hash(a), config_hash(b))


class TestDataHash(unittest.TestCase):
    def test_order_independent(self):
        recs1 = [{"audio": "a.wav", "text": "x"}, {"audio": "b.wav", "text": "y"}]
        recs2 = list(reversed(recs1))
        self.assertEqual(data_hash(recs1), data_hash(recs2))

    def test_content_sensitive(self):
        recs1 = [{"audio": "a.wav", "text": "x"}]
        recs2 = [{"audio": "a.wav", "text": "y"}]
        self.assertNotEqual(data_hash(recs1), data_hash(recs2))

    def test_none_for_empty_or_none(self):
        """Explicit None signals 'no manifest' (vs an empty list)."""
        self.assertIsNone(data_hash(None))
        self.assertIsNone(data_hash([]))

    def test_only_audio_and_text_used(self):
        """Extra fields shouldn't perturb the hash — keep it stable as the
        manifest schema gains optional columns over time."""
        recs1 = [{"audio": "a.wav", "text": "x"}]
        recs2 = [{"audio": "a.wav", "text": "x", "duration_s": 12.3, "score": 0.9}]
        self.assertEqual(data_hash(recs1), data_hash(recs2))


class TestGitSha(unittest.TestCase):
    def test_returns_sha_in_repo(self):
        """When run inside the repo (the normal case), git_sha is a 40-char hex."""
        sha = git_sha()
        # Outside a git tree the return is the sentinel "unknown" — accept
        # that too so the test passes for downstream consumers who run from
        # a tarball.
        if sha == "unknown":
            self.skipTest("not in a git work tree")
        self.assertEqual(len(sha), 40)
        self.assertTrue(all(c in "0123456789abcdef" for c in sha))


class TestPeakMemory(unittest.TestCase):
    def test_never_raises(self):
        """peak_memory_gb is best-effort and must not raise even when no
        memory backend is importable."""
        pm = peak_memory_gb()
        self.assertIn(pm.source, ("cuda", "mps_driver", "psutil_rss", "unavailable"))
        # gb is None if unavailable, otherwise a non-negative float
        if pm.gb is not None:
            self.assertGreaterEqual(pm.gb, 0.0)


class TestWriteRunCard(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _load(self, dir_):
        return json.loads((Path(dir_) / "run_card.json").read_text())

    def test_writes_required_fields(self):
        args = _Args(seed=42, lr=1e-5, output_dir=str(self.tmpdir))
        recs = [{"audio": "a.wav", "text": "x"}]
        path = write_run_card(self.tmpdir, args=args, train_records=recs,
                              eval_records=None, wall_clock_s=12.0,
                              status="completed", device="cpu")
        self.assertTrue(path.exists())
        card = self._load(self.tmpdir)
        # The contract: these fields MUST exist on every run card so
        # downstream tooling (model registry, regression dashboards) can
        # rely on them without defensive `.get(..., None)` everywhere.
        for field in ("git_sha", "config_hash", "data_hash", "seed",
                      "hostname", "wall_clock_s", "peak_mem_gb",
                      "peak_mem_source", "status", "device",
                      "train_n_clips", "eval_n_clips",
                      "final_train_loss", "final_eval_loss", "args"):
            self.assertIn(field, card, f"missing field: {field}")

    def test_status_crashed_persists(self):
        """A crashed run still leaves a card with status='crashed' — the
        whole point of the try/finally wiring in finetune_path_b.py."""
        args = _Args(seed=42, output_dir=str(self.tmpdir))
        write_run_card(self.tmpdir, args=args, train_records=None,
                       eval_records=None, status="crashed")
        card = self._load(self.tmpdir)
        self.assertEqual(card["status"], "crashed")

    def test_status_interrupted_persists(self):
        args = _Args(seed=42, output_dir=str(self.tmpdir))
        write_run_card(self.tmpdir, args=args, train_records=None,
                       eval_records=None, status="interrupted")
        card = self._load(self.tmpdir)
        self.assertEqual(card["status"], "interrupted")

    def test_creates_output_dir(self):
        """write_run_card creates the output_dir if it doesn't exist —
        important because a crash during model loading means the dir is
        never created by Trainer."""
        out = self.tmpdir / "newly_created"
        args = _Args(seed=42, output_dir=str(out))
        path = write_run_card(out, args=args, train_records=None,
                              eval_records=None, status="crashed")
        self.assertTrue(out.exists())
        self.assertTrue(path.exists())

    def test_overwrites_on_rerun(self):
        """Re-running the same training command should overwrite the card,
        not append. Lineage reflects the latest run."""
        args1 = _Args(seed=42, lr=1e-5, output_dir=str(self.tmpdir))
        args2 = _Args(seed=42, lr=3e-5, output_dir=str(self.tmpdir))
        write_run_card(self.tmpdir, args=args1, train_records=None,
                       eval_records=None, status="completed")
        write_run_card(self.tmpdir, args=args2, train_records=None,
                       eval_records=None, status="completed")
        card = self._load(self.tmpdir)
        self.assertEqual(card["args"]["lr"], 3e-5)

    def test_extra_fields_accepted(self):
        """Optional extra dict lets later phases add per-PR fields without
        modifying the core writer signature."""
        args = _Args(seed=42, output_dir=str(self.tmpdir))
        write_run_card(self.tmpdir, args=args, train_records=None,
                       eval_records=None, status="completed",
                       extra={"phase": "smoke", "operator": "ci"})
        card = self._load(self.tmpdir)
        self.assertEqual(card["phase"], "smoke")
        self.assertEqual(card["operator"], "ci")


if __name__ == "__main__":
    unittest.main()
