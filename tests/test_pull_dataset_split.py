"""Tests for the shabad-level train/val/test splitter in pull_dataset.py.

The split function is the keystone of Phase 1's hygiene story:
**no shabad appears in more than one split**. Every other Phase 1+ metric
depends on this — if a shabad straddles train and val, validation loss
becomes a memorization indicator instead of a generalization signal.

These tests run against the pure helper ``_split_by_shabad`` (no I/O, no
HF). PyYAML is loaded transitively (the import path goes through
scripts/pull_dataset.py which has a load_dataset_config at module scope);
skipped gracefully on bare-clone envs.

Run:
    python -m unittest tests.test_pull_dataset_split -v
    make test
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import yaml  # noqa: F401
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

if _HAS_YAML:
    from scripts.pull_dataset import _parse_split_ratios, _split_by_shabad  # noqa: E402


def _make_manifest(shabad_clip_counts: dict[str, int], clip_seconds: float = 10.0) -> list[dict]:
    """Build a synthetic manifest: ``shabad_clip_counts={"sid_a": 3, "sid_b": 5}``
    produces 3 clips of sid_a and 5 of sid_b, each ``clip_seconds`` long."""
    records: list[dict] = []
    for sid, n in shabad_clip_counts.items():
        for i in range(n):
            records.append({
                "audio": f"clips/{sid}_{i:03d}.wav",
                "text": f"text-{sid}-{i}",
                "source": "kirtan",
                "shabad_id": sid,
                "video_id": f"vid_{sid}",
                "score": 0.9,
                "duration_s": clip_seconds,
            })
    return records


@unittest.skipUnless(_HAS_YAML, "PyYAML not installed")
class TestSplitByShabad(unittest.TestCase):

    def test_zero_leakage(self):
        """The keystone invariant: no shabad_id appears in two splits."""
        manifest = _make_manifest({f"s{i:03d}": 10 for i in range(20)})  # 20 shabads × 10 clips
        splits = _split_by_shabad(manifest)
        train_sids = {r["shabad_id"] for r in splits["train"]}
        val_sids = {r["shabad_id"] for r in splits["val"]}
        test_sids = {r["shabad_id"] for r in splits["test"]}
        self.assertEqual(train_sids & val_sids, set(),
                         f"shabads leaked train↔val: {train_sids & val_sids}")
        self.assertEqual(train_sids & test_sids, set(),
                         f"shabads leaked train↔test: {train_sids & test_sids}")
        self.assertEqual(val_sids & test_sids, set(),
                         f"shabads leaked val↔test: {val_sids & test_sids}")

    def test_all_records_assigned(self):
        """Every input record lands in exactly one split (no clip dropped or duplicated)."""
        manifest = _make_manifest({f"s{i}": 5 for i in range(10)})
        splits = _split_by_shabad(manifest)
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        self.assertEqual(total, len(manifest))

    def test_deterministic(self):
        """Same seed → same split assignment (byte-exact)."""
        manifest = _make_manifest({f"s{i}": 5 for i in range(20)})
        a = _split_by_shabad(manifest, seed=42)
        b = _split_by_shabad(manifest, seed=42)
        for split in ("train", "val", "test"):
            self.assertEqual([r["shabad_id"] for r in a[split]],
                             [r["shabad_id"] for r in b[split]])

    def test_seed_changes_assignment(self):
        """Different seed → at least some assignments change. Probabilistic but
        with 20 shabads the chance of two seeds producing identical splits is
        astronomically small."""
        manifest = _make_manifest({f"s{i}": 5 for i in range(20)})
        a = _split_by_shabad(manifest, seed=42)
        b = _split_by_shabad(manifest, seed=99)
        a_train_sids = sorted({r["shabad_id"] for r in a["train"]})
        b_train_sids = sorted({r["shabad_id"] for r in b["train"]})
        self.assertNotEqual(a_train_sids, b_train_sids)

    def test_ratios_approximately_match(self):
        """Greedy allocation isn't exact, but with many equal-sized shabads the
        ratios should be within ±20% of target. With wildly uneven shabad
        sizes the tolerance has to widen — that's why we want the audit to
        warn loudly in real pulls."""
        manifest = _make_manifest({f"s{i:03d}": 10 for i in range(100)})  # 100 shabads, equal hours
        splits = _split_by_shabad(manifest, ratios=(0.8, 0.1, 0.1))
        total = len(manifest)
        for split, target_ratio in [("train", 0.8), ("val", 0.1), ("test", 0.1)]:
            actual_ratio = len(splits[split]) / total
            tolerance = 0.2 * target_ratio  # ±20% of target
            self.assertAlmostEqual(actual_ratio, target_ratio, delta=tolerance,
                                   msg=f"{split}: expected ~{target_ratio}, got {actual_ratio:.3f}")

    def test_custom_ratios(self):
        """70/15/15 split should produce roughly 70% in train."""
        manifest = _make_manifest({f"s{i:03d}": 10 for i in range(100)})
        splits = _split_by_shabad(manifest, ratios=(0.7, 0.15, 0.15))
        train_ratio = len(splits["train"]) / len(manifest)
        self.assertAlmostEqual(train_ratio, 0.7, delta=0.15)

    def test_empty_manifest(self):
        """Edge case: no input → empty splits, no exception."""
        splits = _split_by_shabad([])
        self.assertEqual(splits, {"train": [], "val": [], "test": []})

    def test_rejects_empty_shabad_id(self):
        """Records without shabad_id must raise — shabad-level splitting is
        incoherent on data with no shabad metadata."""
        manifest = _make_manifest({"a": 3})
        manifest.append({"audio": "x.wav", "text": "y", "shabad_id": "", "duration_s": 1.0})
        with self.assertRaises(ValueError):
            _split_by_shabad(manifest)

    def test_rejects_ratios_not_summing_to_one(self):
        """Defensive guard inside the pure function — covers Python callers
        that bypass the CLI's _parse_split_ratios."""
        manifest = _make_manifest({"a": 3})
        with self.assertRaises(ValueError):
            _split_by_shabad(manifest, ratios=(0.5, 0.3, 0.3))  # sums to 1.1

    def test_diversity_warning_emitted(self):
        """Val/test with < 3 unique shabads triggers stderr warning (not raise).
        Verified by capturing stderr — non-fatal but visible."""
        # 4 shabads → with 80/10/10, val and test will have ~0-1 shabad each
        manifest = _make_manifest({f"s{i}": 5 for i in range(4)})
        buf = io.StringIO()
        with redirect_stderr(buf):
            _split_by_shabad(manifest)
        stderr = buf.getvalue()
        # At least one of val/test should have < 3 → at least one warning
        self.assertIn("Warning:", stderr)

    def test_no_warning_when_diverse(self):
        """With many shabads, no diversity warning."""
        manifest = _make_manifest({f"s{i:03d}": 5 for i in range(100)})
        buf = io.StringIO()
        with redirect_stderr(buf):
            _split_by_shabad(manifest)
        self.assertNotIn("Warning:", buf.getvalue())

    def test_falls_back_to_count_when_durations_missing(self):
        """If every clip has duration_s=0 (metadata gap), the splitter falls
        back to equal-weight-by-clip-count so the split is still meaningful."""
        manifest = _make_manifest({f"s{i:03d}": 5 for i in range(20)}, clip_seconds=0.0)
        splits = _split_by_shabad(manifest)
        # Should still produce a usable 3-way split with no leakage
        train_sids = {r["shabad_id"] for r in splits["train"]}
        val_sids = {r["shabad_id"] for r in splits["val"]}
        test_sids = {r["shabad_id"] for r in splits["test"]}
        self.assertEqual(train_sids & val_sids, set())
        self.assertEqual(train_sids & test_sids, set())
        self.assertGreater(len(splits["train"]), 0)


@unittest.skipUnless(_HAS_YAML, "PyYAML not installed")
class TestParseSplitRatios(unittest.TestCase):

    def test_default_80_10_10(self):
        self.assertEqual(_parse_split_ratios("0.8,0.1,0.1"), (0.8, 0.1, 0.1))

    def test_tolerates_whitespace(self):
        self.assertEqual(_parse_split_ratios("0.7, 0.15 , 0.15"), (0.7, 0.15, 0.15))

    def test_rejects_wrong_count(self):
        import argparse
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_split_ratios("0.8,0.2")
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_split_ratios("0.5,0.2,0.2,0.1")

    def test_rejects_not_summing_to_one(self):
        import argparse
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_split_ratios("0.5,0.3,0.3")  # sums to 1.1
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_split_ratios("0.5,0.3,0.1")  # sums to 0.9


if __name__ == "__main__":
    unittest.main()
