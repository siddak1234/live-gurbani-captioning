"""Tests for Phase 2.10 silver weak-slice report helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.report_silver_weak_slices import (  # noqa: E402
    duration_bucket,
    load_paired,
    render_report,
)


def _row(idx: int, *, video: str, shabad: str, target: str, pred: str,
         wratio: float, duration: float = 7.0, exact: bool = False) -> dict:
    return {
        "idx": idx,
        "audio": f"clip_{idx}.wav",
        "video_id": video,
        "shabad_id": shabad,
        "duration_s": duration,
        "label_score": 1.0,
        "target": target,
        "pred": pred,
        "norm_target": target,
        "norm_pred": pred,
        "ratio": wratio,
        "token_sort_ratio": wratio,
        "wratio": wratio,
        "exact_norm": exact,
    }


class TestReportSilverWeakSlices(unittest.TestCase):
    def test_duration_bucket(self):
        self.assertEqual(duration_bucket(4.9), "<5s")
        self.assertEqual(duration_bucket(5.0), "5-10s")
        self.assertEqual(duration_bucket(10.0), "10-15s")
        self.assertEqual(duration_bucket(15.0), "15-20s")
        self.assertEqual(duration_bucket(20.0), ">=20s")

    def test_load_paired_and_render_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.json"
            v5b = root / "v5b.json"
            base.write_text(json.dumps({
                "rows": [
                    _row(1, video="v1", shabad="s1", target="target a", pred="bad", wratio=55),
                    _row(2, video="v2", shabad="s2", target="target b", pred="good", wratio=98, exact=True),
                ],
            }))
            v5b.write_text(json.dumps({
                "rows": [
                    _row(1, video="v1", shabad="s1", target="target a", pred="better", wratio=75),
                    _row(2, video="v2", shabad="s2", target="target b", pred="worse", wratio=90),
                ],
            }))

            rows = load_paired(base, v5b)
            report = render_report(rows, base_path=base, v5b_path=v5b)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].delta, 20)
        self.assertIn("Weak videos", report)
        self.assertIn("Largest v5b improvements", report)
        self.assertIn("v1", report)

    def test_load_paired_rejects_mismatched_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.json"
            v5b = root / "v5b.json"
            base.write_text(json.dumps({"rows": [
                _row(1, video="v1", shabad="s1", target="a", pred="a", wratio=100),
            ]}))
            v5b.write_text(json.dumps({"rows": [
                _row(2, video="v2", shabad="s2", target="b", pred="b", wratio=100),
            ]}))

            with self.assertRaises(ValueError):
                load_paired(base, v5b)


if __name__ == "__main__":
    unittest.main()
