"""Unit tests for scripts/pull_corrections.py pure logic.

Covers holdout loading + enforcement, canonical-text joining, manifest record
shape, and the data card — none of which need the network or the service_role
key. The actual pull/download is exercised once real corrections exist + the
key is provided (Phase 6b run).

Run:
    python -m unittest tests.test_pull_corrections -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.pull_corrections import (  # noqa: E402
    DATASETS_CONFIG,
    build_record,
    data_card,
    is_held_out,
    load_holdout_shabad_ids,
    shabad_text_from_lines,
)


class HoldoutTests(unittest.TestCase):

    def test_real_config_includes_benchmark_shabads(self):
        ids = load_holdout_shabad_ids(DATASETS_CONFIG)
        for sid in ("4377", "1821", "1341", "3712"):
            self.assertIn(sid, ids)

    def test_only_enforced_sources_count(self):
        yaml_text = (
            "sources:\n"
            "  a:\n"
            "    holdout:\n"
            "      enforce: true\n"
            "      shabad_ids: [\"100\", \"200\"]\n"
            "  b:\n"
            "    holdout:\n"
            "      enforce: false\n"
            "      shabad_ids: [\"999\"]\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            path = Path(f.name)
        ids = load_holdout_shabad_ids(path)
        self.assertEqual(ids, {"100", "200"})
        self.assertNotIn("999", ids)

    def test_is_held_out_on_ground_truth_or_predicted(self):
        holdout = {"4377", "1821"}
        self.assertTrue(is_held_out({"ground_truth_shabad_id": 4377}, holdout))
        self.assertTrue(is_held_out({"predicted_shabad_id": 1821}, holdout))
        self.assertFalse(is_held_out({"ground_truth_shabad_id": 5621,
                                      "predicted_shabad_id": 9999}, holdout))


class TextTests(unittest.TestCase):

    def test_joins_nonempty_gurmukhi_lines(self):
        lines = [
            {"banidb_gurmukhi": "ਪੋਥੀ ਪਰਮੇਸਰ ਕਾ ਥਾਨੁ"},
            {"banidb_gurmukhi": ""},
            {"banidb_gurmukhi": "  ਸਾਧਸੰਗਿ  "},
        ]
        self.assertEqual(shabad_text_from_lines(lines), "ਪੋਥੀ ਪਰਮੇਸਰ ਕਾ ਥਾਨੁ ਸਾਧਸੰਗਿ")

    def test_empty_lines_give_empty_text(self):
        self.assertEqual(shabad_text_from_lines([]), "")


class RecordTests(unittest.TestCase):

    def _row(self):
        return {
            "id": "abc",
            "ground_truth_shabad_id": 5621,
            "ground_truth_line_idx": None,
            "predicted_shabad_id": 4377,
            "kind": "hardNegPos",
            "device_id": "dev-1",
            "model_version": "surt-small-v3-kirtan@6bit",
            "created_at": "2026-05-25T18:00:00Z",
            "audio_start": 10.0,
            "audio_end": 40.0,
        }

    def test_record_is_shabad_granularity_with_label(self):
        rec = build_record(self._row(), "clips/abc.m4a", "ਪੋਥੀ …")
        self.assertEqual(rec["audio"], "clips/abc.m4a")
        self.assertEqual(rec["text"], "ਪੋਥੀ …")
        self.assertEqual(rec["text_granularity"], "shabad")
        self.assertEqual(rec["shabad_id"], 5621)             # corrected (ground-truth) shabad is the label
        self.assertEqual(rec["predicted_shabad_id"], 4377)
        self.assertEqual(rec["review_status"], "unreviewed")
        self.assertEqual(rec["source"], "correction")

    def test_data_card_counts_and_hours(self):
        rows = [build_record(self._row(), "clips/abc.m4a", "x")]
        card = data_card(rows, dropped_holdout=2, missing_text=0)
        self.assertIn("records: **1**", card)
        self.assertIn("dropped (benchmark holdout): 2", card)
        # 30s clip window → 0.008 h
        self.assertIn("0.008 h", card)
        self.assertIn("5621: 1", card)


if __name__ == "__main__":
    unittest.main()
