from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.score_lattice import (  # noqa: E402
    build_score_lattice,
    choose_stay_bias_path,
    line_at_midpoint,
    score_lattice_summary,
)


@dataclass
class Chunk:
    start: float
    end: float
    text: str


class TestScoreLattice(unittest.TestCase):
    def test_line_at_midpoint(self):
        gt = [
            {"start": 0.0, "end": 5.0, "line_idx": 1},
            {"start": 5.0, "end": 10.0, "line_idx": 2},
        ]
        self.assertEqual(line_at_midpoint(1.0, 3.0, gt), 1)
        self.assertEqual(line_at_midpoint(5.0, 7.0, gt), 2)
        self.assertIsNone(line_at_midpoint(10.0, 11.0, gt))

    def test_choose_stay_bias_path_reports_chunk_choices(self):
        chunks = [
            (0.0, 1.0, [0.0, 90.0, 10.0]),
            (1.0, 2.0, [0.0, 87.0, 89.0]),
            (2.0, 3.0, [0.0, 10.0, 91.0]),
        ]
        self.assertEqual(
            choose_stay_bias_path(chunks, stay_margin=5.0, score_threshold=0.0),
            [1, 1, 2],
        )

    def test_build_score_lattice_includes_gt_and_top_scores(self):
        lines = [
            {"line_idx": 0, "transliteration_english": "title"},
            {"line_idx": 1, "transliteration_english": "sat naam"},
            {"line_idx": 2, "transliteration_english": "waheguru"},
        ]
        gt = [
            {"start": 0.0, "end": 5.0, "line_idx": 1},
            {"start": 5.0, "end": 10.0, "line_idx": 2},
        ]
        rows = build_score_lattice(
            [Chunk(0.0, 4.0, "sat naam"), Chunk(5.0, 9.0, "waheguru")],
            lines,
            gt,
            ratio="WRatio",
            blend=None,
            top_k=2,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].gt_line_idx, 1)
        self.assertEqual(rows[0].best_line_idx, 1)
        self.assertEqual(rows[1].gt_line_idx, 2)
        self.assertEqual(rows[1].best_line_idx, 2)
        self.assertEqual(len(rows[0].top_scores), 2)

    def test_score_lattice_summary_counts_matches(self):
        lines = [
            {"line_idx": 0, "transliteration_english": "title"},
            {"line_idx": 1, "transliteration_english": "sat naam"},
        ]
        rows = build_score_lattice(
            [Chunk(0.0, 4.0, "sat naam"), Chunk(10.0, 12.0, "sat naam")],
            lines,
            [{"start": 0.0, "end": 5.0, "line_idx": 1}],
            ratio="WRatio",
            blend=None,
            top_k=1,
        )
        self.assertEqual(score_lattice_summary(rows), {
            "chunks": 2,
            "chunks_with_gt": 1,
            "best_matches_gt": 1,
            "stay_matches_gt": 1,
            "null_gt_chunks": 1,
        })


if __name__ == "__main__":
    unittest.main()
