from __future__ import annotations

import unittest
from pathlib import Path

from scripts.report_alignment_errors import (
    ErrorSpan,
    classify_error,
    error_spans,
    render_markdown,
    summarize_case,
)


class TestReportAlignmentErrors(unittest.TestCase):
    def test_classify_error(self):
        self.assertEqual(classify_error({"pred": None}), "missing_pred")
        self.assertEqual(classify_error({"pred": "__no_match__"}), "unresolved_pred")
        self.assertEqual(
            classify_error(
                {"pred": "__no_match__"},
                raw_pred_segment={"shabad_id": 7},
                gt_shabad_id=7,
            ),
            "outside_gt_line",
        )
        self.assertEqual(
            classify_error(
                {"pred": "__no_match__"},
                raw_pred_segment={"shabad_id": 8},
                gt_shabad_id=7,
            ),
            "wrong_shabad_line",
        )
        self.assertEqual(classify_error({"pred": 2, "type": "boundary_error"}), "boundary_wrong")
        self.assertEqual(classify_error({"pred": 2, "type": "error"}), "wrong_line")

    def test_error_spans_groups_contiguous_same_error(self):
        details = [
            {"t": 0, "gt": 1, "pred": 2, "correct": False, "type": "error"},
            {"t": 1, "gt": 1, "pred": 2, "correct": False, "type": "error"},
            {"t": 2, "gt": 1, "pred": 1, "correct": True, "type": "exact"},
            {"t": 3, "gt": 2, "pred": None, "correct": False, "type": "error"},
        ]
        self.assertEqual(
            error_spans(details),
            [
                ErrorSpan(0, 2, 1, 2, "wrong_line"),
                ErrorSpan(3, 4, 2, None, "missing_pred"),
            ],
        )

    def test_summarize_case_counts_error_kinds_by_frame(self):
        score = {
            "video_id": "case",
            "shabad_id": 7,
            "frame_accuracy": 50.0,
            "correct": 2,
            "total": 4,
            "n_pred_segments": 1,
            "n_gt_segments": 2,
            "details": [
                {"t": 0, "gt": 1, "pred": 1, "correct": True, "type": "exact"},
                {"t": 1, "gt": 1, "pred": 2, "correct": False, "type": "error"},
                {"t": 2, "gt": 2, "pred": None, "correct": False, "type": "error"},
                {"t": 3, "gt": 2, "pred": 2, "correct": True, "type": "exact"},
            ],
        }
        summary = summarize_case(score)
        self.assertEqual(summary["error_frames"], 2)
        self.assertEqual(summary["by_kind"], {"missing_pred": 1, "wrong_line": 1})

    def test_render_markdown_contains_summary_and_spans(self):
        summaries = [
            {
                "video_id": "case",
                "case_id": "case",
                "shabad_id": 7,
                "accuracy": 50.0,
                "correct": 2,
                "total": 4,
                "error_frames": 2,
                "n_pred_segments": 1,
                "n_gt_segments": 2,
                "by_kind": {"missing_pred": 1, "wrong_line": 1},
                "spans": [
                    ErrorSpan(1, 2, 1, 2, "wrong_line"),
                    ErrorSpan(2, 3, 2, None, "missing_pred"),
                ],
            }
        ]
        text = render_markdown(
            pred_dir=Path("pred"),
            gt_dir=Path("gt"),
            summaries=summaries,
            collar=1,
        )
        self.assertIn("# Alignment error report", text)
        self.assertIn("Overall frame accuracy: `50.0%`", text)
        self.assertIn("| `wrong_line` | 1 | 50.0% |", text)
        self.assertIn("| `case` | 1 | 2 | 1 | `wrong_line` | 1 | 2 |", text)


if __name__ == "__main__":
    unittest.main()
