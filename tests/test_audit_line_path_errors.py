from __future__ import annotations

import unittest

from scripts.audit_line_path_errors import (
    ErrorSpan,
    relation_for_span,
    render_markdown,
    span_record,
)


class TestAuditLinePathErrors(unittest.TestCase):
    def test_relation_for_span_classifies_core_path_errors(self):
        gt_lines = {1, 2, 6}
        base = {"gt_shabad_id": 7, "gt_lines": gt_lines}

        self.assertEqual(
            relation_for_span(
                ErrorSpan(0, 1, 1, 2, "wrong_line"),
                raw_pred_segment={"shabad_id": 7, "line_idx": 2},
                **base,
            ),
            "adjacent_future",
        )
        self.assertEqual(
            relation_for_span(
                ErrorSpan(0, 1, 2, 1, "wrong_line"),
                raw_pred_segment={"shabad_id": 7, "line_idx": 1},
                **base,
            ),
            "adjacent_backtrack",
        )
        self.assertEqual(
            relation_for_span(
                ErrorSpan(0, 1, 1, "__no_match__", "outside_gt_line"),
                raw_pred_segment={"shabad_id": 7, "line_idx": 4},
                **base,
            ),
            "outside_gt_line_set",
        )
        self.assertEqual(
            relation_for_span(
                ErrorSpan(0, 1, 1, "__no_match__", "wrong_shabad_line"),
                raw_pred_segment={"shabad_id": 8, "line_idx": 1},
                **base,
            ),
            "wrong_shabad",
        )
        self.assertEqual(
            relation_for_span(
                ErrorSpan(0, 1, None, 2, "wrong_line"),
                raw_pred_segment={"shabad_id": 7, "line_idx": 2},
                **base,
            ),
            "predicted_during_unlabeled_gt",
        )

    def test_span_record_enriches_with_corpus_text_and_delta(self):
        corpora = {
            7: {
                1: {"banidb_gurmukhi": "line one"},
                3: {"banidb_gurmukhi": "line three"},
            }
        }
        gt = {"shabad_id": 7, "segments": [{"line_idx": 1}]}
        record = span_record(
            case_id="case",
            span=ErrorSpan(10, 15, 1, 3, "wrong_line"),
            raw_pred_segment={"shabad_id": 7, "line_idx": 3},
            gt=gt,
            corpora=corpora,
        )
        self.assertEqual(record["line_delta"], 2)
        self.assertEqual(record["relation"], "outside_gt_line_set")
        self.assertEqual(record["gt_text"], "line one")
        self.assertEqual(record["pred_text"], "line three")

    def test_render_markdown_contains_path_sections(self):
        text = render_markdown(
            pred_dir="pred",
            gt_dir="gt",
            corpus_dir="corpus",
            collar=1,
            cases=[
                {
                    "case_id": "case",
                    "accuracy": 50.0,
                    "correct": 5,
                    "total": 10,
                    "error_frames": 5,
                    "n_pred_segments": 1,
                    "n_gt_segments": 1,
                    "spans": [
                        {
                            "case_id": "case",
                            "start": 0,
                            "end": 5,
                            "duration_s": 5,
                            "error_kind": "wrong_line",
                            "relation": "future_jump",
                            "gt_line_idx": 1,
                            "pred_line_idx": 3,
                            "gt_text": "first line",
                            "pred_text": "third line",
                        }
                    ],
                }
            ],
        )
        self.assertIn("# Locked-shabad line-path audit", text)
        self.assertIn("| `future_jump` | 5 | 100.0% |", text)
        self.assertIn("1: first line", text)


if __name__ == "__main__":
    unittest.main()
