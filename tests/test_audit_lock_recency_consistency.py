from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.audit_lock_recency_consistency import (  # noqa: E402
    DatasetSpec,
    RecencyRow,
    feature_specs_for_policy,
    parse_fusion_policy,
    render_markdown,
    should_flag,
)


class TestLockRecencyConsistency(unittest.TestCase):
    def test_parse_fusion_policy(self):
        policy = parse_fusion_policy("tfidf_45 + 0.5*chunk_vote_90")
        self.assertEqual(
            policy.features,
            (("tfidf_45", 1.0), ("chunk_vote_90", 0.5)),
        )
        self.assertEqual(policy.label, "tfidf_45 + 0.5*chunk_vote_90")

    def test_feature_specs_for_policy_preserves_names_and_shifts_offset(self):
        policy = parse_fusion_policy("tfidf_45+0.5*chunk_vote_90+topk3_30")
        specs = feature_specs_for_policy(policy, offset=90.0)
        by_name = {spec.name: spec for spec in specs}
        self.assertEqual(by_name["tfidf_45"].aggregate, "tfidf")
        self.assertEqual(by_name["tfidf_45"].window, 45.0)
        self.assertEqual(by_name["tfidf_45"].offset, 90.0)
        self.assertEqual(by_name["chunk_vote_90"].aggregate, "chunk_vote")
        self.assertEqual(by_name["topk3_30"].aggregate, "topk:3")

    def test_should_flag_only_when_prefix_loses_late_support(self):
        row = RecencyRow(
            dataset="paired",
            case_id="z",
            gt_shabad_id=3712,
            prefix_pred=4892,
            prefix_score=1.5,
            prefix_runner_up=3712,
            validation_offset=90.0,
            validation_pred=3712,
            validation_score=0.5,
            prefix_validation_score=0.09,
            validation_gt_rank=1,
        )
        self.assertTrue(
            should_flag(row, low_support_threshold=0.15, min_validation_score=0.5)
        )

        same_late_winner = RecencyRow(
            dataset="paired",
            case_id="ok",
            gt_shabad_id=4377,
            prefix_pred=4377,
            prefix_score=1.5,
            prefix_runner_up=3712,
            validation_offset=90.0,
            validation_pred=4377,
            validation_score=0.9,
            prefix_validation_score=0.9,
            validation_gt_rank=1,
        )
        self.assertFalse(
            should_flag(same_late_winner, low_support_threshold=0.15, min_validation_score=0.5)
        )

    def test_render_markdown_includes_flagged_rows_and_decision(self):
        rows = [
            RecencyRow(
                dataset="paired",
                case_id="zOtIpxMT9hU",
                gt_shabad_id=3712,
                prefix_pred=4892,
                prefix_score=1.5,
                prefix_runner_up=3712,
                validation_offset=90.0,
                validation_pred=3712,
                validation_score=0.5,
                prefix_validation_score=0.09,
                validation_gt_rank=1,
            )
        ]
        text = render_markdown(
            datasets=[DatasetSpec("paired", Path("gt"), "paired")],
            corpus_dir=Path("corpus_cache"),
            asr_cache_dir=Path("asr_cache"),
            asr_tag="medium_word",
            policy=parse_fusion_policy("tfidf_45+0.5*chunk_vote_90"),
            rows=rows,
            validation_offset=90.0,
            low_support_threshold=0.15,
            min_validation_score=0.5,
        )
        self.assertIn("# Phase 3 lock recency-consistency audit", text)
        self.assertIn("| paired | zOtIpxMT9hU | 3712 | 4892 |", text)
        self.assertIn("Do not start full 300h / multi-seed training", text)


if __name__ == "__main__":
    unittest.main()
