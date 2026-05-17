from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tune_lock_evidence_fusion import (  # noqa: E402
    FusionPolicy,
    build_feature_table,
    make_feature_specs,
    make_policy_grid,
    rank_candidates,
    render_markdown,
)
from scripts.tune_shabad_lock_policy import DatasetScore, DatasetSpec  # noqa: E402
from scripts.tune_lock_evidence_fusion import CandidateDecision, FusionResult  # noqa: E402
from src.asr import AsrChunk  # noqa: E402


class TestLockEvidenceFusion(unittest.TestCase):
    def test_make_feature_specs(self):
        specs = make_feature_specs([30.0, 45.0])
        self.assertEqual(
            [spec.name for spec in specs],
            ["chunk_vote_30", "tfidf_30", "topk3_30",
             "chunk_vote_45", "tfidf_45", "topk3_45"],
        )

    def test_make_policy_grid_contains_sparse_fusions(self):
        specs = make_feature_specs([30.0])
        policies = make_policy_grid(specs, max_features=2)
        labels = {policy.label for policy in policies}
        self.assertIn("chunk_vote_30", labels)
        self.assertIn("chunk_vote_30 + tfidf_30", labels)
        self.assertIn("2*chunk_vote_30 + tfidf_30", labels)

    def test_build_feature_table_normalizes_each_feature_per_case(self):
        chunks = [AsrChunk(0.0, 2.0, "alpha beta")]
        corpora = {
            1: [{"transliteration_english": "alpha beta"}],
            2: [{"transliteration_english": "gamma"}],
        }
        table = build_feature_table(chunks, corpora, make_feature_specs([30.0]))
        self.assertEqual(set(table), {1, 2})
        self.assertAlmostEqual(table[1]["chunk_vote_30"], 1.0)
        self.assertGreater(table[1]["topk3_30"], table[2]["topk3_30"])

    def test_rank_candidates_applies_weighted_features(self):
        table = {
            1: {"a": 1.0, "b": 0.0},
            2: {"a": 0.0, "b": 1.0},
        }
        self.assertEqual(rank_candidates(table, FusionPolicy((("a", 1.0),)))[0][0], 1)
        self.assertEqual(rank_candidates(table, FusionPolicy((("b", 1.0),)))[0][0], 2)
        self.assertEqual(rank_candidates(table, FusionPolicy((("a", 1.0), ("b", 2.0))))[0][0], 2)

    def test_render_markdown_reports_best_policy(self):
        result = FusionResult(
            policy=FusionPolicy((("chunk_vote_45", 1.0), ("tfidf_45", 0.5))),
            by_dataset={
                "paired": DatasetScore(9, 12, 0),
                "assisted_oos": DatasetScore(4, 5, 0),
            },
            decisions=[
                CandidateDecision(
                    dataset="paired",
                    case_id="case_a",
                    gt_shabad_id=4377,
                    predicted_shabad_id=4377,
                    score=1.5,
                    runner_up_id=1341,
                    runner_up_score=0.2,
                    gt_rank=1,
                    ok=True,
                )
            ],
        )
        text = render_markdown(
            datasets=[
                DatasetSpec("paired", Path("paired_gt"), "paired"),
                DatasetSpec("assisted_oos", Path("oos_gt"), "silver"),
            ],
            corpus_dir=Path("corpus_cache"),
            asr_cache_dir=Path("asr_cache"),
            asr_tag="medium_word",
            feature_specs=make_feature_specs([45.0]),
            ranked=[result],
            top_n=1,
        )
        self.assertIn("# Phase 2.13 lock-evidence fusion", text)
        self.assertIn("chunk_vote_45 + 0.5*tfidf_45", text)
        self.assertIn("| paired | case_a | 4377 | 4377 |", text)


if __name__ == "__main__":
    unittest.main()
