from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tune_shabad_lock_policy import (  # noqa: E402
    DatasetScore,
    DatasetSpec,
    Decision,
    Policy,
    PolicyResult,
    make_policy_grid,
    parse_score_grid,
    parse_window_grid,
    rank_results,
    render_markdown,
)


class TestTuneShabadLockPolicyHelpers(unittest.TestCase):
    def test_parse_window_grid(self):
        self.assertEqual(
            parse_window_grid("30;30,45; 30,45,60 ;"),
            [(30.0,), (30.0, 45.0), (30.0, 45.0, 60.0)],
        )

    def test_parse_score_grid(self):
        self.assertEqual(parse_score_grid("0, 1,25,,50"), [0.0, 1.0, 25.0, 50.0])

    def test_make_policy_grid_cross_product(self):
        policies = make_policy_grid(
            aggregates=["chunk_vote", "tfidf"],
            window_sets=[(30.0,), (30.0, 45.0)],
            min_scores=[0.0, 25.0],
        )
        self.assertEqual(len(policies), 8)
        self.assertEqual(policies[0].label, "chunk_vote@30s|min=0")
        self.assertEqual(policies[-1].label, "tfidf@30,45s|min=25")

    def test_rank_results_uses_macro_then_regression_guardrails(self):
        weak_paired = PolicyResult(
            policy=Policy("tfidf", (45.0,), 0.0),
            by_dataset={
                "paired": DatasetScore(correct=6, total=12, missing_cache=0),
                "assisted_oos": DatasetScore(correct=5, total=5, missing_cache=0),
            },
            decisions=[],
        )
        balanced = PolicyResult(
            policy=Policy("chunk_vote", (30.0, 45.0), 50.0),
            by_dataset={
                "paired": DatasetScore(correct=9, total=12, missing_cache=0),
                "assisted_oos": DatasetScore(correct=4, total=5, missing_cache=0),
            },
            decisions=[],
        )
        ranked = rank_results([weak_paired, balanced])
        # balanced macro = 77.5%, weak_paired macro = 75.0%
        self.assertIs(ranked[0], balanced)

    def test_render_markdown_contains_silver_warning_and_best_policy(self):
        policy = Policy("chunk_vote", (30.0, 45.0), 50.0)
        result = PolicyResult(
            policy=policy,
            by_dataset={
                "paired": DatasetScore(correct=9, total=12, missing_cache=0),
                "assisted_oos": DatasetScore(correct=4, total=5, missing_cache=0),
            },
            decisions=[
                Decision(
                    dataset="paired",
                    case_id="case_a",
                    gt_shabad_id=4377,
                    predicted_shabad_id=4377,
                    score=171.0,
                    runner_up_id=1341,
                    runner_up_score=0.0,
                    selected_window=45.0,
                    mode="chunk_vote",
                    ok=True,
                )
            ],
        )
        text = render_markdown(
            datasets=[
                DatasetSpec("paired", Path("paired_gt"), "paired benchmark"),
                DatasetSpec("assisted_oos", Path("oos_gt"), "silver labels"),
            ],
            corpus_dir=Path("corpus_cache"),
            asr_cache_dir=Path("asr_cache"),
            asr_tag="medium_word",
            ranked=[result],
            top_n=1,
        )
        self.assertIn("# Phase 2.12 silver lock-policy tuning", text)
        self.assertIn("Diagnostic only", text)
        self.assertIn("## Guardrail views", text)
        self.assertIn("chunk_vote@30,45s|min=50", text)
        self.assertIn("| paired | case_a | 4377 | 4377 | 45s |", text)


if __name__ == "__main__":
    unittest.main()
