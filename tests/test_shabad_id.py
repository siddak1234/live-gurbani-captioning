from __future__ import annotations

import unittest

from src.asr import AsrChunk
from src.shabad_id import (
    fusion_score_map,
    identify_shabad,
    identify_shabad_fusion,
    identify_shabad_guarded_fusion,
    parse_fusion_spec,
)


def _line(text: str, idx: int = 1) -> dict:
    return {
        "line_idx": idx,
        "verse_id": f"v{idx}",
        "banidb_gurmukhi": text,
        "transliteration_english": text,
    }


class TestFusionSpec(unittest.TestCase):
    def test_parse_weighted_terms(self):
        self.assertEqual(
            parse_fusion_spec("tfidf_45+0.5*chunk_vote_90+2*topk3_45"),
            [(1.0, "tfidf", 45.0, 0.0), (0.5, "chunk_vote", 90.0, 0.0), (2.0, "topk:3", 45.0, 0.0)],
        )

    def test_parse_tail_term(self):
        self.assertEqual(
            parse_fusion_spec("2*tail_chunk_vote_30_90"),
            [(2.0, "chunk_vote", 30.0, 60.0)],
        )

    def test_rejects_unknown_feature(self):
        with self.assertRaises(ValueError):
            parse_fusion_spec("mystery_30")

    def test_rejects_empty_spec(self):
        with self.assertRaises(ValueError):
            parse_fusion_spec("")


class TestIdentifyShabadFusion(unittest.TestCase):
    def test_fusion_selects_best_candidate(self):
        chunks = [AsrChunk(0.0, 2.0, "alpha beta")]
        corpora = {
            1: [_line("alpha beta")],
            2: [_line("gamma delta")],
        }
        result = identify_shabad_fusion(chunks, corpora, spec="tfidf_30+0.5*chunk_vote_30")
        self.assertEqual(result.shabad_id, 1)
        self.assertGreater(result.score, result.runner_up_score)

    def test_identify_shabad_accepts_fusion_aggregate(self):
        chunks = [AsrChunk(0.0, 2.0, "alpha beta")]
        corpora = {
            1: [_line("wrong text")],
            2: [_line("alpha beta")],
        }
        result = identify_shabad(
            chunks,
            corpora,
            aggregate="fusion:tfidf_30+chunk_vote_30",
            lookback_seconds=30.0,
        )
        self.assertEqual(result.shabad_id, 2)

    def test_fusion_result_includes_score_map(self):
        chunks = [AsrChunk(0.0, 2.0, "alpha beta")]
        corpora = {
            1: [_line("alpha beta")],
            2: [_line("gamma delta")],
        }
        result = identify_shabad_fusion(chunks, corpora, spec="tfidf_30+0.5*chunk_vote_30")
        self.assertEqual(set(result.score_by_shabad or {}), {1, 2})
        self.assertAlmostEqual((result.score_by_shabad or {})[1], result.score)

    def test_fusion_score_map_can_shift_later(self):
        chunks = [
            AsrChunk(0.0, 2.0, "alpha beta"),
            AsrChunk(90.0, 92.0, "gamma delta"),
        ]
        corpora = {
            1: [_line("alpha beta")],
            2: [_line("gamma delta")],
        }
        prefix = fusion_score_map(chunks, corpora, spec="chunk_vote_30", start_t=0.0)
        late = fusion_score_map(chunks, corpora, spec="chunk_vote_30", start_t=90.0)
        self.assertGreater(prefix[1], prefix[2])
        self.assertGreater(late[2], late[1])

    def test_guarded_fusion_switches_when_prefix_loses_late_support(self):
        chunks = [
            AsrChunk(0.0, 2.0, "alpha beta"),
            AsrChunk(90.0, 92.0, "gamma delta"),
        ]
        corpora = {
            1: [_line("alpha beta")],
            2: [_line("gamma delta")],
        }
        result = identify_shabad_guarded_fusion(
            chunks,
            corpora,
            spec="chunk_vote_30|offset=90|low=0.15|min=0.5",
        )
        self.assertEqual(result.shabad_id, 2)
        self.assertEqual(result.runner_up_id, 1)

    def test_guarded_fusion_keeps_prefix_when_late_support_is_not_low(self):
        chunks = [
            AsrChunk(0.0, 2.0, "alpha beta"),
            AsrChunk(90.0, 92.0, "alpha beta"),
            AsrChunk(93.0, 95.0, "gamma delta"),
        ]
        corpora = {
            1: [_line("alpha beta")],
            2: [_line("gamma delta")],
        }
        result = identify_shabad_guarded_fusion(
            chunks,
            corpora,
            spec="chunk_vote_30|offset=90|low=0.15|min=0.5",
        )
        self.assertEqual(result.shabad_id, 1)

    def test_identify_shabad_accepts_guarded_fusion_aggregate(self):
        chunks = [
            AsrChunk(0.0, 2.0, "alpha beta"),
            AsrChunk(90.0, 92.0, "gamma delta"),
        ]
        corpora = {
            1: [_line("alpha beta")],
            2: [_line("gamma delta")],
        }
        result = identify_shabad(
            chunks,
            corpora,
            aggregate="guarded_fusion:chunk_vote_30|offset=90|low=0.15|min=0.5",
            lookback_seconds=180.0,
        )
        self.assertEqual(result.shabad_id, 2)


if __name__ == "__main__":
    unittest.main()
