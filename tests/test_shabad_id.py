from __future__ import annotations

import unittest

from src.asr import AsrChunk
from src.shabad_id import identify_shabad, identify_shabad_fusion, parse_fusion_spec


def _line(text: str, idx: int = 0) -> dict:
    return {
        "line_idx": idx,
        "verse_id": f"v{idx}",
        "banidb_gurmukhi": text,
        "transliteration_english": text,
    }


class TestFusionSpec(unittest.TestCase):
    def test_parse_weighted_terms(self):
        self.assertEqual(
            parse_fusion_spec("tfidf_60+0.5*chunk_vote_90+2*topk3_45"),
            [(1.0, "tfidf", 60.0), (0.5, "chunk_vote", 90.0), (2.0, "topk:3", 45.0)],
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


if __name__ == "__main__":
    unittest.main()
