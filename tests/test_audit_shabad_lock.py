from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.audit_shabad_lock import (  # noqa: E402
    load_cached_chunks,
    load_cases,
    parse_csv,
    parse_lookbacks,
    render_markdown,
)


class TestAuditShabadLockHelpers(unittest.TestCase):
    def test_parse_csv_trims_empty_values(self):
        self.assertEqual(parse_csv("chunk_vote, tfidf,, topk:3 "), ["chunk_vote", "tfidf", "topk:3"])

    def test_parse_lookbacks(self):
        self.assertEqual(parse_lookbacks("30,45, 60.5"), [30.0, 45.0, 60.5])

    def test_load_cached_chunks_uses_cache_naming_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            path = cache / "case_x_16k__medium_word__pa.json"
            path.write_text(json.dumps([
                {"start": 0, "end": 1.5, "text": "hello"},
                {"start": 1.5, "end": 3, "text": "world"},
            ]))
            chunks = load_cached_chunks(cache, "case_x", "medium_word")
        self.assertIsNotNone(chunks)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "hello")
        self.assertEqual(chunks[1].end, 3.0)

    def test_load_cached_chunks_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_cached_chunks(Path(tmp), "missing", "medium_word"))

    def test_load_cases_preserves_uem_start_for_cold_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            gt_dir = Path(tmp)
            (gt_dir / "case_cold.json").write_text(json.dumps({
                "video_id": "video_x",
                "shabad_id": 3712,
                "uem": {"start": 96.7, "end": 292.9},
                "segments": [],
            }))
            cases = load_cases(gt_dir)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "case_cold")
        self.assertEqual(cases[0].uem_start, 96.7)

    def test_render_markdown_contains_summary_and_details(self):
        rows = {
            (30.0, "chunk_vote"): [],
            (45.0, "chunk_vote"): [],
        }
        for key in rows:
            rows[key] = [
                # Minimal duck-typed row object is clearer than constructing
                # full audit output through rapidfuzz in this helper test.
                type("Row", (), {
                    "case_id": "case_001",
                    "gt_shabad_id": 2333,
                    "predicted_shabad_id": 2333,
                    "score": 42.0,
                    "runner_up_id": 1821,
                    "runner_up_score": 12.0,
                    "mode": "chunk_vote",
                    "ok": True,
                    "missing_cache": False,
                })(),
            ]
        text = render_markdown(
            gt_dir=Path("gt"),
            corpus_dir=Path("corpus_cache"),
            asr_cache_dir=Path("asr_cache"),
            asr_tag="medium_word",
            lookbacks=[30.0, 45.0],
            aggregates=["chunk_vote"],
            results=rows,
        )
        self.assertIn("# Shabad-lock audit", text)
        self.assertIn("| 30s | `chunk_vote` | 1/1 | 0 |", text)
        self.assertIn("| case_001 | 2333 | 2333 |", text)


if __name__ == "__main__":
    unittest.main()
