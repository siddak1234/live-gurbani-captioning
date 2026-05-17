"""Unit tests for the automated silver manifest evaluator."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_silver_manifest import (  # noqa: E402
    ManifestRecord,
    load_manifest,
    score_prediction,
    select_records,
    summarize,
)


class TestEvalSilverManifest(unittest.TestCase):
    def test_load_manifest_filters_and_resolves_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clips = root / "clips"
            clips.mkdir()
            (clips / "keep.wav").write_bytes(b"RIFFfake")
            (clips / "low.wav").write_bytes(b"RIFFfake")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps([
                {
                    "audio": "clips/keep.wav",
                    "text": "ਵਾਹਿਗੁਰੂ ਜੀ",
                    "score": 0.95,
                    "video_id": "v1",
                    "shabad_id": "s1",
                    "duration_s": 4.0,
                },
                {
                    "audio": "clips/low.wav",
                    "text": "low score",
                    "score": 0.2,
                },
                {
                    "audio": "clips/missing.wav",
                    "text": "missing audio",
                    "score": 1.0,
                },
                {
                    "audio": "clips/keep.wav",
                    "text": "",
                    "score": 1.0,
                },
            ], ensure_ascii=False))

            rows = load_manifest(manifest, min_score=0.9)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].audio.name, "keep.wav")
        self.assertEqual(rows[0].video_id, "v1")
        self.assertEqual(rows[0].shabad_id, "s1")

    def test_load_manifest_limit_applies_after_filtering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.wav").write_bytes(b"RIFFfake")
            (root / "b.wav").write_bytes(b"RIFFfake")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps([
                {"audio": "a.wav", "text": "ਇਕ", "score": 0.95},
                {"audio": "b.wav", "text": "ਦੋ", "score": 0.95},
            ], ensure_ascii=False))

            rows = load_manifest(manifest, min_score=0.9, limit=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].text, "ਇਕ")

    def test_round_robin_video_sampling_spreads_limit_across_videos(self):
        records = [
            ManifestRecord(0, Path("/a0.wav"), "a0", 1.0, "a", "s", 1.0),
            ManifestRecord(1, Path("/a1.wav"), "a1", 1.0, "a", "s", 1.0),
            ManifestRecord(2, Path("/a2.wav"), "a2", 1.0, "a", "s", 1.0),
            ManifestRecord(3, Path("/b0.wav"), "b0", 1.0, "b", "s", 1.0),
            ManifestRecord(4, Path("/b1.wav"), "b1", 1.0, "b", "s", 1.0),
            ManifestRecord(5, Path("/c0.wav"), "c0", 1.0, "c", "s", 1.0),
        ]

        picked = select_records(records, limit=5, sample_strategy="round_robin_video")

        self.assertEqual([r.idx for r in picked], [0, 3, 5, 1, 4])

    def test_score_prediction_normalizes_and_scores(self):
        record = ManifestRecord(
            idx=3,
            audio=Path("/tmp/example.wav"),
            text="ਸੁਣਿ ਯਾਰ ਹਮਾਰੇ ਸਜਣ",
            score=0.99,
            video_id="vid",
            shabad_id="sid",
            duration_s=9.0,
        )

        row = score_prediction(record, "ਸੁਣ ਯਾਰ ਹਮਾਰੇ ਸਜਣ")

        self.assertEqual(row.idx, 3)
        self.assertGreater(row.ratio, 80)
        self.assertGreater(row.wratio, 80)
        self.assertFalse(row.exact_norm)

    def test_summarize(self):
        r1 = ManifestRecord(0, Path("/a.wav"), "ਵਾਹਿਗੁਰੂ", 1.0, "v1", "s1", 3.0)
        r2 = ManifestRecord(1, Path("/b.wav"), "ਸਤਿਨਾਮ", 1.0, "v2", "s2", 7.0)
        rows = [
            score_prediction(r1, "ਵਾਹਿਗੁਰੂ"),
            score_prediction(r2, "ਗਲਤ"),
        ]

        summary = summarize(rows)

        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["unique_videos"], 2)
        self.assertEqual(summary["unique_shabads"], 2)
        self.assertEqual(summary["total_duration_s"], 10.0)
        self.assertEqual(summary["exact_norm_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
