"""Unit tests for machine-assisted OOS GT cross-check helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import yaml  # noqa: F401
    from rapidfuzz import fuzz  # noqa: F401
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

if _HAS_DEPS:
    from scripts.audit_oos_assisted_gt import (  # noqa: E402
        load_caption_chunks,
        render_report,
    )
    from scripts.bootstrap_oos_gt import OosCase  # noqa: E402


@unittest.skipUnless(_HAS_DEPS, "PyYAML + rapidfuzz required")
class TestAuditOosAssistedGt(unittest.TestCase):
    def test_load_caption_chunks_offsets_youtube_times_into_clip_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            captions_dir = Path(tmp)
            (captions_dir / "vid.pa-orig.json3").write_text(
                json.dumps(
                    {
                        "events": [
                            {"tStartMs": 55000, "dDurationMs": 4000,
                             "segs": [{"utf8": "ਕੋਈ ਬੋਲੈ"}]},
                            {"tStartMs": 800000, "dDurationMs": 1000,
                             "segs": [{"utf8": "outside"}]},
                        ]
                    },
                    ensure_ascii=False,
                )
            )
            case = OosCase(
                case_id="case_001",
                shabad_id=1,
                source_url="https://example.com",
                source_video_id="vid",
                clip_start_s=45.0,
                clip_end_s=225.0,
            )
            chunks = load_caption_chunks(captions_dir, case)
            self.assertEqual(len(chunks), 1)
            self.assertAlmostEqual(chunks[0].start, 10.0)
            self.assertAlmostEqual(chunks[0].end, 14.0)
            self.assertEqual(chunks[0].text, "ਕੋਈ ਬੋਲੈ")

    def test_render_report_flags_asr_line_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_dir = root / "test"
            corpus_dir = root / "corpus"
            asr_dir = root / "asr"
            caps_dir = root / "captions"
            for d in (gt_dir, corpus_dir, asr_dir, caps_dir):
                d.mkdir()
            (corpus_dir / "2333.json").write_text(
                json.dumps(
                    {
                        "shabad_id": 2333,
                        "lines": [
                            {"line_idx": 0, "banidb_gurmukhi": "ਸਿਰਲੇਖ"},
                            {"line_idx": 1, "banidb_gurmukhi": "ਪਹਿਲੀ ਲਾਈਨ"},
                            {"line_idx": 2, "banidb_gurmukhi": "ਦੂਜੀ ਲਾਈਨ"},
                        ],
                    },
                    ensure_ascii=False,
                )
            )
            (gt_dir / "case_001.json").write_text(
                json.dumps(
                    {
                        "video_id": "case_001",
                        "shabad_id": 2333,
                        "total_duration": 10.0,
                        "curation_status": "NEEDS_HUMAN_CORRECTION",
                        "segments": [
                            {
                                "start": 0.0,
                                "end": 5.0,
                                "line_idx": 1,
                                "verse_id": "A",
                                "banidb_gurmukhi": "ਪਹਿਲੀ ਲਾਈਨ",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            )
            (asr_dir / "case_001_16k__medium__pa.json").write_text(
                json.dumps([{"start": 0.0, "end": 5.0, "text": "ਦੂਜੀ ਲਾਈਨ"}], ensure_ascii=False)
            )
            case = OosCase(
                case_id="case_001",
                shabad_id=2333,
                source_url="https://example.com",
                source_video_id="vid",
                clip_start_s=0.0,
                clip_end_s=10.0,
            )
            report = render_report(
                [case],
                gt_dir=gt_dir,
                corpus_dir=corpus_dir,
                asr_cache=asr_dir,
                captions_dir=caps_dir,
            )
            self.assertIn("ASR better matches corpus line 2", report)
            self.assertIn("NEEDS_HUMAN_CORRECTION", report)

    def test_render_report_flags_audio_in_unlabeled_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_dir = root / "test"
            corpus_dir = root / "corpus"
            asr_dir = root / "asr"
            caps_dir = root / "captions"
            for d in (gt_dir, corpus_dir, asr_dir, caps_dir):
                d.mkdir()
            (corpus_dir / "2333.json").write_text(
                json.dumps(
                    {
                        "shabad_id": 2333,
                        "lines": [
                            {"line_idx": 0, "banidb_gurmukhi": "ਸਿਰਲੇਖ"},
                            {"line_idx": 1, "banidb_gurmukhi": "ਪਹਿਲੀ ਲਾਈਨ"},
                        ],
                    },
                    ensure_ascii=False,
                )
            )
            (gt_dir / "case_001.json").write_text(
                json.dumps(
                    {
                        "video_id": "case_001",
                        "shabad_id": 2333,
                        "total_duration": 30.0,
                        "curation_status": "NEEDS_HUMAN_CORRECTION",
                        "segments": [
                            {
                                "start": 20.0,
                                "end": 25.0,
                                "line_idx": 1,
                                "verse_id": "A",
                                "banidb_gurmukhi": "ਪਹਿਲੀ ਲਾਈਨ",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            )
            (asr_dir / "case_001_16k__medium__pa.json").write_text(
                json.dumps([{"start": 5.0, "end": 10.0, "text": "ਪਹਿਲੀ ਲਾਈਨ"}], ensure_ascii=False)
            )
            case = OosCase(
                case_id="case_001",
                shabad_id=2333,
                source_url="https://example.com",
                source_video_id="vid",
                clip_start_s=0.0,
                clip_end_s=30.0,
            )
            report = render_report(
                [case],
                gt_dir=gt_dir,
                corpus_dir=corpus_dir,
                asr_cache=asr_dir,
                captions_dir=caps_dir,
            )
            self.assertIn("Large unlabeled gap", report)
            self.assertIn("ASR hears likely line 1", report)


if __name__ == "__main__":
    unittest.main()
