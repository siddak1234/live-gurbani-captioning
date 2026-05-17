"""Unit tests for the OOS review-pack renderer."""

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
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

if _HAS_YAML:
    from scripts.render_oos_review import render_index  # noqa: E402
    from scripts.bootstrap_oos_gt import OosCase  # noqa: E402


@unittest.skipUnless(_HAS_YAML, "PyYAML required")
class TestRenderOosReview(unittest.TestCase):
    def test_render_index_links_audio_and_marks_human_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir = root / "drafts"
            audio_dir = root / "audio"
            test_dir = root / "test"
            review_dir = root / "review"
            draft_dir.mkdir()
            audio_dir.mkdir()
            test_dir.mkdir()
            review_dir.mkdir()

            case = OosCase(
                case_id="case_001",
                shabad_id=2333,
                source_url="https://example.com/watch?v=x",
                source_video_id="x",
                clip_start_s=30.0,
                clip_end_s=210.0,
                role="representative",
                opening_line="ਮੇਰਾ ਬੈਦੁ ਗੁਰੂ ਗੋਵਿੰਦਾ ॥",
                rationale="clean representative case",
            )
            (draft_dir / "case_001.json").write_text(
                json.dumps(
                    {
                        "video_id": "case_001",
                        "shabad_id": 2333,
                        "total_duration": 180.0,
                        "curation_status": "DRAFT_FROM_ORACLE_ENGINE__HAND_CORRECT_BEFORE_COMMIT",
                        "segments": [
                            {
                                "start": 1.0,
                                "end": 2.0,
                                "line_idx": 0,
                                "verse_id": "ABC1",
                                "banidb_gurmukhi": "ਮੇਰਾ ਬੈਦੁ ਗੁਰੂ ਗੋਵਿੰਦਾ ॥",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            )
            out_path = review_dir / "index.html"
            page = render_index(
                [case],
                draft_dir=draft_dir,
                audio_dir=audio_dir,
                test_dir=test_dir,
                out_path=out_path,
            )
            self.assertIn("../audio/case_001_16k.wav", page)
            self.assertIn("HUMAN_CORRECTED_V1", page)
            self.assertIn("test/case_001.json", page)
            self.assertIn("playSegment('case_001',1.000,2.000)", page)
            self.assertIn("missing working GT file", page)

    def test_render_index_prefers_working_gt_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_dir = root / "drafts"
            audio_dir = root / "audio"
            test_dir = root / "test"
            review_dir = root / "review"
            draft_dir.mkdir()
            audio_dir.mkdir()
            test_dir.mkdir()
            review_dir.mkdir()

            case = OosCase(
                case_id="case_001",
                shabad_id=2333,
                source_url="https://example.com/watch?v=x",
                source_video_id="x",
                clip_start_s=30.0,
                clip_end_s=210.0,
                role="representative",
                opening_line="ਮੇਰਾ ਬੈਦੁ ਗੁਰੂ ਗੋਵਿੰਦਾ ॥",
            )
            common = {
                "video_id": "case_001",
                "shabad_id": 2333,
                "total_duration": 180.0,
                "uem": {"start": 0.0, "end": 180.0},
                "curation_status": "NEEDS_HUMAN_CORRECTION",
            }
            (draft_dir / "case_001.json").write_text(
                json.dumps(
                    {
                        **common,
                        "segments": [
                            {
                                "start": 1.0,
                                "end": 2.0,
                                "line_idx": 0,
                                "verse_id": "DRAFT1",
                                "banidb_gurmukhi": "ਡਰਾਫਟ ਲਾਈਨ",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            )
            (test_dir / "case_001.json").write_text(
                json.dumps(
                    {
                        **common,
                        "segments": [
                            {
                                "start": 3.0,
                                "end": 4.0,
                                "line_idx": 0,
                                "verse_id": "WORK1",
                                "banidb_gurmukhi": "ਵਰਕਿੰਗ ਲਾਈਨ",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            )
            page = render_index(
                [case],
                draft_dir=draft_dir,
                audio_dir=audio_dir,
                test_dir=test_dir,
                out_path=review_dir / "index.html",
            )
            self.assertIn("reviewing: working GT", page)
            self.assertIn("ਵਰਕਿੰਗ ਲਾਈਨ", page)
            self.assertNotIn("ਡਰਾਫਟ ਲਾਈਨ", page)
            self.assertIn("curation_status must be", page)


if __name__ == "__main__":
    unittest.main()
