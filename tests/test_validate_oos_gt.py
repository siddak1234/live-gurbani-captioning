"""Unit tests for strict OOS GT validation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import wave
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
    from scripts.validate_oos_gt import HUMAN_STATUS, validate_all  # noqa: E402


def _write_cases(path: Path) -> None:
    path.write_text(
        """
cases:
  - case_id: case_001
    shabad_id: 2333
    source_url: https://example.com/a
    source_video_id: abc123
    clip_start_s: 30
    clip_end_s: 210
"""
    )


def _write_wav(path: Path, duration_s: float = 2.0) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * int(duration_s * 16000))


def _valid_gt() -> dict:
    return {
        "video_id": "case_001",
        "shabad_id": 2333,
        "total_duration": 2.0,
        "uem": {"start": 0.0, "end": 2.0},
        "curation_status": HUMAN_STATUS,
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "line_idx": 0,
                "verse_id": "ABC1",
                "banidb_gurmukhi": "ਮੇਰਾ ਬੈਦੁ ਗੁਰੂ ਗੋਵਿੰਦਾ ॥",
            }
        ],
    }


@unittest.skipUnless(_HAS_YAML, "PyYAML required")
class TestValidateOosGt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.cases = root / "cases.yaml"
        self.gt_dir = root / "test"
        self.audio_dir = root / "audio"
        self.gt_dir.mkdir()
        self.audio_dir.mkdir()
        _write_cases(self.cases)
        _write_wav(self.audio_dir / "case_001_16k.wav")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self):
        return validate_all(cases_path=self.cases, gt_dir=self.gt_dir, audio_dir=self.audio_dir)

    def test_valid_human_corrected_gt_passes(self):
        (self.gt_dir / "case_001.json").write_text(json.dumps(_valid_gt(), ensure_ascii=False))
        results = self._run()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].errors, [])

    def test_missing_gt_file_fails(self):
        results = self._run()
        self.assertIn("missing GT file", results[0].errors[0])

    def test_draft_status_fails(self):
        gt = _valid_gt()
        gt["curation_status"] = "DRAFT_FROM_ORACLE_ENGINE__HAND_CORRECT_BEFORE_COMMIT"
        (self.gt_dir / "case_001.json").write_text(json.dumps(gt, ensure_ascii=False))
        errors = self._run()[0].errors
        self.assertTrue(any("draft" in e.lower() for e in errors))
        self.assertTrue(any(HUMAN_STATUS in e for e in errors))

    def test_duration_mismatch_fails(self):
        gt = _valid_gt()
        gt["total_duration"] = 10.0
        (self.gt_dir / "case_001.json").write_text(json.dumps(gt, ensure_ascii=False))
        errors = self._run()[0].errors
        self.assertTrue(any("differs from audio duration" in e for e in errors))

    def test_missing_segment_canonical_fields_fail(self):
        gt = _valid_gt()
        gt["segments"][0].pop("verse_id")
        gt["segments"][0]["banidb_gurmukhi"] = ""
        (self.gt_dir / "case_001.json").write_text(json.dumps(gt, ensure_ascii=False))
        errors = self._run()[0].errors
        self.assertTrue(any("verse_id" in e for e in errors))
        self.assertTrue(any("banidb_gurmukhi" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
