"""Unit tests for OOS GT bootstrap helpers."""

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
    from scripts.bootstrap_oos_gt import (  # noqa: E402
        OosCase,
        draft_payload,
        load_cases,
        wav_duration_s,
    )


@unittest.skipUnless(_HAS_YAML, "PyYAML required")
class TestLoadCases(unittest.TestCase):
    def test_loads_cases_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.yaml"
            path.write_text(
                """
schema_version: 1
cases:
  - case_id: case_001
    shabad_id: 2333
    source_url: https://example.com/a
    source_video_id: abc123
    clip_start_s: 30
    clip_end_s: 210
    role: representative
"""
            )
            cases = load_cases(path)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].case_id, "case_001")
            self.assertEqual(cases[0].shabad_id, 2333)
            self.assertEqual(cases[0].expected_duration_s, 180.0)

    def test_rejects_duplicate_case_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.yaml"
            path.write_text(
                """
cases:
  - case_id: dup
    shabad_id: 1
    source_url: https://example.com/a
    source_video_id: a
    clip_start_s: 0
    clip_end_s: 10
  - case_id: dup
    shabad_id: 2
    source_url: https://example.com/b
    source_video_id: b
    clip_start_s: 0
    clip_end_s: 10
"""
            )
            with self.assertRaises(ValueError):
                load_cases(path)

    def test_rejects_bad_clip_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.yaml"
            path.write_text(
                """
cases:
  - case_id: case_001
    shabad_id: 1
    source_url: https://example.com/a
    source_video_id: a
    clip_start_s: 10
    clip_end_s: 10
"""
            )
            with self.assertRaises(ValueError):
                load_cases(path)


@unittest.skipUnless(_HAS_YAML, "PyYAML required")
class TestWavDuration(unittest.TestCase):
    def test_wav_duration_s(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\x00\x00" * 32000)
            self.assertAlmostEqual(wav_duration_s(path), 2.0)


@unittest.skipUnless(_HAS_YAML, "PyYAML required")
class TestDraftPayload(unittest.TestCase):
    def test_marks_payload_as_draft_not_ground_truth(self):
        case = OosCase(
            case_id="case_001",
            shabad_id=2333,
            source_url="https://example.com/a",
            source_video_id="abc123",
            clip_start_s=30.0,
            clip_end_s=210.0,
        )
        payload = draft_payload(case, 180.0, [{"start": 0, "end": 1, "line_idx": 0}])
        self.assertEqual(payload["video_id"], "case_001")
        self.assertEqual(payload["shabad_id"], 2333)
        self.assertEqual(payload["total_duration"], 180.0)
        self.assertEqual(payload["uem"], {"start": 0.0, "end": 180.0})
        self.assertIn("DRAFT", payload["curation_status"])
        self.assertEqual(payload["source_clip"], {"start_s": 30.0, "end_s": 210.0})
        self.assertEqual(len(payload["segments"]), 1)


if __name__ == "__main__":
    unittest.main()
