"""Unit tests for the OOS review workspace preparer."""

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
    from scripts.bootstrap_oos_gt import load_cases  # noqa: E402
    from scripts.prepare_oos_review_workspace import (  # noqa: E402
        HUMAN_STATUS,
        WORKING_STATUS,
        prepare_all,
    )
    from scripts.validate_oos_gt import validate_all  # noqa: E402


def _write_cases(path: Path) -> None:
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
"""
    )


def _draft_payload() -> dict:
    return {
        "video_id": "case_001",
        "shabad_id": 2333,
        "total_duration": 180.0,
        "uem": {"start": 0.0, "end": 180.0},
        "curation_status": "DRAFT_FROM_ORACLE_ENGINE__HAND_CORRECT_BEFORE_COMMIT",
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "line_idx": 0,
                "verse_id": "ABC1",
                "banidb_gurmukhi": "ਮੇਰਾ ਬੈਦੁ ਗੁਰੂ ਗੋਵਿੰਦਾ ॥",
            }
        ],
    }


@unittest.skipUnless(_HAS_YAML, "PyYAML required")
class TestPrepareOosReviewWorkspace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.cases_path = root / "cases.yaml"
        self.draft_dir = root / "drafts"
        self.test_dir = root / "test"
        self.audio_dir = root / "audio"
        self.draft_dir.mkdir()
        self.test_dir.mkdir()
        self.audio_dir.mkdir()
        _write_cases(self.cases_path)
        self.cases = load_cases(self.cases_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_draft(self, payload: dict | None = None) -> Path:
        path = self.draft_dir / "case_001.json"
        path.write_text(json.dumps(payload or _draft_payload(), ensure_ascii=False))
        return path

    def test_writes_working_copy_without_marking_human_corrected(self):
        self._write_draft()
        results = prepare_all(self.cases, draft_dir=self.draft_dir, test_dir=self.test_dir)
        self.assertEqual([(r.case_id, r.status) for r in results], [("case_001", "written")])

        out = json.loads((self.test_dir / "case_001.json").read_text())
        self.assertEqual(out["curation_status"], WORKING_STATUS)
        self.assertNotEqual(out["curation_status"], HUMAN_STATUS)
        self.assertTrue(out["human_review_required"])
        self.assertEqual(out["draft_curation_status"], _draft_payload()["curation_status"])
        self.assertIn("draft_source", out)

    def test_validate_oos_gt_still_rejects_prepared_working_copy(self):
        self._write_draft()
        prepare_all(self.cases, draft_dir=self.draft_dir, test_dir=self.test_dir)
        results = validate_all(
            cases_path=self.cases_path,
            gt_dir=self.test_dir,
            audio_dir=self.audio_dir,
        )
        self.assertTrue(any(WORKING_STATUS in e or HUMAN_STATUS in e for e in results[0].errors))

    def test_skips_existing_file_without_force(self):
        self._write_draft()
        existing = self.test_dir / "case_001.json"
        existing.write_text('{"keep": true}')
        results = prepare_all(self.cases, draft_dir=self.draft_dir, test_dir=self.test_dir)
        self.assertEqual(results[0].status, "skipped")
        self.assertEqual(json.loads(existing.read_text()), {"keep": True})

    def test_force_overwrites_existing_file(self):
        self._write_draft()
        existing = self.test_dir / "case_001.json"
        existing.write_text('{"keep": true}')
        results = prepare_all(
            self.cases,
            draft_dir=self.draft_dir,
            test_dir=self.test_dir,
            force=True,
        )
        self.assertEqual(results[0].status, "written")
        self.assertEqual(json.loads(existing.read_text())["curation_status"], WORKING_STATUS)

    def test_missing_draft_is_reported(self):
        results = prepare_all(self.cases, draft_dir=self.draft_dir, test_dir=self.test_dir)
        self.assertEqual(results[0].status, "missing")
        self.assertIn("missing", results[0].message)

    def test_rejects_draft_with_wrong_case_identity(self):
        bad = _draft_payload()
        bad["video_id"] = "wrong_case"
        self._write_draft(bad)
        with self.assertRaises(ValueError):
            prepare_all(self.cases, draft_dir=self.draft_dir, test_dir=self.test_dir)


if __name__ == "__main__":
    unittest.main()
