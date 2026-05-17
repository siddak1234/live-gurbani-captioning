"""Unit tests for machine-assisted OOS proposal generation."""

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
    from scripts.bootstrap_oos_gt import OosCase  # noqa: E402
    from scripts.propose_oos_assisted_gt import ASSISTED_STATUS, propose_case  # noqa: E402


@unittest.skipUnless(_HAS_DEPS, "PyYAML + rapidfuzz required")
class TestProposeOosAssistedGt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.input_dir = root / "test"
        self.corpus_dir = root / "corpus"
        self.asr_dir = root / "asr"
        self.caps_dir = root / "captions"
        for path in (self.input_dir, self.corpus_dir, self.asr_dir, self.caps_dir):
            path.mkdir()
        self.case = OosCase(
            case_id="case_001",
            shabad_id=2333,
            source_url="https://example.com",
            source_video_id="vid",
            clip_start_s=0.0,
            clip_end_s=30.0,
        )
        (self.corpus_dir / "2333.json").write_text(
            json.dumps(
                {
                    "shabad_id": 2333,
                    "lines": [
                        {"line_idx": 0, "verse_id": "T", "banidb_gurmukhi": "ਸਿਰਲੇਖ"},
                        {"line_idx": 1, "verse_id": "A", "banidb_gurmukhi": "ਪਹਿਲੀ ਲਾਈਨ"},
                        {"line_idx": 2, "verse_id": "B", "banidb_gurmukhi": "ਦੂਜੀ ਲਾਈਨ"},
                    ],
                },
                ensure_ascii=False,
            )
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _write_gt(self):
        (self.input_dir / "case_001.json").write_text(
            json.dumps(
                {
                    "video_id": "case_001",
                    "shabad_id": 2333,
                    "total_duration": 30.0,
                    "uem": {"start": 0.0, "end": 30.0},
                    "curation_status": "NEEDS_HUMAN_CORRECTION",
                    "segments": [
                        {
                            "start": 20.0,
                            "end": 25.0,
                            "line_idx": 1,
                            "shabad_id": 2333,
                            "verse_id": "A",
                            "banidb_gurmukhi": "ਪਹਿਲੀ ਲਾਈਨ",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )

    def test_inserts_high_confidence_line_from_unlabeled_gap(self):
        self._write_gt()
        (self.asr_dir / "case_001_16k__medium__pa.json").write_text(
            json.dumps([{"start": 5.0, "end": 10.0, "text": "ਪਹਿਲੀ ਲਾਈਨ"}], ensure_ascii=False)
        )
        proposal, notes = propose_case(
            self.case,
            input_dir=self.input_dir,
            corpus_dir=self.corpus_dir,
            asr_cache=self.asr_dir,
            captions_dir=self.caps_dir,
        )
        self.assertEqual(proposal["curation_status"], ASSISTED_STATUS)
        self.assertTrue(proposal["gold_review_required"])
        self.assertEqual(len(proposal["segments"]), 2)
        self.assertEqual(proposal["segments"][0]["line_idx"], 1)
        self.assertEqual(proposal["segments"][0]["machine_assisted"]["action"], "insert_from_gap_evidence")
        self.assertTrue(any("inserted line 1" in n for n in notes))

    def test_replaces_segment_line_when_evidence_strongly_prefers_other_line(self):
        self._write_gt()
        (self.asr_dir / "case_001_16k__medium__pa.json").write_text(
            json.dumps([{"start": 20.0, "end": 25.0, "text": "ਦੂਜੀ ਲਾਈਨ"}], ensure_ascii=False)
        )
        proposal, notes = propose_case(
            self.case,
            input_dir=self.input_dir,
            corpus_dir=self.corpus_dir,
            asr_cache=self.asr_dir,
            captions_dir=self.caps_dir,
        )
        self.assertEqual(proposal["segments"][-1]["line_idx"], 2)
        self.assertEqual(proposal["segments"][-1]["verse_id"], "B")
        self.assertEqual(proposal["segments"][-1]["machine_assisted"]["action"], "replace_line_from_evidence")
        self.assertTrue(any("replaced" in n for n in notes))


if __name__ == "__main__":
    unittest.main()
