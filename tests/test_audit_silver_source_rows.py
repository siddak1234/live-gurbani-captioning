"""Tests for source metadata audit helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.audit_silver_source_rows import (  # noqa: E402
    _assessment,
    clip_id_from_audio,
    parse_shards,
)


class TestAuditSilverSourceRows(unittest.TestCase):
    def test_clip_id_from_audio(self):
        self.assertEqual(
            clip_id_from_audio("training_data/silver/clips/video_clip_00001.wav"),
            "video_clip_00001",
        )

    def test_parse_shards(self):
        self.assertEqual(parse_shards("10-12,15"), [10, 11, 12, 15])

    def test_assessment_prefers_raw_caption_signal(self):
        self.assertEqual(
            _assessment(
                raw_best=98,
                final_best=62,
                decision="review",
                margin=0.5,
                op_counts='{"fix": 1}',
            ),
            "silver-label-risk: prediction matches raw caption better than canonical final",
        )

    def test_assessment_low_margin_replacement(self):
        self.assertEqual(
            _assessment(
                raw_best=70,
                final_best=72,
                decision="replaced",
                margin=0.05,
                op_counts='{"fix": 1}',
            ),
            "silver-label-risk: low-margin canonical replacement",
        )

    def test_assessment_true_or_ambiguous_miss(self):
        self.assertEqual(
            _assessment(
                raw_best=75,
                final_best=77,
                decision="unchanged",
                margin=0.8,
                op_counts='{"fix": 0}',
            ),
            "true-or-ambiguous-asr-miss",
        )


if __name__ == "__main__":
    unittest.main()
