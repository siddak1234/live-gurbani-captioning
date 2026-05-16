"""Tests for Phase 2.5 data-pull diversity controls."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.pull_dataset import (  # noqa: E402
    _check_diversity_floors,
    _diversity_counts,
    _parse_shards,
    _pull_target_met,
)


def _rec(video: str, shabad: str) -> dict:
    return {"video_id": video, "shabad_id": shabad, "duration_s": 5.0}


class TestParseShards(unittest.TestCase):
    def test_single_shard(self):
        self.assertEqual(_parse_shards("0"), (0,))

    def test_comma_list(self):
        self.assertEqual(_parse_shards("0,2,5"), (0, 2, 5))

    def test_range(self):
        self.assertEqual(_parse_shards("0-3"), (0, 1, 2, 3))

    def test_mixed_dedupes_preserving_order(self):
        self.assertEqual(_parse_shards("0-2,2,4,1"), (0, 1, 2, 4))

    def test_rejects_negative(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_shards("-1")

    def test_rejects_reversed_range(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_shards("3-1")

    def test_rejects_empty_spec(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_shards(" , ")


class TestDiversityFloors(unittest.TestCase):
    def test_counts_unique_videos_and_shabads(self):
        manifest = [
            _rec("v1", "s1"),
            _rec("v1", "s2"),
            _rec("v2", "s2"),
            _rec("v2", "s3"),
        ]
        self.assertEqual(_diversity_counts(manifest), {
            "unique_videos": 2,
            "unique_shabads": 3,
        })

    def test_passes_when_floors_met(self):
        manifest = [_rec("v1", "s1"), _rec("v2", "s2")]
        counts, failures = _check_diversity_floors(
            manifest,
            min_unique_videos=2,
            min_unique_shabads=2,
        )
        self.assertEqual(counts["unique_videos"], 2)
        self.assertEqual(counts["unique_shabads"], 2)
        self.assertEqual(failures, [])

    def test_reports_both_failures(self):
        manifest = [_rec("v1", "s1"), _rec("v1", "s1")]
        _, failures = _check_diversity_floors(
            manifest,
            min_unique_videos=2,
            min_unique_shabads=2,
        )
        self.assertEqual(len(failures), 2)
        self.assertIn("unique videos 1 < required 2", failures[0])
        self.assertIn("unique shabads 1 < required 2", failures[1])

    def test_zero_floors_are_noop(self):
        _, failures = _check_diversity_floors([], min_unique_videos=0, min_unique_shabads=0)
        self.assertEqual(failures, [])


class TestPullTargetMet(unittest.TestCase):
    def test_requires_sample_count_first(self):
        manifest = [_rec("v1", "s1")]
        self.assertFalse(_pull_target_met(
            manifest,
            num_samples=2,
            min_unique_videos=1,
            min_unique_shabads=1,
        ))

    def test_no_floors_means_sample_count_is_enough(self):
        manifest = [_rec("v1", "s1"), _rec("v1", "s1")]
        self.assertTrue(_pull_target_met(
            manifest,
            num_samples=2,
            min_unique_videos=0,
            min_unique_shabads=0,
        ))

    def test_active_floors_can_extend_beyond_num_samples(self):
        # Reached num_samples=2, but only one video/shabad. Keep scanning.
        manifest = [_rec("v1", "s1"), _rec("v1", "s1")]
        self.assertFalse(_pull_target_met(
            manifest,
            num_samples=2,
            min_unique_videos=2,
            min_unique_shabads=2,
        ))

        # Extra clips satisfy the floors; now the pull can stop.
        manifest.append(_rec("v2", "s2"))
        self.assertTrue(_pull_target_met(
            manifest,
            num_samples=2,
            min_unique_videos=2,
            min_unique_shabads=2,
        ))


if __name__ == "__main__":
    unittest.main()
