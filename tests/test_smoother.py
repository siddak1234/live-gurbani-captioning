from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.smoother import (  # noqa: E402
    is_simran_dominant,
    smooth,
    smooth_with_loop_align,
    smooth_with_loop_align_confirmed,
    smooth_with_stay_bias,
    smooth_with_viterbi,
)


class TestSmoother(unittest.TestCase):
    def test_smooth_collapses_adjacent_same_line(self):
        segments = smooth([
            (0.0, 1.0, 1),
            (1.0, 2.0, 1),
            (2.0, 3.0, None),
            (3.0, 4.0, 2),
        ])
        self.assertEqual([(s.start, s.end, s.line_idx) for s in segments], [
            (0.0, 2.0, 1),
            (3.0, 4.0, 2),
        ])

    def test_stay_bias_keeps_previous_line_when_close_to_top(self):
        chunks = [
            (0.0, 1.0, [0.0, 80.0, 10.0]),
            (1.0, 2.0, [0.0, 77.0, 80.0]),
        ]
        segments = smooth_with_stay_bias(chunks, stay_margin=5.0, score_threshold=0.0)
        self.assertEqual([(s.start, s.end, s.line_idx) for s in segments], [
            (0.0, 2.0, 1),
        ])

    def test_viterbi_ignores_single_noisy_far_jump(self):
        chunks = [
            (0.0, 1.0, [0.0, 80.0, 10.0, 0.0, 0.0, 0.0]),
            # Local best is a far jump to line 5, but continuity should hold line 1.
            (1.0, 2.0, [0.0, 70.0, 10.0, 0.0, 0.0, 76.0]),
            (2.0, 3.0, [0.0, 78.0, 20.0, 0.0, 0.0, 0.0]),
        ]
        segments = smooth_with_viterbi(
            chunks,
            jump_penalty=4.0,
            backtrack_penalty=8.0,
            score_threshold=0.0,
        )
        self.assertEqual([(s.start, s.end, s.line_idx) for s in segments], [
            (0.0, 3.0, 1),
        ])

    def test_viterbi_allows_sustained_evidence_to_advance(self):
        chunks = [
            (0.0, 1.0, [0.0, 85.0, 10.0]),
            (1.0, 2.0, [0.0, 84.0, 15.0]),
            (2.0, 3.0, [0.0, 40.0, 92.0]),
            (3.0, 4.0, [0.0, 35.0, 94.0]),
        ]
        segments = smooth_with_viterbi(
            chunks,
            jump_penalty=4.0,
            backtrack_penalty=8.0,
            score_threshold=0.0,
        )
        self.assertEqual([(s.start, s.end, s.line_idx) for s in segments], [
            (0.0, 2.0, 1),
            (2.0, 4.0, 2),
        ])

    def test_viterbi_null_state_absorbs_low_confidence_filler(self):
        chunks = [
            (0.0, 1.0, [0.0, 82.0, 10.0]),
            (1.0, 2.0, [0.0, 20.0, 25.0]),
            (2.0, 3.0, [0.0, 18.0, 24.0]),
            (3.0, 4.0, [0.0, 15.0, 86.0]),
        ]
        segments = smooth_with_viterbi(
            chunks,
            jump_penalty=4.0,
            backtrack_penalty=8.0,
            score_threshold=0.0,
            null_score=45.0,
            null_switch_penalty=0.0,
        )
        self.assertEqual([(s.start, s.end, s.line_idx) for s in segments], [
            (0.0, 1.0, 1),
            (3.0, 4.0, 2),
        ])

    def test_viterbi_rejects_mismatched_score_vector_lengths(self):
        with self.assertRaises(ValueError):
            smooth_with_viterbi([
                (0.0, 1.0, [1.0, 2.0]),
                (1.0, 2.0, [1.0]),
            ])

    def test_simran_detector_requires_repeated_waheguru_tokens(self):
        self.assertTrue(is_simran_dominant("ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ"))
        self.assertFalse(is_simran_dominant("ਸੰਤ ਜੀਵਹਿ ਜਪਿ ਪ੍ਰਾਨ ਅਧਾਰਾ"))
        self.assertFalse(is_simran_dominant("ਵਾਹਿਗੁਰੂ ਜੀ"))

    def test_loop_align_nulls_simran_without_resetting_line_trajectory(self):
        chunks = [
            (0.0, 10.0, [0.0, 90.0, 10.0], "first line"),
            (10.0, 20.0, [0.0, 48.0, 47.0], "ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ"),
            (20.0, 30.0, [0.0, 84.0, 87.0], "first line still close"),
        ]
        segments = smooth_with_loop_align(
            chunks,
            stay_margin=5.0,
            score_threshold=0.0,
        )
        self.assertEqual([(s.start, s.end, s.line_idx) for s in segments], [
            (0.0, 10.0, 1),
            (20.0, 30.0, 1),
        ])

    def test_loop_align_confirmed_suppresses_single_far_jump(self):
        chunks = [
            (0.0, 10.0, [0.0, 90.0, 10.0, 0.0, 0.0, 0.0], "line one"),
            (10.0, 20.0, [0.0, 70.0, 10.0, 0.0, 0.0, 76.0], "noisy far line"),
            (20.0, 30.0, [0.0, 88.0, 12.0, 0.0, 0.0, 0.0], "line one again"),
        ]
        segments = smooth_with_loop_align_confirmed(
            chunks,
            stay_margin=5.0,
            confirm_chunks=2,
            hard_jump_margin=15.0,
        )
        self.assertEqual([(s.start, s.end, s.line_idx) for s in segments], [
            (0.0, 30.0, 1),
        ])

    def test_loop_align_confirmed_allows_sustained_far_jump(self):
        chunks = [
            (0.0, 10.0, [0.0, 90.0, 10.0, 0.0, 0.0, 0.0], "line one"),
            (10.0, 20.0, [0.0, 65.0, 10.0, 0.0, 0.0, 82.0], "far line"),
            (20.0, 30.0, [0.0, 60.0, 12.0, 0.0, 0.0, 84.0], "far line continues"),
        ]
        segments = smooth_with_loop_align_confirmed(
            chunks,
            stay_margin=5.0,
            confirm_chunks=2,
            hard_jump_margin=15.0,
        )
        self.assertEqual([(s.start, s.end, s.line_idx) for s in segments], [
            (0.0, 10.0, 1),
            (10.0, 30.0, 5),
        ])


if __name__ == "__main__":
    unittest.main()
