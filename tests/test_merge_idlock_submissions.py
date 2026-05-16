from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.merge_idlock_submissions import merge_at_commit, merge_one  # noqa: E402


def _seg(start: float, end: float, line_idx: int, shabad_id: int = 1) -> dict:
    return {
        "start": start,
        "end": end,
        "line_idx": line_idx,
        "shabad_id": shabad_id,
        "verse_id": f"v{line_idx}",
        "banidb_gurmukhi": f"line {line_idx}",
    }


class TestMergeAtCommit(unittest.TestCase):
    def test_merges_and_truncates_segments_at_commit(self):
        pre = [_seg(0, 10, 1), _seg(10, 35, 2), _seg(35, 45, 3)]
        post = [_seg(20, 40, 4), _seg(40, 50, 5)]
        merged = merge_at_commit(pre, post, commit_time=30.0)

        self.assertEqual(
            [(s["start"], s["end"], s["line_idx"]) for s in merged],
            [(0, 10, 1), (10, 30.0, 2), (30.0, 40, 4), (40, 50, 5)],
        )

    def test_drops_zero_width_segments(self):
        merged = merge_at_commit(
            [_seg(30, 40, 1)],
            [_seg(20, 30, 2), _seg(30, 40, 3)],
            commit_time=30.0,
        )
        self.assertEqual([(s["start"], s["end"], s["line_idx"]) for s in merged], [(30, 40, 3)])


class TestMergeOne(unittest.TestCase):
    def test_merge_one_uses_uem_start_plus_lookback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_dir = root / "gt"
            pre_dir = root / "pre"
            post_dir = root / "post"
            out_dir = root / "out"
            for d in (gt_dir, pre_dir, post_dir):
                d.mkdir()

            gt = {"video_id": "case", "uem": {"start": 12.5}, "segments": []}
            pre = {"video_id": "case", "segments": [_seg(0, 50, 1)]}
            post = {"video_id": "case", "segments": [_seg(30, 60, 2)]}
            (gt_dir / "case.json").write_text(json.dumps(gt))
            (pre_dir / "case.json").write_text(json.dumps(pre))
            (post_dir / "case.json").write_text(json.dumps(post))

            with redirect_stdout(StringIO()):
                merge_one(gt_dir / "case.json", pre_dir, post_dir, out_dir, lookback_s=30.0)
            merged = json.loads((out_dir / "case.json").read_text())

            self.assertEqual(merged["video_id"], "case")
            self.assertEqual(
                [(s["start"], s["end"], s["line_idx"]) for s in merged["segments"]],
                [(0, 42.5, 1), (42.5, 60, 2)],
            )

    def test_merge_one_rejects_video_id_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_dir = root / "gt"
            pre_dir = root / "pre"
            post_dir = root / "post"
            out_dir = root / "out"
            for d in (gt_dir, pre_dir, post_dir):
                d.mkdir()
            (gt_dir / "case.json").write_text(json.dumps({"video_id": "case", "segments": []}))
            (pre_dir / "case.json").write_text(json.dumps({"video_id": "case", "segments": []}))
            (post_dir / "case.json").write_text(json.dumps({"video_id": "other", "segments": []}))

            with self.assertRaises(ValueError):
                merge_one(gt_dir / "case.json", pre_dir, post_dir, out_dir, lookback_s=30.0)


if __name__ == "__main__":
    unittest.main()
