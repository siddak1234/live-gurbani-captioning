"""Unit tests for benchmark/OOS audio fetching helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.fetch_audio import (  # noqa: E402
    collect_video_ids,
    fetch_one_url,
    parse_clip_specs,
    parse_url_specs,
)


class TestParseUrlSpecs(unittest.TestCase):
    def test_parses_case_id_url_pairs(self):
        targets = parse_url_specs([
            "rep_01=https://youtube.com/watch?v=abc123",
            "stress_01=https://archive.org/details/some-item",
        ])
        self.assertEqual(targets, [
            ("rep_01", "https://youtube.com/watch?v=abc123"),
            ("stress_01", "https://archive.org/details/some-item"),
        ])

    def test_splits_only_once(self):
        targets = parse_url_specs(["case_a=https://example.com/watch?a=1&b=2"])
        self.assertEqual(targets, [("case_a", "https://example.com/watch?a=1&b=2")])

    def test_rejects_missing_equals(self):
        with self.assertRaises(ValueError):
            parse_url_specs(["https://youtube.com/watch?v=abc123"])

    def test_rejects_empty_case_id(self):
        with self.assertRaises(ValueError):
            parse_url_specs(["=https://youtube.com/watch?v=abc123"])

    def test_rejects_empty_url(self):
        with self.assertRaises(ValueError):
            parse_url_specs(["case_a="])

    def test_rejects_path_separator_in_case_id(self):
        with self.assertRaises(ValueError):
            parse_url_specs(["../case=https://youtube.com/watch?v=abc123"])


class TestParseClipSpecs(unittest.TestCase):
    def test_parses_case_clip_windows(self):
        clips = parse_clip_specs(["case_a=30-210", "case_b=0.5-60.25"])
        self.assertEqual(clips, {
            "case_a": (30.0, 210.0),
            "case_b": (0.5, 60.25),
        })

    def test_rejects_missing_equals(self):
        with self.assertRaises(ValueError):
            parse_clip_specs(["case_a:30-210"])

    def test_rejects_non_numeric_window(self):
        with self.assertRaises(ValueError):
            parse_clip_specs(["case_a=start-end"])

    def test_rejects_empty_or_path_case_id(self):
        with self.assertRaises(ValueError):
            parse_clip_specs(["=30-210"])
        with self.assertRaises(ValueError):
            parse_clip_specs(["../case=30-210"])

    def test_rejects_non_positive_duration(self):
        with self.assertRaises(ValueError):
            parse_clip_specs(["case_a=30-30"])
        with self.assertRaises(ValueError):
            parse_clip_specs(["case_a=30-20"])


class TestCollectVideoIds(unittest.TestCase):
    def test_collects_unique_sorted_video_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            gt_dir = Path(tmp)
            (gt_dir / "b.json").write_text(json.dumps({"video_id": "vid_b"}))
            (gt_dir / "a.json").write_text(json.dumps({"video_id": "vid_a"}))
            (gt_dir / "dup.json").write_text(json.dumps({"video_id": "vid_b"}))
            self.assertEqual(collect_video_ids(gt_dir), ["vid_a", "vid_b"])


class TestFetchOneUrl(unittest.TestCase):
    def test_existing_output_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_dir = Path(tmp)
            out = audio_dir / "case_a_16k.wav"
            out.write_bytes(b"already here")
            with mock.patch("scripts.fetch_audio.subprocess.run") as run:
                self.assertTrue(fetch_one_url("case_a", "https://example.com/audio", audio_dir))
            run.assert_not_called()

    def test_download_and_convert_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_dir = Path(tmp)
            raw = audio_dir / ".case_a.wav"

            def fake_run(cmd, check, capture_output, text):
                if cmd[0] == "yt-dlp":
                    raw.write_bytes(b"raw wav")
                elif cmd[0] == "ffmpeg":
                    Path(cmd[-1]).write_bytes(b"converted wav")
                return mock.Mock(returncode=0)

            with mock.patch("scripts.fetch_audio.subprocess.run", side_effect=fake_run) as run:
                self.assertTrue(fetch_one_url("case_a", "https://example.com/audio", audio_dir))

            self.assertEqual(run.call_count, 2)
            self.assertFalse(raw.exists())
            self.assertTrue((audio_dir / "case_a_16k.wav").exists())

    def test_download_and_convert_with_clip_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_dir = Path(tmp)
            raw = audio_dir / ".case_a.wav"

            def fake_run(cmd, check, capture_output, text):
                if cmd[0] == "yt-dlp":
                    raw.write_bytes(b"raw wav")
                elif cmd[0] == "ffmpeg":
                    Path(cmd[-1]).write_bytes(b"converted wav")
                return mock.Mock(returncode=0)

            with mock.patch("scripts.fetch_audio.subprocess.run", side_effect=fake_run) as run:
                self.assertTrue(
                    fetch_one_url("case_a", "https://example.com/audio", audio_dir, (30.0, 210.0))
                )

            ffmpeg_cmd = run.call_args_list[1].args[0]
            self.assertIn("-ss", ffmpeg_cmd)
            self.assertIn("30.000", ffmpeg_cmd)
            self.assertIn("-t", ffmpeg_cmd)
            self.assertIn("180.000", ffmpeg_cmd)


if __name__ == "__main__":
    unittest.main()
