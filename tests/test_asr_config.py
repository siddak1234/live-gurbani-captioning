from __future__ import annotations

import pathlib
import unittest

from src.asr import _cache_path, _extra_tag


class TestAsrExtraTag(unittest.TestCase):
    def test_default_preserves_legacy_cache_key(self):
        self.assertEqual(_extra_tag(), "")

    def test_word_and_vad_tags_are_explicit(self):
        self.assertEqual(
            _extra_tag(word_timestamps=True, vad_filter=True),
            "_word_vad",
        )

    def test_no_speech_threshold_and_adapter_tags(self):
        self.assertEqual(
            _extra_tag(
                no_speech_threshold=0.3,
                adapter_dir="lora_adapters/v5b_mac_diverse",
            ),
            "_nst0.3_lora-v5b_mac_diverse",
        )


class TestAsrCachePath(unittest.TestCase):
    def test_faster_whisper_default_keeps_legacy_name(self):
        path = _cache_path(
            pathlib.Path("audio/case_16k.wav"),
            pathlib.Path("asr_cache"),
            "faster_whisper",
            "medium",
            "pa",
            "",
        )
        self.assertEqual(path, pathlib.Path("asr_cache/case_16k__medium__pa.json"))

    def test_probe_tags_do_not_collide_with_baseline(self):
        path = _cache_path(
            pathlib.Path("audio/case_16k.wav"),
            pathlib.Path("asr_cache"),
            "faster_whisper",
            "medium",
            "pa",
            "_word_vad",
        )
        self.assertEqual(path, pathlib.Path("asr_cache/case_16k__medium_word_vad__pa.json"))


if __name__ == "__main__":
    unittest.main()
