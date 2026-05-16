from __future__ import annotations

import pathlib
import unittest
from unittest.mock import patch

from src.engine import EngineConfig, PredictionResult, Segment
from src.idlock_engine import merge_segments_at_commit, predict_idlocked


def _seg(start: float, end: float, line_idx: int, shabad_id: int = 1) -> Segment:
    return Segment(
        start=start,
        end=end,
        line_idx=line_idx,
        shabad_id=shabad_id,
        verse_id=f"v{line_idx}",
        banidb_gurmukhi=f"line {line_idx}",
    )


class TestMergeSegmentsAtCommit(unittest.TestCase):
    def test_merges_and_truncates_segments_at_commit(self):
        pre = [_seg(0, 10, 1), _seg(10, 35, 2), _seg(35, 45, 3)]
        post = [_seg(20, 40, 4), _seg(40, 50, 5)]
        merged = merge_segments_at_commit(pre, post, commit_time=30.0)

        self.assertEqual(
            [(s.start, s.end, s.line_idx) for s in merged],
            [(0, 10, 1), (10, 30.0, 2), (30.0, 40, 4), (40, 50, 5)],
        )

    def test_drops_zero_width_segments(self):
        merged = merge_segments_at_commit(
            [_seg(30, 40, 1)],
            [_seg(20, 30, 2), _seg(30, 40, 3)],
            commit_time=30.0,
        )
        self.assertEqual([(s.start, s.end, s.line_idx) for s in merged], [(30, 40, 3)])

    def test_does_not_mutate_input_segments(self):
        pre = [_seg(10, 35, 2)]
        post = [_seg(20, 40, 4)]
        merge_segments_at_commit(pre, post, commit_time=30.0)
        self.assertEqual((pre[0].start, pre[0].end), (10, 35))
        self.assertEqual((post[0].start, post[0].end), (20, 40))


class TestPredictIdlocked(unittest.TestCase):
    def test_buffered_mode_orchestrates_two_engine_calls(self):
        calls: list[dict] = []

        def fake_predict(audio, corpora, *, shabad_id, uem_start, config):
            calls.append({
                "audio": audio,
                "shabad_id": shabad_id,
                "uem_start": uem_start,
                "config": config,
            })
            if shabad_id is None:
                return PredictionResult(
                    segments=[_seg(0, 35, 1, shabad_id=99)],
                    shabad_id=99,
                    n_chunks=6,
                    blind_id_score=88.0,
                    blind_runner_up_score=70.0,
                )
            return PredictionResult(
                segments=[_seg(25, 55, 2, shabad_id=shabad_id)],
                shabad_id=shabad_id,
                n_chunks=7,
            )

        pre = EngineConfig(backend="faster_whisper", model_size="medium", blind_lookback=30.0)
        post = EngineConfig(
            backend="huggingface_whisper",
            model_size="surindersinghssj/surt-small-v3",
            adapter_dir="lora_adapters/v5b_mac_diverse",
            live=True,  # should be overridden by buffered mode
        )

        with patch("src.idlock_engine.predict", side_effect=fake_predict):
            result = predict_idlocked(
                pathlib.Path("case.wav"),
                {99: []},
                uem_start=12.5,
                pre_config=pre,
                post_config=post,
                post_context="buffered",
            )

        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0]["shabad_id"])
        self.assertTrue(calls[0]["config"].live)
        self.assertTrue(calls[0]["config"].tentative_emit)
        self.assertEqual(calls[1]["shabad_id"], 99)
        self.assertFalse(calls[1]["config"].live)
        self.assertFalse(calls[1]["config"].tentative_emit)

        self.assertEqual(result.commit_time, 42.5)
        self.assertEqual(result.prediction.shabad_id, 99)
        self.assertEqual(result.prediction.n_chunks, 13)
        self.assertEqual(result.prediction.blind_id_score, 88.0)
        self.assertEqual(
            [(s.start, s.end, s.line_idx) for s in result.prediction.segments],
            [(0, 35, 1), (42.5, 55, 2)],
        )

    def test_strict_live_mode_makes_post_lock_engine_causal(self):
        configs: list[EngineConfig] = []

        def fake_predict(audio, corpora, *, shabad_id, uem_start, config):
            configs.append(config)
            if shabad_id is None:
                return PredictionResult(segments=[], shabad_id=7, n_chunks=1)
            return PredictionResult(segments=[], shabad_id=shabad_id, n_chunks=1)

        pre = EngineConfig(blind_lookback=18.0)
        post = EngineConfig(live=False, blind_lookback=99.0)

        with patch("src.idlock_engine.predict", side_effect=fake_predict):
            result = predict_idlocked(
                pathlib.Path("case.wav"),
                {7: []},
                pre_config=pre,
                post_config=post,
                post_context="strict-live",
            )

        self.assertEqual(result.commit_time, 18.0)
        self.assertTrue(configs[1].live)
        self.assertEqual(configs[1].blind_lookback, 18.0)

    def test_rejects_unknown_post_context(self):
        with self.assertRaises(ValueError):
            predict_idlocked(pathlib.Path("case.wav"), {}, post_context="future")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
