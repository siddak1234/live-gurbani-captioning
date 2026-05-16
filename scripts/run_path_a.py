#!/usr/bin/env python3
"""Path A benchmark runner: thin I/O shim around ``src/engine.predict()``.

For each GT case:
  1. Load GT JSON (here)
  2. Resolve audio path (here)
  3. Call ``engine.predict()`` (engine library)
  4. Serialize segments to submission JSON (here)

All inference logic lives in ``src/engine.py``. This file knows about the
benchmark layout; the engine doesn't.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.engine import predict, EngineConfig  # noqa: E402
from src.streaming_engine import StreamingEngine  # noqa: E402

DEFAULT_GT_DIR = REPO_ROOT.parent / "live-gurbani-captioning-benchmark-v1" / "test"
DEFAULT_AUDIO_DIR = REPO_ROOT / "audio"
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus_cache"
DEFAULT_ASR_CACHE = REPO_ROOT / "asr_cache"
DEFAULT_OUT_DIR = REPO_ROOT / "submissions" / "v1_pathA_oracle"


def _run_batch(audio_path, corpora, shabad_id, uem_start, engine_config):
    """Standard batch path — full audio in, segments out."""
    return predict(audio_path, corpora,
                   shabad_id=shabad_id, uem_start=uem_start, config=engine_config)


def _run_streaming(audio_path, corpora, shabad_id, uem_start,
                   engine_config, chunk_s: float):
    """Simulate iOS-style streaming by feeding the file through StreamingEngine.

    The benchmark audit gate: this path must produce segments equivalent to
    batch within 1 s tolerance, validating that the streaming contract is
    correct before Swift mirrors it in M5.
    """
    import soundfile as sf
    from src.engine import PredictionResult

    audio_np, sr = sf.read(str(audio_path), dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    # Size the ring to hold the full file so nothing ages out during the audit.
    capacity_s = max(120.0, float(len(audio_np)) / sr + 10.0)
    eng = StreamingEngine(
        corpora, config=engine_config,
        buffer_capacity_s=capacity_s,
        process_interval_s=chunk_s,
    )
    eng.reset(shabad_id=shabad_id)

    chunk_samples = int(chunk_s * sr)
    for i in range(0, len(audio_np), chunk_samples):
        eng.process_pcm(audio_np[i : i + chunk_samples], sr=sr)
    segments = eng.flush()
    state = eng.get_state()

    # Shim into the PredictionResult contract so the caller doesn't branch.
    return PredictionResult(
        segments=segments,
        shabad_id=state.committed_shabad_id if state.committed_shabad_id is not None else 0,
        n_chunks=len(segments),
        blind_id_score=None,
        blind_runner_up_score=None,
    )


def process_one(
    gt_path: pathlib.Path,
    *,
    audio_dir: pathlib.Path,
    corpora: dict[int, list[dict]],
    out_dir: pathlib.Path,
    blind: bool,
    engine_config: EngineConfig,
    streaming: bool = False,
    streaming_chunk_s: float = 5.0,
) -> bool:
    gt = json.loads(gt_path.read_text())
    video_id = gt["video_id"]
    gt_shabad_id = gt["shabad_id"]

    audio_path = audio_dir / f"{video_id}_16k.wav"
    if not audio_path.exists():
        print(f"  error: audio missing {audio_path}", file=sys.stderr)
        return False

    uem_start = float(gt.get("uem", {}).get("start", 0.0))
    shabad_for_engine = None if blind else gt_shabad_id

    try:
        if streaming:
            result = _run_streaming(
                audio_path, corpora, shabad_for_engine, uem_start,
                engine_config, chunk_s=streaming_chunk_s,
            )
        else:
            result = _run_batch(
                audio_path, corpora, shabad_for_engine, uem_start, engine_config,
            )
    except ValueError as e:
        print(f"  error: {e}", file=sys.stderr)
        return False

    if blind and not streaming:
        sid_correct = "✓" if result.shabad_id == gt_shabad_id else "✗"
        print(f"  {gt_path.stem}: blind ID predicts {result.shabad_id} (GT {gt_shabad_id}) {sid_correct} "
              f"top={result.blind_id_score:.1f} runner_up={result.blind_runner_up_score:.1f}")

    submission_segments = [
        {
            "start": s.start,
            "end": s.end,
            "line_idx": s.line_idx,
            "shabad_id": s.shabad_id,
            "verse_id": s.verse_id,
            "banidb_gurmukhi": s.banidb_gurmukhi,
        }
        for s in result.segments
    ]

    out_path = out_dir / gt_path.name
    out_path.write_text(json.dumps(
        {"video_id": video_id, "segments": submission_segments},
        ensure_ascii=False, indent=2,
    ))
    mode_tag = "streaming" if streaming else "batch"
    print(f"  {gt_path.stem}: [{mode_tag}] {result.n_chunks} chunks → {len(result.segments)} segments")
    return True


def _parse_blend(spec: str) -> dict[str, float] | None:
    if not spec.strip():
        return None
    out: dict[str, float] = {}
    for part in spec.split(","):
        name, w = part.split(":")
        out[name.strip()] = float(w)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=pathlib.Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--audio-dir", type=pathlib.Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--corpus-dir", type=pathlib.Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--asr-cache-dir", type=pathlib.Path, default=DEFAULT_ASR_CACHE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--backend", default="faster_whisper",
                        choices=["faster_whisper", "mlx_whisper", "huggingface_whisper"],
                        help="ASR backend; default reproduces v3.2 (Path A canonical). "
                             "Use 'huggingface_whisper' with --model <user/repo> for "
                             "custom-fine-tuned models (e.g. surindersinghssj/surt-small-v3).")
    parser.add_argument("--model", default="medium")
    parser.add_argument("--threshold", type=float, default=55.0)
    parser.add_argument("--margin", type=float, default=0.0,
                        help="Minimum top1-top2 score gap; below it the chunk is null")
    parser.add_argument("--ratio", default="WRatio",
                        help="rapidfuzz scorer to use (WRatio, ratio, token_sort_ratio, ...)")
    parser.add_argument("--blend", default="",
                        help='Weighted blend, e.g. "token_sort_ratio:0.7,ratio:0.3"')
    parser.add_argument("--stay-bias", type=float, default=0.0,
                        help="Stay-bias margin; if >0, prefer previous line when within margin")
    parser.add_argument("--blind", action="store_true",
                        help="Identify shabad from audio instead of using GT shabad_id")
    parser.add_argument("--blind-lookback", type=float, default=30.0,
                        help="Seconds of audio to buffer for blind shabad ID")
    parser.add_argument("--blind-aggregate", default="max",
                        help='Aggregation: "max" or "topk:N" (default max)')
    parser.add_argument("--blind-ratio", default="WRatio",
                        help="Scorer for blind shabad ID (independent from --ratio)")
    parser.add_argument("--blind-blend", default="",
                        help="Blend for blind shabad ID, same format as --blend")
    parser.add_argument("--live", action="store_true",
                        help="Causal: matcher only processes chunks after shabad-ID commit time")
    parser.add_argument("--tentative-emit", action="store_true",
                        help="In live mode, emit per-chunk global-best (shabad, line) during ID buffer")
    parser.add_argument("--adapter-dir", default=None,
                        help="Path to a LoRA/PEFT adapter (only used with huggingface_whisper backend)")
    parser.add_argument("--word-timestamps", action="store_true",
                        help="Pass word_timestamps=True to ASR. Phase 2.8 timing probe.")
    parser.add_argument("--vad-filter", action="store_true",
                        help="Enable faster-whisper Silero VAD filtering. Default is off.")
    parser.add_argument("--no-speech-threshold", type=float, default=None,
                        help="Override faster-whisper no_speech_threshold (default: backend default).")
    parser.add_argument("--streaming", action="store_true",
                        help="Route through StreamingEngine (iOS-shape contract) instead of "
                             "batch predict(). Audit gate for M3.3 / iOS Swift port.")
    parser.add_argument("--streaming-chunk-s", type=float, default=5.0,
                        help="Streaming chunk size in seconds (default 5.0). Smaller = lower "
                             "latency at higher redundant-work cost.")
    args = parser.parse_args()

    engine_config = EngineConfig(
        backend=args.backend,
        model_size=args.model,
        adapter_dir=args.adapter_dir,
        asr_cache_dir=args.asr_cache_dir.resolve(),
        word_timestamps=args.word_timestamps,
        no_speech_threshold=args.no_speech_threshold,
        vad_filter=args.vad_filter,
        ratio=args.ratio,
        blend=_parse_blend(args.blend),
        score_threshold=args.threshold,
        margin_threshold=args.margin,
        stay_bias=args.stay_bias,
        blind_lookback=args.blind_lookback,
        blind_aggregate=args.blind_aggregate,
        blind_ratio=args.blind_ratio,
        blind_blend=_parse_blend(args.blind_blend),
        live=args.live,
        tentative_emit=args.tentative_emit,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt_files = sorted(args.gt_dir.glob("*.json"))

    corpus_dir = args.corpus_dir.resolve()
    corpora: dict[int, list[dict]] = {}
    for cf in sorted(corpus_dir.glob("*.json")):
        c = json.loads(cf.read_text())
        corpora[int(c["shabad_id"])] = c["lines"]
    if not corpora:
        print(f"error: no corpus files in {corpus_dir}", file=sys.stderr)
        return 1

    label = f"blend={engine_config.blend}" if engine_config.blend else f"ratio={engine_config.ratio}"
    mode_parts = []
    mode_parts.append("BLIND" if args.blind else "ORACLE")
    mode_parts.append("LIVE" if args.live else "OFFLINE")
    mode = "+".join(mode_parts)
    print(f"processing {len(gt_files)} GT cases "
          f"(mode={mode}, model={args.model}, {label}, threshold={args.threshold}, "
          f"margin={args.margin}, stay_bias={args.stay_bias})\n")
    failures: list[str] = []
    for gt_file in gt_files:
        if not process_one(
            gt_file,
            audio_dir=args.audio_dir.resolve(),
            corpora=corpora,
            out_dir=args.out_dir.resolve(),
            blind=args.blind,
            engine_config=engine_config,
            streaming=args.streaming,
            streaming_chunk_s=args.streaming_chunk_s,
        ):
            failures.append(gt_file.stem)

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1
    print(f"\nwrote {len(gt_files)} submissions to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
