#!/usr/bin/env python3
"""Phase 2.7 runtime ID-lock benchmark runner.

Thin I/O shim around ``src.idlock_engine.predict_idlocked``:

  1. Load benchmark GT JSON.
  2. Resolve audio and corpus paths.
  3. Build pre-lock and post-lock EngineConfig objects.
  4. Call the library orchestrator.
  5. Serialize submission JSON.

Inference composition lives in ``src/idlock_engine.py``. This script owns only
benchmark layout, argparse, and output files.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.engine import EngineConfig  # noqa: E402
from src.idlock_engine import MergePolicy, PostContextMode, predict_idlocked  # noqa: E402

DEFAULT_GT_DIR = REPO_ROOT.parent / "live-gurbani-captioning-benchmark-v1" / "test"
DEFAULT_AUDIO_DIR = REPO_ROOT / "audio"
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus_cache"
DEFAULT_ASR_CACHE = REPO_ROOT / "asr_cache"
DEFAULT_OUT_DIR = REPO_ROOT / "submissions" / "v5b_idlock_runtime"


def _parse_blend(spec: str) -> dict[str, float] | None:
    if not spec.strip():
        return None
    out: dict[str, float] = {}
    for part in spec.split(","):
        name, weight = part.split(":")
        out[name.strip()] = float(weight)
    return out


def _load_corpora(corpus_dir: pathlib.Path) -> dict[int, list[dict]]:
    corpora: dict[int, list[dict]] = {}
    for path in sorted(corpus_dir.glob("*.json")):
        corpus = json.loads(path.read_text())
        corpora[int(corpus["shabad_id"])] = corpus["lines"]
    return corpora


def process_one(
    gt_path: pathlib.Path,
    *,
    audio_dir: pathlib.Path,
    corpora: dict[int, list[dict]],
    out_dir: pathlib.Path,
    pre_config: EngineConfig,
    post_config: EngineConfig,
    post_context: PostContextMode,
    merge_policy: MergePolicy,
) -> bool:
    gt = json.loads(gt_path.read_text())
    video_id = gt["video_id"]
    gt_shabad_id = gt["shabad_id"]
    audio_path = audio_dir / f"{video_id}_16k.wav"
    if not audio_path.exists():
        print(f"  error: audio missing {audio_path}", file=sys.stderr)
        return False

    try:
        result = predict_idlocked(
            audio_path,
            corpora,
            uem_start=float(gt.get("uem", {}).get("start", 0.0)),
            pre_config=pre_config,
            post_config=post_config,
            post_context=post_context,
            merge_policy=merge_policy,
        )
    except ValueError as exc:
        print(f"  error: {exc}", file=sys.stderr)
        return False

    sid_status = "OK" if result.prediction.shabad_id == gt_shabad_id else "BAD"
    top = result.prediction.blind_id_score
    runner_up = result.prediction.blind_runner_up_score
    top_s = "None" if top is None else f"{top:.1f}"
    runner_s = "None" if runner_up is None else f"{runner_up:.1f}"
    print(
        f"  {gt_path.stem}: lock={result.commit_time:.1f}s "
        f"sid={result.prediction.shabad_id} (GT {gt_shabad_id}) {sid_status} "
        f"top={top_s} runner_up={runner_s} "
        f"pre={len(result.pre_lock.segments)} post={len(result.post_lock.segments)} "
        f"merged={len(result.prediction.segments)}",
        flush=True,
    )

    submission_segments = [
        {
            "start": segment.start,
            "end": segment.end,
            "line_idx": segment.line_idx,
            "shabad_id": segment.shabad_id,
            "verse_id": segment.verse_id,
            "banidb_gurmukhi": segment.banidb_gurmukhi,
        }
        for segment in result.prediction.segments
    ]
    out_path = out_dir / gt_path.name
    out_path.write_text(json.dumps(
        {"video_id": video_id, "segments": submission_segments},
        ensure_ascii=False,
        indent=2,
    ))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=pathlib.Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--audio-dir", type=pathlib.Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--corpus-dir", type=pathlib.Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--asr-cache-dir", type=pathlib.Path, default=DEFAULT_ASR_CACHE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)

    parser.add_argument("--pre-backend", default="faster_whisper",
                        choices=["faster_whisper", "mlx_whisper", "huggingface_whisper"])
    parser.add_argument("--pre-model", default="medium")
    parser.add_argument("--post-backend", default="huggingface_whisper",
                        choices=["faster_whisper", "mlx_whisper", "huggingface_whisper"])
    parser.add_argument("--post-model", default="surindersinghssj/surt-small-v3")
    parser.add_argument("--post-adapter-dir", default="lora_adapters/v5b_mac_diverse")

    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--ratio", default="WRatio")
    parser.add_argument("--blend", default="token_sort_ratio:0.5,WRatio:0.5")
    parser.add_argument("--stay-bias", type=float, default=6.0)
    parser.add_argument("--smoother", default="auto",
                        choices=["auto", "basic", "stay_bias", "viterbi"],
                        help="Post/pre smoother implementation. auto preserves historical behavior: "
                             "stay_bias when --stay-bias > 0, otherwise basic.")
    parser.add_argument("--viterbi-jump-penalty", type=float, default=4.0,
                        help="Penalty per line-distance for the Viterbi sequence smoother.")
    parser.add_argument("--viterbi-backtrack-penalty", type=float, default=8.0,
                        help="Extra penalty per backward line-distance for Viterbi smoothing.")
    parser.add_argument("--viterbi-null-score", type=float, default=None,
                        help="Enable Viterbi no-line state with this constant local score.")
    parser.add_argument("--viterbi-null-switch-penalty", type=float, default=0.0,
                        help="Penalty for entering/exiting Viterbi no-line state.")
    parser.add_argument("--blind-lookback", type=float, default=30.0)
    parser.add_argument("--blind-aggregate", default="chunk_vote")
    parser.add_argument("--blind-ratio", default="WRatio")
    parser.add_argument("--blind-blend", default="")
    parser.add_argument("--pre-word-timestamps", action="store_true",
                        help="Use word_timestamps=True for the pre-lock ASR engine.")
    parser.add_argument("--pre-vad-filter", action="store_true",
                        help="Enable faster-whisper VAD filtering for the pre-lock ASR engine.")
    parser.add_argument("--pre-no-speech-threshold", type=float, default=None,
                        help="Override no_speech_threshold for the pre-lock ASR engine.")
    parser.add_argument("--post-context", choices=["buffered", "strict-live"],
                        default="buffered",
                        help="buffered uses pre-lock transcript as post-lock smoother context; "
                             "strict-live makes post-lock matching ignore pre-commit chunks.")
    parser.add_argument("--merge-policy", choices=["commit-cutover", "retro-buffered"],
                        default="commit-cutover",
                        help="commit-cutover preserves tentative pre-lock segments before the "
                             "ID commit; retro-buffered lets the locked post engine revise "
                             "the buffered window from UEM start.")
    args = parser.parse_args()

    corpus_dir = args.corpus_dir.resolve()
    corpora = _load_corpora(corpus_dir)
    if not corpora:
        print(f"error: no corpus files in {corpus_dir}", file=sys.stderr)
        return 1

    pre_config = EngineConfig(
        backend=args.pre_backend,
        model_size=args.pre_model,
        asr_cache_dir=args.asr_cache_dir.resolve(),
        word_timestamps=args.pre_word_timestamps,
        no_speech_threshold=args.pre_no_speech_threshold,
        vad_filter=args.pre_vad_filter,
        ratio=args.ratio,
        blend=_parse_blend(args.blend),
        score_threshold=args.threshold,
        stay_bias=args.stay_bias,
        smoother=args.smoother,
        viterbi_jump_penalty=args.viterbi_jump_penalty,
        viterbi_backtrack_penalty=args.viterbi_backtrack_penalty,
        viterbi_null_score=args.viterbi_null_score,
        viterbi_null_switch_penalty=args.viterbi_null_switch_penalty,
        blind_lookback=args.blind_lookback,
        blind_aggregate=args.blind_aggregate,
        blind_ratio=args.blind_ratio,
        blind_blend=_parse_blend(args.blind_blend),
        live=True,
        tentative_emit=True,
    )
    post_config = EngineConfig(
        backend=args.post_backend,
        model_size=args.post_model,
        adapter_dir=args.post_adapter_dir,
        asr_cache_dir=args.asr_cache_dir.resolve(),
        ratio=args.ratio,
        blend=_parse_blend(args.blend),
        score_threshold=args.threshold,
        stay_bias=args.stay_bias,
        smoother=args.smoother,
        viterbi_jump_penalty=args.viterbi_jump_penalty,
        viterbi_backtrack_penalty=args.viterbi_backtrack_penalty,
        viterbi_null_score=args.viterbi_null_score,
        viterbi_null_switch_penalty=args.viterbi_null_switch_penalty,
        blind_lookback=args.blind_lookback,
        live=False,
        tentative_emit=False,
    )

    gt_files = sorted(args.gt_dir.glob("*.json"))
    if not gt_files:
        print(f"error: no GT JSON files in {args.gt_dir}", file=sys.stderr)
        return 1

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"processing {len(gt_files)} GT cases "
        f"(pre={args.pre_backend}:{args.pre_model}, "
        f"post={args.post_backend}:{args.post_model}, "
        f"adapter={args.post_adapter_dir}, post_context={args.post_context}, "
        f"merge_policy={args.merge_policy}, lookback={args.blind_lookback}s, "
        f"smoother={args.smoother})",
        flush=True,
    )
    failures: list[str] = []
    for gt_file in gt_files:
        ok = process_one(
            gt_file,
            audio_dir=args.audio_dir.resolve(),
            corpora=corpora,
            out_dir=out_dir,
            pre_config=pre_config,
            post_config=post_config,
            post_context=args.post_context,  # type: ignore[arg-type]
            merge_policy=args.merge_policy,  # type: ignore[arg-type]
        )
        if not ok:
            failures.append(gt_file.stem)

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1
    print(f"\nwrote {len(gt_files)} ID-lock submissions to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
