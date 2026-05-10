#!/usr/bin/env python3
"""Stage 2 / Path A runner: Whisper + fuzzy match + smoothing (oracle, offline).

For each GT case:
  1. Transcribe audio with faster-whisper (cached).
  2. Match each ASR chunk to a corpus line (oracle: GT shabad_id given).
  3. Smooth chunk-level matches into segments.
  4. Emit submission JSON with line_idx + verse_id + banidb_gurmukhi.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.asr import transcribe                                      # noqa: E402
from src.matcher import match_chunk, score_chunk, normalize, TfidfScorer  # noqa: E402
from src.shabad_id import identify_shabad, per_chunk_global_match    # noqa: E402
from src.smoother import smooth, smooth_with_stay_bias               # noqa: E402

DEFAULT_GT_DIR = REPO_ROOT.parent / "live-gurbani-captioning-benchmark-v1" / "test"
DEFAULT_AUDIO_DIR = REPO_ROOT / "audio"
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus_cache"
DEFAULT_ASR_CACHE = REPO_ROOT / "asr_cache"
DEFAULT_OUT_DIR = REPO_ROOT / "submissions" / "v1_pathA_oracle"


def process_one(
    gt_path: pathlib.Path,
    *,
    audio_dir: pathlib.Path,
    corpora: dict[int, list[dict]],
    asr_cache_dir: pathlib.Path,
    out_dir: pathlib.Path,
    model_size: str,
    score_threshold: float,
    margin_threshold: float,
    ratio: str,
    blend: dict[str, float] | None,
    stay_bias: float,
    blind: bool,
    blind_lookback: float,
    blind_aggregate: str,
    blind_ratio: str,
    blind_blend: dict[str, float] | None,
    live: bool,
    tentative_emit: bool,
    backend: str,
) -> bool:
    gt = json.loads(gt_path.read_text())
    video_id = gt["video_id"]
    gt_shabad_id = gt["shabad_id"]

    audio_path = audio_dir / f"{video_id}_16k.wav"
    if not audio_path.exists():
        print(f"  error: audio missing {audio_path}", file=sys.stderr)
        return False

    chunks = transcribe(audio_path, backend=backend, model_size=model_size, cache_dir=asr_cache_dir)

    uem_start = float(gt.get("uem", {}).get("start", 0.0))
    commit_time = uem_start + blind_lookback

    if blind:
        sid_result = identify_shabad(
            chunks, corpora,
            start_t=uem_start, lookback_seconds=blind_lookback,
            ratio=blind_ratio, blend=blind_blend, aggregate=blind_aggregate,
        )
        shabad_id = sid_result.shabad_id
        sid_correct = "✓" if shabad_id == gt_shabad_id else "✗"
        print(f"  {gt_path.stem}: blind ID predicts {shabad_id} (GT {gt_shabad_id}) {sid_correct} "
              f"top={sid_result.score:.1f} runner_up={sid_result.runner_up_score:.1f}")
    else:
        shabad_id = gt_shabad_id

    if shabad_id not in corpora:
        print(f"  error: predicted shabad {shabad_id} not in corpora", file=sys.stderr)
        return False
    lines = corpora[shabad_id]

    tfidf = None
    if blend and "tfidf" in blend:
        tfidf = TfidfScorer([normalize(l.get("transliteration_english", "")) for l in lines])

    pre_commit_segments: list[dict] = []
    if live:
        # Causal: matcher only sees chunks that start at or after the commit time.
        # Frames before commit are emitted as `null` UNLESS `tentative_emit` is set,
        # in which case each pre-commit chunk emits a global-best (shabad, line) pair.
        if tentative_emit and blind:
            pre_chunks = [c for c in chunks
                          if c.start >= uem_start and c.start < commit_time]
            matches = per_chunk_global_match(pre_chunks, corpora, ratio=ratio, blend=blend)
            cur_sid: int | None = None
            cur_lidx: int | None = None
            cur_start: float = 0.0
            cur_end: float = 0.0
            for c, (sid, lidx, _) in zip(pre_chunks, matches):
                if sid is None or lidx is None:
                    continue
                if (sid, lidx) == (cur_sid, cur_lidx):
                    cur_end = c.end
                else:
                    if cur_sid is not None and cur_lidx is not None:
                        ln = next(l for l in corpora[cur_sid] if l["line_idx"] == cur_lidx)
                        pre_commit_segments.append({
                            "start": cur_start, "end": cur_end,
                            "line_idx": cur_lidx, "shabad_id": cur_sid,
                            "verse_id": ln["verse_id"],
                            "banidb_gurmukhi": ln["banidb_gurmukhi"],
                        })
                    cur_sid, cur_lidx = sid, lidx
                    cur_start, cur_end = c.start, c.end
            if cur_sid is not None and cur_lidx is not None:
                ln = next(l for l in corpora[cur_sid] if l["line_idx"] == cur_lidx)
                pre_commit_segments.append({
                    "start": cur_start, "end": cur_end,
                    "line_idx": cur_lidx, "shabad_id": cur_sid,
                    "verse_id": ln["verse_id"],
                    "banidb_gurmukhi": ln["banidb_gurmukhi"],
                })
        chunks = [c for c in chunks if c.start >= commit_time]

    if stay_bias > 0.0:
        scored = [
            (c.start, c.end,
             score_chunk(c.text, lines, ratio=ratio, blend=blend, tfidf=tfidf))
            for c in chunks
        ]
        segments = smooth_with_stay_bias(scored, stay_margin=stay_bias,
                                         score_threshold=score_threshold)
    else:
        raw = [
            (c.start, c.end,
             match_chunk(c.text, lines,
                         score_threshold=score_threshold,
                         margin_threshold=margin_threshold,
                         ratio=ratio,
                         blend=blend,
                         tfidf=tfidf).line_idx)
            for c in chunks
        ]
        segments = smooth(raw)

    by_idx = {l["line_idx"]: l for l in lines}
    submission_segments = [
        {
            "start": s.start,
            "end": s.end,
            "line_idx": s.line_idx,
            "shabad_id": shabad_id,
            "verse_id": by_idx[s.line_idx]["verse_id"],
            "banidb_gurmukhi": by_idx[s.line_idx]["banidb_gurmukhi"],
        }
        for s in segments
    ]

    submission_segments = pre_commit_segments + submission_segments

    out_path = out_dir / gt_path.name
    out_path.write_text(json.dumps(
        {"video_id": video_id, "segments": submission_segments},
        ensure_ascii=False, indent=2,
    ))
    print(f"  {gt_path.stem}: {len(chunks)} chunks → {len(segments)} segments")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=pathlib.Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--audio-dir", type=pathlib.Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--corpus-dir", type=pathlib.Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--asr-cache-dir", type=pathlib.Path, default=DEFAULT_ASR_CACHE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--backend", default="faster_whisper",
                        choices=["faster_whisper", "mlx_whisper"],
                        help="ASR backend; default reproduces v3.2 (Path A canonical)")
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
    args = parser.parse_args()
    def _parse_blend(spec: str) -> dict[str, float] | None:
        if not spec.strip():
            return None
        out: dict[str, float] = {}
        for part in spec.split(","):
            name, w = part.split(":")
            out[name.strip()] = float(w)
        return out
    blend = _parse_blend(args.blend)
    blind_blend = _parse_blend(args.blind_blend)

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

    label = f"blend={blend}" if blend else f"ratio={args.ratio}"
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
            asr_cache_dir=args.asr_cache_dir.resolve(),
            out_dir=args.out_dir.resolve(),
            model_size=args.model,
            score_threshold=args.threshold,
            margin_threshold=args.margin,
            ratio=args.ratio,
            blend=blend,
            stay_bias=args.stay_bias,
            blind=args.blind,
            blind_lookback=args.blind_lookback,
            blind_aggregate=args.blind_aggregate,
            blind_ratio=args.blind_ratio,
            blind_blend=blind_blend,
            live=args.live,
            tentative_emit=args.tentative_emit,
            backend=args.backend,
        ):
            failures.append(gt_file.stem)

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1
    print(f"\nwrote {len(gt_files)} submissions to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
