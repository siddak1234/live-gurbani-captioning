#!/usr/bin/env python3
"""Stage 2 smoke runner for Path B (CTC + per-line acoustic scoring, no HMM yet).

For each GT case:
  1. Encode the audio with MMS-1B (Punjabi adapter) → per-frame log-probs.
  2. Tile audio into fixed-size windows.
  3. For each window: tokenize every corpus line, compute CTC log P(window | line),
     pick the line with the highest log P.
  4. Smooth consecutive same-line windows into segments.
  5. Emit submission JSON.

This is the simplest end-to-end Path B baseline — no HMM, no temporal smoothing
beyond consecutive merging. Goal: prove the CTC scoring approach beats Path A
fuzzy matching in oracle/offline.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.path_b.ctc_scorer import ctc_log_prob          # noqa: E402
from src.path_b.encoder import encode_file              # noqa: E402
from src.path_b.tokenizer import tokenize_line          # noqa: E402
from src.smoother import smooth                          # noqa: E402

DEFAULT_GT_DIR = REPO_ROOT.parent / "live-gurbani-captioning-benchmark-v1" / "test"
DEFAULT_AUDIO_DIR = REPO_ROOT / "audio"
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus_cache"
DEFAULT_MMS_CACHE = REPO_ROOT / "mms_cache"
DEFAULT_OUT_DIR = REPO_ROOT / "submissions" / "pb_smoke"


def process_one(
    gt_path: pathlib.Path,
    *,
    audio_dir: pathlib.Path,
    corpora: dict[int, list[dict]],
    mms_cache_dir: pathlib.Path,
    out_dir: pathlib.Path,
    window_seconds: float,
    hop_seconds: float,
) -> bool:
    gt = json.loads(gt_path.read_text())
    video_id = gt["video_id"]
    shabad_id = gt["shabad_id"]

    audio_path = audio_dir / f"{video_id}_16k.wav"
    if not audio_path.exists():
        print(f"  error: audio missing {audio_path}", file=sys.stderr)
        return False
    if shabad_id not in corpora:
        print(f"  error: corpus missing {shabad_id}", file=sys.stderr)
        return False

    t0 = time.time()
    ctc = encode_file(audio_path, cache_dir=mms_cache_dir)
    t_encode = time.time() - t0

    # Skip line 0 (shabad title) — matches Path A's convention.
    lines = [l for l in corpora[shabad_id] if l["line_idx"] != 0]
    line_tokens: list[tuple[int, list[int]]] = []
    for line in lines:
        toks = tokenize_line(line.get("banidb_gurmukhi", ""), ctc.vocab)
        line_tokens.append((int(line["line_idx"]), toks))

    T = ctc.log_probs.shape[0]
    frames_per_window = max(1, int(round(window_seconds / ctc.frame_duration)))
    hop_frames = max(1, int(round(hop_seconds / ctc.frame_duration)))

    t0 = time.time()
    raw: list[tuple[float, float, int | None]] = []
    for start_frame in range(0, T, hop_frames):
        end_frame = min(T, start_frame + frames_per_window)
        window = ctc.log_probs[start_frame:end_frame]
        if window.shape[0] == 0:
            continue
        best_idx: int | None = None
        best_score = float("-inf")
        for line_idx, toks in line_tokens:
            if not toks:
                continue
            score = ctc_log_prob(window, toks, ctc.blank_id)
            if score > best_score:
                best_score = score
                best_idx = line_idx
        start_s = start_frame * ctc.frame_duration
        end_s = end_frame * ctc.frame_duration
        raw.append((start_s, end_s, best_idx))
    t_score = time.time() - t0

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

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / gt_path.name
    out_path.write_text(json.dumps(
        {"video_id": video_id, "segments": submission_segments},
        ensure_ascii=False, indent=2,
    ))
    print(f"  {gt_path.stem}: encode={t_encode:.1f}s score={t_score:.1f}s "
          f"frames={T} windows={len(raw)} → {len(segments)} segments")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=pathlib.Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--audio-dir", type=pathlib.Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--corpus-dir", type=pathlib.Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--mms-cache-dir", type=pathlib.Path, default=DEFAULT_MMS_CACHE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window", type=float, default=2.0,
                        help="Audio window size in seconds (default 2.0)")
    parser.add_argument("--hop", type=float, default=1.0,
                        help="Hop size in seconds (default 1.0; 50%% overlap with 2s window)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N GT cases (0 = all)")
    args = parser.parse_args()

    corpus_dir = args.corpus_dir.resolve()
    corpora: dict[int, list[dict]] = {}
    for cf in sorted(corpus_dir.glob("*.json")):
        c = json.loads(cf.read_text())
        corpora[int(c["shabad_id"])] = c["lines"]
    if not corpora:
        print(f"error: no corpus files in {corpus_dir}", file=sys.stderr)
        return 1

    gt_files = sorted(args.gt_dir.glob("*.json"))
    if args.limit > 0:
        gt_files = gt_files[: args.limit]

    print(f"processing {len(gt_files)} GT cases "
          f"(window={args.window}s, hop={args.hop}s)\n")
    failures: list[str] = []
    for gt_file in gt_files:
        if not process_one(
            gt_file,
            audio_dir=args.audio_dir.resolve(),
            corpora=corpora,
            mms_cache_dir=args.mms_cache_dir.resolve(),
            out_dir=args.out_dir.resolve(),
            window_seconds=args.window,
            hop_seconds=args.hop,
        ):
            failures.append(gt_file.stem)

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1
    print(f"\nwrote {len(gt_files)} submissions to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
