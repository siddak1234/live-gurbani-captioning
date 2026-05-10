#!/usr/bin/env python3
"""Phase B3 runner: MMS CTC encoder + ShabadHmm line tracker (oracle, offline).

For each GT case:
  1. Encode audio with MMS-1B → per-frame log-probabilities.
  2. Build a ShabadHmm over the GT shabad's lines.
  3. Forward-decode causally → per-frame line index.
  4. Collapse consecutive same-line frames into segments.
  5. Emit submission JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.path_b.encoder import encode_file              # noqa: E402
from src.path_b.hmm import LineSpec, ShabadHmm          # noqa: E402
from src.path_b.tokenizer import tokenize_line          # noqa: E402

DEFAULT_GT_DIR = REPO_ROOT.parent / "live-gurbani-captioning-benchmark-v1" / "test"
DEFAULT_AUDIO_DIR = REPO_ROOT / "audio"
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus_cache"
DEFAULT_MMS_CACHE = REPO_ROOT / "mms_cache"
DEFAULT_OUT_DIR = REPO_ROOT / "submissions" / "pb1_hmm"


def frame_runs_to_segments(per_frame_line: list[int], frame_duration: float, lines_by_idx: dict) -> list[dict]:
    """Collapse consecutive same-line frames into segments."""
    segments: list[dict] = []
    cur_line: int | None = None
    cur_start = 0.0
    for i, line_idx in enumerate(per_frame_line):
        t = i * frame_duration
        if line_idx != cur_line:
            if cur_line is not None:
                segments.append(_make_segment(cur_start, t, cur_line, lines_by_idx))
            cur_line = int(line_idx)
            cur_start = t
    if cur_line is not None:
        end_t = len(per_frame_line) * frame_duration
        segments.append(_make_segment(cur_start, end_t, cur_line, lines_by_idx))
    return segments


def _make_segment(start: float, end: float, line_idx: int, lines_by_idx: dict) -> dict:
    line = lines_by_idx[line_idx]
    return {
        "start": float(start),
        "end": float(end),
        "line_idx": int(line_idx),
        "shabad_id": int(line.get("_shabad_id", 0)),
        "verse_id": line.get("verse_id"),
        "banidb_gurmukhi": line.get("banidb_gurmukhi", ""),
    }


def process_one(
    gt_path: pathlib.Path,
    *,
    audio_dir: pathlib.Path,
    corpora: dict[int, list[dict]],
    mms_cache_dir: pathlib.Path,
    out_dir: pathlib.Path,
    switch_log_prob: float,
    viterbi: bool,
    model_id: str,
    target_lang: str | None,
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
    ctc = encode_file(
        audio_path, model_id=model_id, target_lang=target_lang, cache_dir=mms_cache_dir,
    )
    t_encode = time.time() - t0

    lines_raw = [l for l in corpora[shabad_id] if l["line_idx"] != 0]
    lines: list[LineSpec] = []
    lines_by_idx: dict[int, dict] = {}
    for line in lines_raw:
        toks = tokenize_line(line.get("banidb_gurmukhi", ""), ctc.vocab)
        if not toks:
            continue
        lines.append(LineSpec(line_idx=int(line["line_idx"]), tokens=toks))
        # Stash shabad_id alongside for segment emission
        line_with_sid = dict(line)
        line_with_sid["_shabad_id"] = shabad_id
        lines_by_idx[int(line["line_idx"])] = line_with_sid

    t0 = time.time()
    hmm = ShabadHmm(lines, blank_id=ctc.blank_id, switch_log_prob=switch_log_prob)
    per_frame = hmm.decode(ctc.log_probs, viterbi=viterbi).tolist()
    t_decode = time.time() - t0

    segments = frame_runs_to_segments(per_frame, ctc.frame_duration, lines_by_idx)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / gt_path.name
    out_path.write_text(json.dumps(
        {"video_id": video_id, "segments": segments},
        ensure_ascii=False, indent=2,
    ))
    print(f"  {gt_path.stem}: encode={t_encode:.1f}s decode={t_decode:.1f}s "
          f"frames={ctc.log_probs.shape[0]} → {len(segments)} segments")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=pathlib.Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--audio-dir", type=pathlib.Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--corpus-dir", type=pathlib.Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--mms-cache-dir", type=pathlib.Path, default=DEFAULT_MMS_CACHE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--switch-log-prob", type=float, default=math.log(1e-4),
                        help="Cross-line transition log prob per frame (default log(1e-4) = -9.2)")
    parser.add_argument("--viterbi", action="store_true",
                        help="Use Viterbi (max-paths) instead of forward (sum-paths)")
    parser.add_argument("--model-id", default="facebook/mms-1b-all",
                        help="HF model ID for the CTC acoustic model")
    parser.add_argument("--target-lang", default="pan",
                        help='Language adapter for MMS models (ignored for others; "" to disable)')
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    target_lang = args.target_lang if args.target_lang else None

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
          f"(switch_log_prob={args.switch_log_prob:.2f})\n")
    failures: list[str] = []
    for gt_file in gt_files:
        if not process_one(
            gt_file,
            audio_dir=args.audio_dir.resolve(),
            corpora=corpora,
            mms_cache_dir=args.mms_cache_dir.resolve(),
            out_dir=args.out_dir.resolve(),
            switch_log_prob=args.switch_log_prob,
            viterbi=args.viterbi,
            model_id=args.model_id,
            target_lang=target_lang,
        ):
            failures.append(gt_file.stem)

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1
    print(f"\nwrote {len(gt_files)} submissions to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
