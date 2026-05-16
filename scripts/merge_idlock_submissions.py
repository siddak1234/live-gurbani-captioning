#!/usr/bin/env python3
"""Merge two submission directories around a shabad-ID lock time.

This is a diagnostic helper for Phase 2.6. It answers:

    If a conservative engine owns the blind-ID buffer, and a second engine
    owns line tracking after shabad lock, what score would that integration
    shape get?

It is deliberately a Layer-3 submission transformer, not an inference engine.
No ASR, matcher, smoother, or model code runs here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _clip_segment(segment: dict[str, Any], start: float | None, end: float | None) -> dict[str, Any] | None:
    clipped = dict(segment)
    if start is not None:
        clipped["start"] = max(float(clipped["start"]), start)
    if end is not None:
        clipped["end"] = min(float(clipped["end"]), end)
    if float(clipped["end"]) <= float(clipped["start"]):
        return None
    return clipped


def merge_at_commit(
    pre_segments: list[dict[str, Any]],
    post_segments: list[dict[str, Any]],
    *,
    commit_time: float,
) -> list[dict[str, Any]]:
    """Keep ``pre`` before commit and ``post`` after commit, truncating overlaps."""
    merged: list[dict[str, Any]] = []
    for segment in pre_segments:
        clipped = _clip_segment(segment, start=None, end=commit_time)
        if clipped is not None:
            merged.append(clipped)
    for segment in post_segments:
        clipped = _clip_segment(segment, start=commit_time, end=None)
        if clipped is not None:
            merged.append(clipped)
    merged.sort(key=lambda s: (float(s["start"]), float(s["end"])))
    return merged


def merge_one(gt_path: Path, pre_dir: Path, post_dir: Path, out_dir: Path, lookback_s: float) -> None:
    gt = _load_json(gt_path)
    commit_time = float(gt.get("uem", {}).get("start", 0.0)) + lookback_s

    pre_doc = _load_json(pre_dir / gt_path.name)
    post_doc = _load_json(post_dir / gt_path.name)
    if pre_doc.get("video_id") != post_doc.get("video_id"):
        raise ValueError(f"video_id mismatch for {gt_path.name}: {pre_doc.get('video_id')} != {post_doc.get('video_id')}")

    merged = merge_at_commit(
        pre_doc.get("segments", []),
        post_doc.get("segments", []),
        commit_time=commit_time,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / gt_path.name).write_text(json.dumps(
        {"video_id": gt["video_id"], "segments": merged},
        ensure_ascii=False,
        indent=2,
    ))
    print(f"{gt_path.stem}: commit={commit_time:.1f}s pre={len(pre_doc.get('segments', []))} "
          f"post={len(post_doc.get('segments', []))} merged={len(merged)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--pre-dir", type=Path, required=True,
                        help="Submission used before shabad-ID lock")
    parser.add_argument("--post-dir", type=Path, required=True,
                        help="Submission used after shabad-ID lock")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lookback-seconds", type=float, default=30.0)
    args = parser.parse_args()

    gt_files = sorted(args.gt_dir.glob("*.json"))
    if not gt_files:
        raise SystemExit(f"no GT JSON files in {args.gt_dir}")

    for gt_path in gt_files:
        merge_one(
            gt_path,
            args.pre_dir,
            args.post_dir,
            args.out_dir,
            args.lookback_seconds,
        )
    print(f"\nwrote {len(gt_files)} merged submissions to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
