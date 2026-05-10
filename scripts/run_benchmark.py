#!/usr/bin/env python3
"""Stage 0 runner: emit empty submissions for every GT case in the benchmark.

Reads each ground-truth JSON from the paired benchmark's `test/` directory
and writes a corresponding submission (with no segments) to the output
directory. Score the result with the benchmark's `eval.py`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_GT_DIR = REPO_ROOT.parent / "live-gurbani-captioning-benchmark-v1" / "test"
DEFAULT_OUT_DIR = REPO_ROOT / "submissions" / "v0_empty"


def predict_empty(gt: dict) -> list[dict]:
    return []


def run(gt_dir: pathlib.Path, out_dir: pathlib.Path) -> int:
    if not gt_dir.is_dir():
        print(f"error: GT directory not found: {gt_dir}", file=sys.stderr)
        return 1

    gt_files = sorted(gt_dir.glob("*.json"))
    if not gt_files:
        print(f"error: no GT JSON files in {gt_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    for gt_file in gt_files:
        gt = json.loads(gt_file.read_text())
        submission = {
            "video_id": gt["video_id"],
            "segments": predict_empty(gt),
        }
        out_path = out_dir / gt_file.name
        out_path.write_text(json.dumps(submission, indent=2, ensure_ascii=False))
        print(f"wrote {out_path.relative_to(REPO_ROOT)} ({len(submission['segments'])} segments)")

    print(f"\n{len(gt_files)} submission file(s) written to {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=pathlib.Path, default=DEFAULT_GT_DIR,
                        help=f"benchmark test/ directory (default: {DEFAULT_GT_DIR})")
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR,
                        help=f"submission output directory (default: {DEFAULT_OUT_DIR})")
    args = parser.parse_args()
    return run(args.gt_dir.resolve(), args.out_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
