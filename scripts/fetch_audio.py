#!/usr/bin/env python3
"""Stage 1a: download benchmark audio.

Reads unique `video_id`s from the paired benchmark's test/*.json, downloads
each via yt-dlp, and converts to 16kHz mono WAV under audio/. Skips files
that already exist so re-runs are cheap.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_GT_DIR = REPO_ROOT.parent / "live-gurbani-captioning-benchmark-v1" / "test"
DEFAULT_AUDIO_DIR = REPO_ROOT / "audio"


def collect_video_ids(gt_dir: pathlib.Path) -> list[str]:
    ids: set[str] = set()
    for f in sorted(gt_dir.glob("*.json")):
        ids.add(json.loads(f.read_text())["video_id"])
    return sorted(ids)


def fetch_one(video_id: str, audio_dir: pathlib.Path) -> bool:
    out_path = audio_dir / f"{video_id}_16k.wav"
    if out_path.exists():
        print(f"skip: {out_path.name} already exists")
        return True

    audio_dir.mkdir(parents=True, exist_ok=True)
    raw_path = audio_dir / f"{video_id}.wav"
    url = f"https://youtube.com/watch?v={video_id}"

    print(f"downloading {video_id}...")
    try:
        subprocess.run(
            [
                "yt-dlp", "-x", "--audio-format", "wav",
                "-o", str(audio_dir / "%(id)s.%(ext)s"),
                url,
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"error downloading {video_id}:\n{e.stderr}", file=sys.stderr)
        return False

    if not raw_path.exists():
        print(f"error: expected {raw_path} but yt-dlp produced no such file", file=sys.stderr)
        return False

    print(f"converting {video_id} → 16kHz mono...")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(raw_path),
                "-ar", "16000", "-ac", "1", str(out_path),
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"error converting {video_id}:\n{e.stderr}", file=sys.stderr)
        return False

    raw_path.unlink()
    print(f"done: {out_path.relative_to(REPO_ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=pathlib.Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--audio-dir", type=pathlib.Path, default=DEFAULT_AUDIO_DIR)
    args = parser.parse_args()

    for tool in ("yt-dlp", "ffmpeg"):
        if not shutil.which(tool):
            print(f"error: {tool} not on PATH (install via brew or pip)", file=sys.stderr)
            return 1

    video_ids = collect_video_ids(args.gt_dir.resolve())
    if not video_ids:
        print(f"error: no GT files in {args.gt_dir}", file=sys.stderr)
        return 1

    print(f"found {len(video_ids)} unique video_id(s): {', '.join(video_ids)}\n")
    audio_dir = args.audio_dir.resolve()
    failures = [v for v in video_ids if not fetch_one(v, audio_dir)]

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1
    print(f"\nall {len(video_ids)} audio file(s) ready in {audio_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
