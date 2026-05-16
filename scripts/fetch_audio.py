#!/usr/bin/env python3
"""Stage 1a: download benchmark or OOS audio.

Reads unique `video_id`s from the paired benchmark's test/*.json, downloads
each via yt-dlp, and converts to 16kHz mono WAV under audio/. Skips files
that already exist so re-runs are cheap.

For Phase 2.5 / OOS curation, pass one or more `--url case_id=URL` specs
and set `--audio-dir eval_data/oos_v1/audio`. Those produce
`<case_id>_16k.wav` files matching the `eval_oos.py` directory contract.
Optional `--clip case_id=START-END` trims long source recordings to a bounded
curation window before resampling.
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


def parse_url_specs(specs: list[str]) -> list[tuple[str, str]]:
    """Parse repeated `case_id=URL` CLI specs for OOS audio.

    URLs themselves often contain `=` query params, so split only once.
    """
    targets: list[tuple[str, str]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--url must be case_id=URL, got: {spec!r}")
        case_id, url = spec.split("=", 1)
        case_id = case_id.strip()
        url = url.strip()
        if not case_id:
            raise ValueError(f"--url case_id is empty in: {spec!r}")
        if not url:
            raise ValueError(f"--url URL is empty in: {spec!r}")
        if "/" in case_id or "\\" in case_id:
            raise ValueError(f"--url case_id must be a filename stem, got: {case_id!r}")
        targets.append((case_id, url))
    return targets


def parse_clip_specs(specs: list[str]) -> dict[str, tuple[float, float]]:
    """Parse repeated `case_id=START-END` clip windows in seconds."""
    clips: dict[str, tuple[float, float]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--clip must be case_id=START-END, got: {spec!r}")
        case_id, window = spec.split("=", 1)
        case_id = case_id.strip()
        window = window.strip()
        if not case_id:
            raise ValueError(f"--clip case_id is empty in: {spec!r}")
        if "/" in case_id or "\\" in case_id:
            raise ValueError(f"--clip case_id must be a filename stem, got: {case_id!r}")
        if "-" not in window:
            raise ValueError(f"--clip window must be START-END seconds, got: {window!r}")
        start_s, end_s = window.split("-", 1)
        try:
            start = float(start_s)
            end = float(end_s)
        except ValueError as e:
            raise ValueError(f"--clip window must be numeric seconds, got: {window!r}") from e
        if start < 0:
            raise ValueError(f"--clip start must be >= 0, got: {start}")
        if end <= start:
            raise ValueError(f"--clip end must be greater than start, got: {window!r}")
        clips[case_id] = (start, end)
    return clips


def fetch_one_url(case_id: str, url: str, audio_dir: pathlib.Path,
                  clip: tuple[float, float] | None = None) -> bool:
    out_path = audio_dir / f"{case_id}_16k.wav"
    if out_path.exists():
        print(f"skip: {out_path.name} already exists")
        return True

    audio_dir.mkdir(parents=True, exist_ok=True)
    raw_template = audio_dir / f".{case_id}.%(ext)s"
    raw_path = audio_dir / f".{case_id}.wav"

    print(f"downloading {case_id}...")
    try:
        subprocess.run(
            [
                "yt-dlp", "-x", "--audio-format", "wav",
                "-o", str(raw_template),
                url,
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"error downloading {case_id}:\n{e.stderr}", file=sys.stderr)
        return False

    if not raw_path.exists():
        print(f"error: expected {raw_path} but yt-dlp produced no such file", file=sys.stderr)
        return False

    if clip:
        start, end = clip
        duration = end - start
        clip_args = ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}"]
        print(f"converting {case_id} -> 16kHz mono ({start:.1f}s-{end:.1f}s clip)...")
    else:
        clip_args = []
        print(f"converting {case_id} -> 16kHz mono...")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(raw_path), *clip_args,
                "-ar", "16000", "-ac", "1", str(out_path),
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"error converting {case_id}:\n{e.stderr}", file=sys.stderr)
        return False

    raw_path.unlink()
    try:
        display_path = out_path.relative_to(REPO_ROOT)
    except ValueError:
        display_path = out_path
    print(f"done: {display_path}")
    return True


def fetch_one(video_id: str, audio_dir: pathlib.Path) -> bool:
    return fetch_one_url(video_id, f"https://youtube.com/watch?v={video_id}", audio_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=pathlib.Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--audio-dir", type=pathlib.Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        metavar="CASE_ID=URL",
        help=(
            "Fetch an arbitrary source URL for OOS curation. Repeatable. "
            "Writes <CASE_ID>_16k.wav under --audio-dir."
        ),
    )
    parser.add_argument(
        "--clip",
        action="append",
        default=[],
        metavar="CASE_ID=START-END",
        help=(
            "Optional OOS clip window in seconds for a --url target. Repeatable. "
            "Example: --clip case_001=30-210"
        ),
    )
    args = parser.parse_args()

    for tool in ("yt-dlp", "ffmpeg"):
        if not shutil.which(tool):
            print(f"error: {tool} not on PATH (install via brew or pip)", file=sys.stderr)
            return 1

    audio_dir = args.audio_dir.resolve()
    if args.url:
        try:
            targets = parse_url_specs(args.url)
            clips = parse_clip_specs(args.clip)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        target_ids = {case_id for case_id, _ in targets}
        unknown_clip_ids = sorted(set(clips) - target_ids)
        if unknown_clip_ids:
            print(f"error: --clip case_id(s) without matching --url: {unknown_clip_ids}", file=sys.stderr)
            return 1
        print(f"found {len(targets)} OOS URL target(s): {', '.join(c for c, _ in targets)}\n")
        failures = [
            case_id for case_id, url in targets
            if not fetch_one_url(case_id, url, audio_dir, clips.get(case_id))
        ]
        total = len(targets)
    else:
        video_ids = collect_video_ids(args.gt_dir.resolve())
        if not video_ids:
            print(f"error: no GT files in {args.gt_dir}", file=sys.stderr)
            return 1

        print(f"found {len(video_ids)} unique video_id(s): {', '.join(video_ids)}\n")
        failures = [v for v in video_ids if not fetch_one(v, audio_dir)]
        total = len(video_ids)

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1
    print(f"\nall {total} audio file(s) ready in {audio_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
