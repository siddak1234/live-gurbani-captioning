#!/usr/bin/env python3
"""Phase X1: vocal source separation with Demucs.

For each audio file under `audio/`, runs Demucs's `htdemucs` model in vocals-only
mode and writes the isolated vocal stem as a 16kHz mono WAV to `audio_vocals/`.
Path A and Path B can then transcribe the cleaner signal (no tabla, harmonium,
sangat in the background).

Idempotent: skips already-separated files.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_DIR = REPO_ROOT / "audio"
DEFAULT_OUT_DIR = REPO_ROOT / "audio_vocals"
DEFAULT_MODEL = "htdemucs"


def separate_one(audio_path: pathlib.Path, out_dir: pathlib.Path, model: str, device: str) -> bool:
    # Use original filename so downstream runners can switch input dirs with no other change.
    out_path = out_dir / audio_path.name
    if out_path.exists():
        print(f"skip: {out_path.name} already exists")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_demucs_work"
    work_dir.mkdir(exist_ok=True)

    print(f"separating {audio_path.name} (device={device})...")
    try:
        subprocess.run(
            [
                "demucs", "--two-stems", "vocals",
                "-n", model,
                "-d", device,
                "-o", str(work_dir),
                str(audio_path),
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"error: demucs failed on {audio_path.name}:\n{e.stderr}", file=sys.stderr)
        return False

    # Demucs writes to <work_dir>/<model>/<stem>/vocals.wav at 44.1kHz stereo
    stem_dir = work_dir / model / audio_path.stem
    src_vocals = stem_dir / "vocals.wav"
    if not src_vocals.exists():
        print(f"error: expected {src_vocals} not produced", file=sys.stderr)
        return False

    print(f"  resampling → 16kHz mono")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src_vocals),
                "-ar", "16000", "-ac", "1", str(out_path),
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"error: ffmpeg failed:\n{e.stderr}", file=sys.stderr)
        return False

    # Clean up the demucs intermediate so the disk doesn't bloat
    shutil.rmtree(stem_dir, ignore_errors=True)
    print(f"  done: {out_path.relative_to(REPO_ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=pathlib.Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Demucs model name (default: htdemucs)")
    parser.add_argument("--device", default="mps",
                        help="torch device for Demucs: mps (Apple GPU), cuda, or cpu")
    args = parser.parse_args()

    if not shutil.which("demucs"):
        print("error: demucs not on PATH (pip install demucs)", file=sys.stderr)
        return 1
    if not shutil.which("ffmpeg"):
        print("error: ffmpeg not on PATH", file=sys.stderr)
        return 1

    audio_dir = args.audio_dir.resolve()
    audio_files = sorted(audio_dir.glob("*.wav"))
    if not audio_files:
        print(f"error: no .wav files in {audio_dir}", file=sys.stderr)
        return 1

    out_dir = args.out_dir.resolve()
    print(f"separating {len(audio_files)} file(s) with {args.model} on {args.device}\n")
    failures = [f.name for f in audio_files if not separate_one(f, out_dir, args.model, args.device)]

    # Tidy up work dir if empty
    work_dir = out_dir / "_demucs_work"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1
    print(f"\nall {len(audio_files)} stems written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
