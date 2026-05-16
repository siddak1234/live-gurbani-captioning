#!/usr/bin/env python3
"""Debug CLI for transcribing one audio file with a chosen ASR backend.

This is intentionally outside ``src/`` so Layer 1 inference primitives stay
library-only. Production and benchmark inference still go through
``src.engine.predict()``; this script is for quick local ASR inspection.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.asr import transcribe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=pathlib.Path)
    parser.add_argument(
        "--backend",
        default="faster_whisper",
        choices=["faster_whisper", "mlx_whisper", "huggingface_whisper"],
    )
    parser.add_argument("--model", default="medium")
    parser.add_argument("--language", default="pa")
    parser.add_argument("--cache-dir", type=pathlib.Path, default=pathlib.Path("asr_cache"))
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--vad-filter", action="store_true",
                        help="Enable faster-whisper VAD filtering. Default is off.")
    parser.add_argument("--no-speech-threshold", type=float, default=None)
    parser.add_argument("--adapter-dir", default=None)
    args = parser.parse_args()

    chunks = transcribe(
        args.audio_path,
        backend=args.backend,
        model_size=args.model,
        language=args.language,
        cache_dir=args.cache_dir,
        word_timestamps=args.word_timestamps,
        vad_filter=args.vad_filter,
        no_speech_threshold=args.no_speech_threshold,
        adapter_dir=args.adapter_dir,
    )
    print(f"{len(chunks)} chunks ({args.backend}, {args.model}):\n")
    for chunk in chunks:
        print(f"  [{chunk.start:7.2f}-{chunk.end:7.2f}] {chunk.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
