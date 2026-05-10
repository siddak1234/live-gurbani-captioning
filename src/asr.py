#!/usr/bin/env python3
"""ASR wrapper around faster-whisper for kirtan audio.

Returns timestamped chunks: list of (start, end, text). Caches transcripts to
disk so re-runs don't re-transcribe (the medium model takes minutes per file).

Standalone use:

    python src/asr.py audio/IZOsmkdmmcg_16k.wav
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict, dataclass


@dataclass
class AsrChunk:
    start: float
    end: float
    text: str


def transcribe(
    audio_path: pathlib.Path | str,
    *,
    model_size: str = "medium",
    language: str = "pa",
    cache_dir: pathlib.Path | str | None = None,
) -> list[AsrChunk]:
    """Transcribe audio with faster-whisper, returning timestamped chunks.

    Caches output keyed on (audio_stem, model_size, language). Subsequent
    calls with the same key are loaded from disk.
    """
    audio_path = pathlib.Path(audio_path)
    cache_path: pathlib.Path | None = None
    if cache_dir is not None:
        cache_dir = pathlib.Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{audio_path.stem}__{model_size}__{language}.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            return [AsrChunk(**c) for c in data]

    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, compute_type="int8")
    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        word_timestamps=False,
    )
    chunks = [
        AsrChunk(start=float(s.start), end=float(s.end), text=s.text.strip())
        for s in segments
    ]

    if cache_path is not None:
        cache_path.write_text(
            json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2)
        )
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=pathlib.Path)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--language", default="pa")
    parser.add_argument("--cache-dir", type=pathlib.Path, default=pathlib.Path("asr_cache"))
    args = parser.parse_args()

    chunks = transcribe(
        args.audio_path,
        model_size=args.model,
        language=args.language,
        cache_dir=args.cache_dir,
    )
    print(f"{len(chunks)} chunks:\n")
    for c in chunks:
        print(f"  [{c.start:7.2f}-{c.end:7.2f}] {c.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
