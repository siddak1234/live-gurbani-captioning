#!/usr/bin/env python3
"""Dump expected transcripts from the HF reference model for the
M5.7d numerical-parity test fixture.

Given an audio file + an HF model id, runs greedy decoding through
the `transformers` ASR pipeline and writes a small JSON file that
the Swift ``ModelParityTests`` reads on the other side of the wire.

Why a separate script rather than computing the expected at test
time: the HF Python pipeline lives in the venv at ``.venv/``, not in
the iOS toolchain. Swift tests can't shell out to Python during
``swift test`` without making the test target's environment fragile.
Pre-computing the reference once + committing it as JSON keeps the
Swift side hermetic.

Usage:

    .venv/bin/python scripts/dump_parity_fixture.py \\
        --audio ios/Tests/GurbaniCaptioningTests/Fixtures/izos_30s_snippet_5s.wav \\
        --model surindersinghssj/surt-small-v3 \\
        --language punjabi \\
        --out ios/Tests/GurbaniCaptioningTests/Fixtures/izos_30s_snippet_5s.expected.json

Re-run when:
  - the bundled Core ML model is regenerated against a different
    HF base (e.g., a new LoRA-merged checkpoint),
  - the audio fixture changes,
  - the language tag or decoder options change.

The script is deterministic so re-running with the same inputs
produces byte-identical JSON.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def _load_audio(path: pathlib.Path) -> "tuple[list[float], int]":
    """Load the WAV file as a mono float32 array at its native rate.
    No resampling — caller must pass 16 kHz mono (matches Whisper's
    pre-processor)."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim != 1:
        # Convert to mono by averaging channels.
        audio = audio.mean(axis=1)
    return audio.tolist(), int(sr)


def _run_hf_pipeline(audio: list, sr: int, model_id: str, language: str) -> str:
    """Greedy-decode the audio through HF's ASR pipeline. Returns the
    raw transcribed text (no special tokens)."""
    from transformers import pipeline
    import numpy as np

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        # `chunk_length_s=30` matches Whisper's native window. Our
        # fixture is shorter so it stays in one window.
        chunk_length_s=30,
    )
    result = asr(
        np.asarray(audio, dtype="float32"),
        generate_kwargs={
            "language": language,
            "task": "transcribe",
            # Greedy decoding — deterministic, single beam.
            "num_beams": 1,
            "do_sample": False,
            # Match what `DecodingOptions(skipSpecialTokens: true,
            # temperature: 0.0)` does on the Swift side.
            "temperature": 0.0,
        },
    )
    return result["text"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--audio", type=pathlib.Path, required=True,
                        help="Input WAV (expected 16 kHz mono).")
    parser.add_argument("--model", required=True,
                        help="HF model id; e.g. surindersinghssj/surt-small-v3")
    parser.add_argument("--language", default="punjabi",
                        help="Whisper generate language tag (default: punjabi)")
    parser.add_argument("--out", type=pathlib.Path, required=True,
                        help="Output JSON path. Sibling of the audio is conventional.")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"error: audio file not found: {args.audio}", file=sys.stderr)
        return 1

    print(f"Loading audio: {args.audio}")
    audio, sr = _load_audio(args.audio)
    duration = len(audio) / sr
    print(f"  duration: {duration:.2f}s @ {sr} Hz, {len(audio)} samples")

    print(f"Running HF reference: {args.model}")
    text = _run_hf_pipeline(audio, sr, args.model, args.language)
    print(f"  transcript: {text!r}")

    payload = {
        "schema_version": 1,
        "audio_filename": args.audio.name,
        "duration_seconds": round(duration, 4),
        "sample_rate_hz": sr,
        "model_id": args.model,
        "language": args.language,
        "decoder": {
            "num_beams": 1,
            "do_sample": False,
            "temperature": 0.0,
        },
        "expected_text": text,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\n✓ Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
