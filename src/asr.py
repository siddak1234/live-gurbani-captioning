#!/usr/bin/env python3
"""ASR wrapper supporting two backends.

  - `faster_whisper` (default): runs Whisper via CTranslate2. CPU on Mac, can use
    Nvidia GPU elsewhere. Bundles Silero VAD as a pre-filter, which produces the
    chunk granularity Path A's matcher and smoother are tuned for. **This is the
    backend that produced v3.2 (86.5%); it's the canonical Path A backend.**

  - `mlx_whisper`: runs Whisper via Apple's mlx framework on Apple Silicon's
    Neural Engine + GPU. Much faster on Mac. Produces longer, time-aligned
    chunks (no Silero VAD) — Path A's downstream pipeline doesn't fit it without
    retuning. Kept available as a flag for experiments and Path B.

Cache keys differ by backend so transcripts don't collide. The fw cache uses
the legacy filename convention (no backend prefix) so existing cached files
keep working.

Standalone use:

    python src/asr.py audio/IZOsmkdmmcg_16k.wav                       # fw default
    python src/asr.py audio/IZOsmkdmmcg_16k.wav --backend mlx_whisper --model large-v3
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict, dataclass

# mlx-whisper short-name → HF repo lookup. Pass-through if the user supplies a full repo path.
MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


@dataclass
class AsrChunk:
    start: float
    end: float
    text: str


def _cache_path(
    audio_path: pathlib.Path,
    cache_dir: pathlib.Path | None,
    backend: str,
    model_size: str,
    language: str,
    extra_tag: str,
) -> pathlib.Path | None:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    if backend == "faster_whisper":
        # Legacy convention — no backend prefix, matches existing cached files.
        return cache_dir / f"{audio_path.stem}__{model_size}{extra_tag}__{language}.json"
    if backend == "mlx_whisper":
        return cache_dir / f"{audio_path.stem}__mlx-{model_size}{extra_tag}__{language}.json"
    raise ValueError(f"unknown backend: {backend}")


def transcribe(
    audio_path: pathlib.Path | str,
    *,
    backend: str = "faster_whisper",
    model_size: str = "medium",
    language: str = "pa",
    cache_dir: pathlib.Path | str | None = None,
    word_timestamps: bool = False,
    no_speech_threshold: float | None = None,
) -> list[AsrChunk]:
    """Transcribe audio with the selected backend, returning timestamped chunks."""
    audio_path = pathlib.Path(audio_path)
    if cache_dir is not None:
        cache_dir = pathlib.Path(cache_dir)

    extra_tag = ""
    if word_timestamps:
        extra_tag += "_word"
    if no_speech_threshold is not None:
        extra_tag += f"_nst{no_speech_threshold}"

    cache_path = _cache_path(audio_path, cache_dir, backend, model_size, language, extra_tag)
    if cache_path is not None and cache_path.exists():
        data = json.loads(cache_path.read_text())
        return [AsrChunk(**c) for c in data]

    if backend == "faster_whisper":
        chunks = _transcribe_fw(audio_path, model_size, language, word_timestamps, no_speech_threshold)
    elif backend == "mlx_whisper":
        chunks = _transcribe_mlx(audio_path, model_size, language, word_timestamps, no_speech_threshold)
    else:
        raise ValueError(f"unknown backend: {backend}")

    if cache_path is not None:
        cache_path.write_text(
            json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2)
        )
    return chunks


def _transcribe_fw(audio_path, model_size, language, word_timestamps, no_speech_threshold):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, compute_type="int8")
    kwargs: dict = {
        "language": language,
        "beam_size": 5,
        "word_timestamps": word_timestamps,
    }
    if no_speech_threshold is not None:
        kwargs["no_speech_threshold"] = no_speech_threshold
    segments, _info = model.transcribe(str(audio_path), **kwargs)
    return [
        AsrChunk(start=float(s.start), end=float(s.end), text=s.text.strip())
        for s in segments
    ]


def _transcribe_mlx(audio_path, model_size, language, word_timestamps, no_speech_threshold):
    import mlx_whisper

    repo = MLX_REPOS.get(model_size, model_size)
    kwargs: dict = {
        "path_or_hf_repo": repo,
        "language": language,
        "word_timestamps": word_timestamps,
    }
    if no_speech_threshold is not None:
        kwargs["no_speech_threshold"] = no_speech_threshold
    result = mlx_whisper.transcribe(str(audio_path), **kwargs)
    return [
        AsrChunk(
            start=float(s["start"]),
            end=float(s["end"]),
            text=str(s["text"]).strip(),
        )
        for s in result.get("segments", [])
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=pathlib.Path)
    parser.add_argument("--backend", default="faster_whisper",
                        choices=["faster_whisper", "mlx_whisper"])
    parser.add_argument("--model", default="medium")
    parser.add_argument("--language", default="pa")
    parser.add_argument("--cache-dir", type=pathlib.Path, default=pathlib.Path("asr_cache"))
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--no-speech-threshold", type=float, default=None)
    args = parser.parse_args()

    chunks = transcribe(
        args.audio_path,
        backend=args.backend,
        model_size=args.model,
        language=args.language,
        cache_dir=args.cache_dir,
        word_timestamps=args.word_timestamps,
        no_speech_threshold=args.no_speech_threshold,
    )
    print(f"{len(chunks)} chunks ({args.backend}, {args.model}):\n")
    for c in chunks:
        print(f"  [{c.start:7.2f}-{c.end:7.2f}] {c.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
