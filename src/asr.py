#!/usr/bin/env python3
"""ASR wrapper supporting two backends.

  - `faster_whisper` (default): runs Whisper via CTranslate2. CPU on Mac, can use
    Nvidia GPU elsewhere. The current known-good command uses faster-whisper's
    default ``vad_filter=False``; Phase 2.8 makes the VAD and timestamp knobs
    explicit so ASR drift is auditable. **This is the backend that produced the
    historical v3.2 artifact (86.5%); the current runtime repro is lower and is
    under Phase 2.8 investigation.**

  - `mlx_whisper`: runs Whisper via Apple's mlx framework on Apple Silicon's
    Neural Engine + GPU. Much faster on Mac. Produces longer, time-aligned
    chunks (no Silero VAD) — Path A's downstream pipeline doesn't fit it without
    retuning. Kept available as a flag for experiments and Path B.

Cache keys differ by backend so transcripts don't collide. The fw cache uses
the legacy filename convention (no backend prefix) so existing cached files
keep working.

Standalone debugging lives in ``scripts/transcribe_audio.py``. This module is
Layer 1 library code and intentionally has no CLI surface.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass

# mlx-whisper short-name → HF repo lookup. Pass-through if the user supplies a full repo path.
# Default windowing for huggingface_whisper backend. Override via env var
# HF_WINDOW_SECONDS to A/B without code changes.
import os as _os
_HF_WINDOW_SECONDS = float(_os.environ.get("HF_WINDOW_SECONDS", "30"))

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
    if backend == "huggingface_whisper":
        # Sanitize HF repo path for use as a filename: "user/repo" → "user_repo"
        sanitized = model_size.replace("/", "_")
        window_tag = f"_w{int(_HF_WINDOW_SECONDS)}"
        return cache_dir / f"{audio_path.stem}__hf-{sanitized}{window_tag}{extra_tag}__{language}.json"
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
    vad_filter: bool = False,
    adapter_dir: str | None = None,
) -> list[AsrChunk]:
    """Transcribe audio with the selected backend, returning timestamped chunks."""
    # HF repo paths like "surindersinghssj/surt-small-v3" force the HF backend,
    # since custom-fine-tuned models aren't natively packaged for fw / mlx.
    if "/" in model_size and backend in ("faster_whisper", "mlx_whisper"):
        backend = "huggingface_whisper"
    audio_path = pathlib.Path(audio_path)
    if cache_dir is not None:
        cache_dir = pathlib.Path(cache_dir)

    if vad_filter and backend != "faster_whisper":
        raise ValueError("vad_filter is currently supported only for faster_whisper")

    extra_tag = _extra_tag(
        word_timestamps=word_timestamps,
        no_speech_threshold=no_speech_threshold,
        vad_filter=vad_filter,
        adapter_dir=adapter_dir,
    )

    cache_path = _cache_path(audio_path, cache_dir, backend, model_size, language, extra_tag)
    if cache_path is not None and cache_path.exists():
        data = json.loads(cache_path.read_text())
        return [AsrChunk(**c) for c in data]

    if backend == "faster_whisper":
        chunks = _transcribe_fw(
            audio_path,
            model_size,
            language,
            word_timestamps,
            no_speech_threshold,
            vad_filter,
        )
    elif backend == "mlx_whisper":
        chunks = _transcribe_mlx(audio_path, model_size, language, word_timestamps, no_speech_threshold)
    elif backend == "huggingface_whisper":
        chunks = _transcribe_hf(audio_path, model_size, language,
                                window_seconds=_HF_WINDOW_SECONDS,
                                adapter_dir=adapter_dir)
    else:
        raise ValueError(f"unknown backend: {backend}")

    if cache_path is not None:
        cache_path.write_text(
            json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2)
        )
    return chunks


def _extra_tag(
    *,
    word_timestamps: bool = False,
    no_speech_threshold: float | None = None,
    vad_filter: bool = False,
    adapter_dir: str | None = None,
) -> str:
    """Cache-key suffix for non-default ASR knobs.

    Empty string preserves the legacy faster-whisper cache names, which now
    explicitly mean ``word_timestamps=False`` and ``vad_filter=False``.
    """
    extra_tag = ""
    if word_timestamps:
        extra_tag += "_word"
    if vad_filter:
        extra_tag += "_vad"
    if no_speech_threshold is not None:
        extra_tag += f"_nst{no_speech_threshold}"
    if adapter_dir:
        adapter_tag = pathlib.Path(adapter_dir).name.replace("/", "_")
        extra_tag += f"_lora-{adapter_tag}"
    return extra_tag


def _transcribe_fw(audio_path, model_size, language, word_timestamps, no_speech_threshold, vad_filter):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, compute_type="int8")
    kwargs: dict = {
        "language": language,
        "beam_size": 5,
        "word_timestamps": word_timestamps,
        "vad_filter": vad_filter,
    }
    if no_speech_threshold is not None:
        kwargs["no_speech_threshold"] = no_speech_threshold
    segments, _info = model.transcribe(str(audio_path), **kwargs)
    return [
        AsrChunk(start=float(s.start), end=float(s.end), text=s.text.strip())
        for s in segments
    ]


def _transcribe_hf(audio_path, model_size, language, *,
                   window_seconds: float = 30.0,
                   adapter_dir: str | None = None):
    """Transcribe via Hugging Face transformers, manually windowed.

    Used for fine-tuned Whisper models that ship only as HF repos (not CT2 or
    MLX format). Example: `surindersinghssj/surt-small-v3` — Whisper-small
    fine-tuned on 660h of Gurbani audio. Trained without timestamp tokens, so
    we can't use the pipeline's `return_timestamps=True` — we manually window
    the audio into fixed-length chunks instead, assigning each chunk's
    timestamp from its window position.
    """
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    whisper_lang = {"pa": "punjabi", "hi": "hindi", "en": "english"}.get(language, language)

    processor = WhisperProcessor.from_pretrained(model_size, language=whisper_lang, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_size)
    if adapter_dir is not None:
        # PEFT 0.19 checks torch.distributed.tensor.DTensor directly when
        # loading LoRA layers. Torch 2.5 on macOS can import the module but
        # does not attach it to torch.distributed, so provide that attribute
        # before PEFT probes it. This is only needed on the adapter path.
        if not hasattr(torch.distributed, "tensor"):
            import torch.distributed.tensor as distributed_tensor
            torch.distributed.tensor = distributed_tensor
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.generation_config.language = whisper_lang
    model.generation_config.task = "transcribe"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 16000, f"expected 16kHz, got {sr}"

    window_samples = int(window_seconds * sr)
    chunks: list[AsrChunk] = []
    for start_sample in range(0, len(audio), window_samples):
        end_sample = min(len(audio), start_sample + window_samples)
        clip = audio[start_sample:end_sample]
        if len(clip) < sr // 2:
            continue
        inputs = processor(clip, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        with torch.no_grad():
            ids = model.generate(input_features, max_new_tokens=440)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if text:
            chunks.append(AsrChunk(
                start=float(start_sample / sr),
                end=float(end_sample / sr),
                text=text,
            ))
    return chunks


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
