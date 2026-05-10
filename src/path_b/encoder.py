"""CTC encoder wrappers for Path B.

Supports two model families:

- **MMS-1B with language adapters** (`facebook/mms-1b-all`, `target_lang="pan"`).
  Generic multilingual model adapted to Punjabi via small adapter layers.

- **Punjabi-specialized CTC models** (e.g. `kdcyberdude/w2v-bert-punjabi`,
  `gagan3012/wav2vec2-xlsr-punjabi`, Vakyansh, etc.). Already fine-tuned on
  Punjabi speech; loaded via the generic `AutoModelForCTC`.

Cache filenames encode the model key + language so transcripts don't collide
across model swaps. Encoding is chunked (60s pieces) so wav2vec2's O(T^2)
attention fits in Apple Silicon unified memory.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np


@dataclass
class CtcOutput:
    log_probs: np.ndarray
    frame_duration: float
    vocab: dict[str, int]
    inv_vocab: dict[int, str]
    blank_id: int


_encoder_cache: dict[tuple[str, str | None], "_CtcEncoder"] = {}


def _is_mms(model_id: str) -> bool:
    return "mms" in model_id.lower()


def _cache_key(model_id: str) -> str:
    """Sanitize a HF repo path for use in a filename."""
    return model_id.split("/")[-1].replace("/", "_").replace(":", "_")


class _CtcEncoder:
    def __init__(
        self,
        model_id: str,
        target_lang: str | None = None,
        adapter_dir: str | None = None,
    ):
        import torch
        from transformers import (
            AutoFeatureExtractor,
            AutoModelForCTC,
            AutoTokenizer,
            Wav2Vec2ForCTC,
        )

        self.model_id = model_id
        self.target_lang = target_lang
        self.adapter_dir = adapter_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

        if _is_mms(model_id) and target_lang:
            self.tokenizer.set_target_lang(target_lang)
            base_model = Wav2Vec2ForCTC.from_pretrained(
                model_id, target_lang=target_lang, ignore_mismatched_sizes=True
            )
        else:
            base_model = AutoModelForCTC.from_pretrained(model_id)

        # If a LoRA / PEFT adapter dir is provided, load and apply on top of base.
        if adapter_dir is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        else:
            self.model = base_model

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.blank_id = self.tokenizer.pad_token_id

    def encode(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        chunk_seconds: float = 60.0,
    ) -> CtcOutput:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        chunk_samples = int(chunk_seconds * sample_rate)
        pieces: list[np.ndarray] = []
        frame_duration = 0.02
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start : start + chunk_samples]
            log_probs, frame_duration = self._encode_chunk(chunk, sample_rate)
            pieces.append(log_probs)
        full = np.concatenate(pieces, axis=0)
        return CtcOutput(full, frame_duration, self.vocab, self.inv_vocab, self.blank_id)

    def _encode_chunk(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
        import torch

        inputs = self.feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        log_probs = torch.log_softmax(logits[0], dim=-1).cpu().numpy().astype(np.float32)
        duration = len(audio) / sample_rate
        frame_duration = duration / log_probs.shape[0]
        return log_probs, frame_duration


def _get_encoder(
    model_id: str, target_lang: str | None, adapter_dir: str | None = None,
) -> _CtcEncoder:
    key = (model_id, target_lang, adapter_dir)
    if key not in _encoder_cache:
        _encoder_cache[key] = _CtcEncoder(model_id, target_lang, adapter_dir)
    return _encoder_cache[key]


def encode_file(
    audio_path: pathlib.Path | str,
    *,
    model_id: str = "facebook/mms-1b-all",
    target_lang: str | None = "pan",
    adapter_dir: str | None = None,
    cache_dir: pathlib.Path | str | None = None,
) -> CtcOutput:
    """Encode an audio file with the chosen CTC model.

    Caches log-probs to disk so re-runs are instant. Cache key includes
    sanitized model_id + lang so swapping models doesn't collide.
    """
    import soundfile as sf

    audio_path = pathlib.Path(audio_path)
    cache_path: pathlib.Path | None = None
    meta_path: pathlib.Path | None = None
    if cache_dir is not None:
        cache_dir = pathlib.Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{_cache_key(model_id)}"
        if target_lang and _is_mms(model_id):
            tag += f"-{target_lang}"
        if adapter_dir:
            tag += f"_lora-{_cache_key(adapter_dir)}"
        cache_path = cache_dir / f"{audio_path.stem}__{tag}.npz"
        meta_path = cache_dir / f"{audio_path.stem}__{tag}.meta.json"
        if cache_path.exists() and meta_path.exists():
            data = np.load(cache_path)
            meta = json.loads(meta_path.read_text())
            inv_vocab = {int(k): v for k, v in meta["inv_vocab"].items()}
            return CtcOutput(
                log_probs=data["log_probs"],
                frame_duration=float(meta["frame_duration"]),
                vocab={v: k for k, v in inv_vocab.items()},
                inv_vocab=inv_vocab,
                blank_id=int(meta["blank_id"]),
            )

    audio, sr = sf.read(str(audio_path), dtype="float32")
    assert sr == 16000, f"expected 16kHz, got {sr}"
    encoder = _get_encoder(model_id, target_lang, adapter_dir)
    out = encoder.encode(audio, sample_rate=sr)

    if cache_path is not None and meta_path is not None:
        np.savez_compressed(cache_path, log_probs=out.log_probs)
        meta_path.write_text(json.dumps({
            "frame_duration": out.frame_duration,
            "blank_id": out.blank_id,
            "inv_vocab": {str(k): v for k, v in out.inv_vocab.items()},
        }, ensure_ascii=False, indent=2))
    return out
