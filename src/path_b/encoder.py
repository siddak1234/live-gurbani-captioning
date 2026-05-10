"""MMS-1B CTC encoder wrapper.

Loads `facebook/mms-1b-all` with the Punjabi (`pan`) language adapter and
produces per-frame log-probability matrices for audio files. Caches the
log-prob matrix to disk so re-runs don't re-encode (encoding ~7 sec per file
on Apple Silicon MPS; cached load is instant).

Cache file layout per audio:
  <cache_dir>/<stem>__mms-pan.npz       log_probs as compressed npz
  <cache_dir>/<stem>__mms-pan.meta.json frame_duration, blank_id, vocab
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np


@dataclass
class CtcOutput:
    log_probs: np.ndarray            # (T_frames, vocab_size), float32, log-space
    frame_duration: float            # seconds per frame (~0.02 for MMS)
    vocab: dict[str, int]            # character → token id
    inv_vocab: dict[int, str]        # token id → character
    blank_id: int


_encoder_singleton = None


class _MmsEncoder:
    def __init__(self, model_id: str = "facebook/mms-1b-all", target_lang: str = "pan"):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.set_target_lang(target_lang)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_id, target_lang=target_lang, ignore_mismatched_sizes=True
        )
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.vocab = self.processor.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.blank_id = self.processor.tokenizer.pad_token_id

    def encode(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        chunk_seconds: float = 60.0,
    ) -> CtcOutput:
        """Encode audio in `chunk_seconds`-sized pieces and concatenate the log-probs.

        wav2vec2 attention is O(T^2); a full kirtan track (7-10 min) overflows
        Apple Silicon unified memory unless we chunk. 60s chunks fit comfortably
        and wav2vec2 has only ~25ms of effective right-context, so chunk
        boundaries don't introduce meaningful seams.
        """
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

        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, T, V)
        log_probs = torch.log_softmax(logits[0], dim=-1).cpu().numpy().astype(np.float32)
        duration = len(audio) / sample_rate
        frame_duration = duration / log_probs.shape[0]
        return log_probs, frame_duration


def _get_encoder() -> _MmsEncoder:
    global _encoder_singleton
    if _encoder_singleton is None:
        _encoder_singleton = _MmsEncoder()
    return _encoder_singleton


def encode_file(
    audio_path: pathlib.Path | str,
    *,
    cache_dir: pathlib.Path | str | None = None,
) -> CtcOutput:
    """Encode an audio file with MMS, returning CtcOutput. Caches to disk if `cache_dir`."""
    import soundfile as sf

    audio_path = pathlib.Path(audio_path)
    cache_path: pathlib.Path | None = None
    meta_path: pathlib.Path | None = None
    if cache_dir is not None:
        cache_dir = pathlib.Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{audio_path.stem}__mms-pan.npz"
        meta_path = cache_dir / f"{audio_path.stem}__mms-pan.meta.json"
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
    encoder = _get_encoder()
    out = encoder.encode(audio, sample_rate=sr)

    if cache_path is not None and meta_path is not None:
        np.savez_compressed(cache_path, log_probs=out.log_probs)
        meta = {
            "frame_duration": out.frame_duration,
            "blank_id": out.blank_id,
            "inv_vocab": {str(k): v for k, v in out.inv_vocab.items()},
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return out
