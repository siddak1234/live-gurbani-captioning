"""Training-data loader for Path B fine-tuning.

Expects a manifest JSON file describing audio/transcript pairs:

    [
      {"audio": "path/to/clip1.wav", "text": "ਪੋਥੀ ਪਰਮੇਸਰ ਕਾ ਥਾਨੁ"},
      {"audio": "path/to/clip2.wav", "text": "..."},
      ...
    ]

`audio` paths can be absolute or relative to the manifest's directory.
`text` is the canonical Gurmukhi line (no verse-number markers, no
punctuation outside the inherent unicode).

Audio is loaded with soundfile, resampled to 16kHz mono if needed.
The dataset returns dicts with `input_values` (audio array) and `labels`
(tokenized transcript), ready for HF's Trainer.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np


def load_manifest(manifest_path: pathlib.Path | str) -> list[dict]:
    """Read a manifest JSON; resolve audio paths relative to the manifest dir."""
    manifest_path = pathlib.Path(manifest_path).resolve()
    base_dir = manifest_path.parent
    records = json.loads(manifest_path.read_text())
    out: list[dict] = []
    for r in records:
        audio_path = pathlib.Path(r["audio"])
        if not audio_path.is_absolute():
            audio_path = (base_dir / audio_path).resolve()
        out.append({"audio_path": str(audio_path), "text": r["text"]})
    return out


def _load_audio_16k_mono(audio_path: str) -> np.ndarray:
    """Load any audio file, downmix to mono, resample to 16kHz, return float32."""
    import soundfile as sf
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr)).astype(np.float32)
    return audio


def to_hf_dataset(records: list[dict], tokenizer, feature_extractor):
    """Wrap CTC training records into a HF Dataset.

    Used by the CTC fine-tune path (w2v-bert, MMS, wav2vec2). Feature extractors
    produce either ``input_values`` (raw waveform) or ``input_features``
    (mel-spectrogram) depending on architecture — we emit rows matching the
    model's forward signature.
    """
    from datasets import Dataset

    def gen():
        for r in records:
            audio = _load_audio_16k_mono(r["audio_path"])
            features = feature_extractor(audio, sampling_rate=16000, return_tensors="np")
            row: dict = {"labels": tokenizer(r["text"], return_tensors="np").input_ids[0].astype(np.int64)}
            if "input_features" in features:
                arr = features["input_features"][0].astype(np.float32)
                row["input_features"] = arr
                row["input_length"] = arr.shape[0]
            else:
                arr = features["input_values"][0].astype(np.float32)
                row["input_values"] = arr
                row["input_length"] = len(arr)
            yield row

    return Dataset.from_generator(gen)


def to_hf_dataset_whisper(records: list[dict], processor):
    """Wrap Whisper training records into a HF Dataset.

    Used by the Whisper / Seq2Seq fine-tune path (surt-small-v3, openai/whisper-*).
    The combined :class:`AutoProcessor` handles both audio preprocessing
    (80-mel filterbank at 16 kHz, 30-second pad/truncate → shape (80, 3000))
    and text tokenization (BPE → token ids).
    """
    from datasets import Dataset

    def gen():
        for r in records:
            audio = _load_audio_16k_mono(r["audio_path"])
            # Whisper feature extractor returns input_features shape (1, 80, 3000)
            features = processor.feature_extractor(
                audio, sampling_rate=16000, return_tensors="np"
            )
            labels = processor.tokenizer(r["text"], return_tensors="np").input_ids[0].astype(np.int64)
            yield {
                "input_features": features["input_features"][0].astype(np.float32),
                "labels": labels,
                "input_length": features["input_features"][0].shape[-1],
            }

    return Dataset.from_generator(gen)
