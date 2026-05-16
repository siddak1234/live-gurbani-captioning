#!/usr/bin/env python3
"""Phase B0: probe an off-the-shelf CTC model on kirtan audio.

Goal: confirm Path B is viable by checking that an off-the-shelf CTC model
produces recognizable character/phoneme outputs for sung kirtan audio.

Uses Meta's MMS-1B model with the Punjabi (`pan`) language adapter. Outputs
per-frame Gurmukhi-character probability distributions; we eye-test the greedy
decode and the top-3 distributions at sample frames.

Usage:
    python scripts/probe_ctc.py                                   # default 60s on IZOsmkdmmcg
    python scripts/probe_ctc.py audio/IZOsmkdmmcg_16k.wav 30      # custom audio + seconds

If the greedy decode contains the canonical Gurmukhi well enough to be
recognizable, Path B is viable. If it's garbage or the model can't even produce
Punjabi characters from kirtan-tempo audio, we need a different approach.
"""

from __future__ import annotations

import pathlib
import sys
import time


def main() -> int:
    audio_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("audio/IZOsmkdmmcg_16k.wav")
    seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    import soundfile as sf
    import torch
    from transformers import Wav2Vec2ForCTC, AutoProcessor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_id = "facebook/mms-1b-all"
    print(f"Loading {model_id} on {device} (first run downloads ~3GB)...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(
        model_id, target_lang="pan", ignore_mismatched_sizes=True
    ).to(device)
    processor.tokenizer.set_target_lang("pan")
    print(f"  loaded in {time.time()-t0:.1f}s")

    print(f"\nLoading first {seconds}s of {audio_path}...")
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 16000, f"expected 16kHz audio, got {sr}"
    audio = audio[: seconds * sr]
    print(f"  shape: {audio.shape}, duration: {len(audio)/16000:.1f}s")

    print("\nRunning inference...")
    t0 = time.time()
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  logits shape: {logits.shape}  ({logits.shape[1]/(len(audio)/16000):.1f} frames/sec)")
    print(f"  vocab size: {logits.shape[2]}")

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    print("\n=== Greedy CTC decode ===")
    print(f"  {transcription}")

    probs = torch.softmax(logits[0], dim=-1).cpu()
    vocab = processor.tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    blank_id = processor.tokenizer.pad_token_id

    n_frames = probs.shape[0]
    sample_every = max(1, n_frames // 20)
    print(f"\n=== Top-3 tokens at sampled frames (every {sample_every} frames) ===")
    for t in range(0, n_frames, sample_every):
        top3 = torch.topk(probs[t], 3)
        secs = t / (n_frames / (len(audio) / 16000))
        items = []
        for i, p in zip(top3.indices, top3.values):
            sym = inv_vocab.get(i.item(), f"?{i.item()}")
            tag = "_" if i.item() == blank_id else sym
            items.append(f"{tag!r}={p.item():.2f}")
        print(f"  t={t:4d} ({secs:5.1f}s): {', '.join(items)}")

    print(f"\n(blank token id={blank_id} — CTC blank, shown as `_` above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
