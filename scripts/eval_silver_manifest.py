#!/usr/bin/env python3
"""Evaluate Whisper/LoRA ASR on a labeled manifest as a silver OOS signal.

This is not the paired benchmark scorer and not a replacement for human-
corrected OOS GT. It answers a narrower question: on broad held-out labeled
segments, does the ASR text move toward or away from canonical Gurmukhi?

Expected manifest shape is the output of ``scripts/pull_dataset.py kirtan``:

    [
      {
        "audio": "clips/<clip>.wav",
        "text": "<canonical gurmukhi>",
        "score": 0.97,
        "video_id": "...",
        "shabad_id": "...",
        "duration_s": 12.3
      },
      ...
    ]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
import time
from dataclasses import asdict, dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rapidfuzz import fuzz  # noqa: E402
from src.matcher import normalize  # noqa: E402


@dataclass
class ManifestRecord:
    idx: int
    audio: pathlib.Path
    text: str
    score: float | None
    video_id: str | None
    shabad_id: str | None
    duration_s: float | None


@dataclass
class SilverRow:
    idx: int
    audio: str
    video_id: str | None
    shabad_id: str | None
    duration_s: float | None
    label_score: float | None
    target: str
    pred: str
    norm_target: str
    norm_pred: str
    ratio: float
    token_sort_ratio: float
    wratio: float
    exact_norm: bool


def _resolve_audio(manifest_path: pathlib.Path, audio_value: str) -> pathlib.Path:
    path = pathlib.Path(audio_value)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def load_manifest(
    manifest_path: pathlib.Path,
    *,
    min_score: float = 0.0,
    limit: int | None = None,
    sample_strategy: str = "first",
) -> list[ManifestRecord]:
    """Load labeled rows from a pull_dataset manifest.

    Rows without audio/text, rows below ``min_score``, and rows whose audio file
    is missing are skipped. ``limit`` applies after filtering.
    """
    raw = json.loads(manifest_path.read_text())
    records: list[ManifestRecord] = []
    for idx, row in enumerate(raw):
        text = str(row.get("text") or "").strip()
        audio_value = str(row.get("audio") or "").strip()
        if not text or not audio_value:
            continue
        score = row.get("score")
        score_f = float(score) if score is not None else None
        if score_f is not None and score_f < min_score:
            continue
        audio = _resolve_audio(manifest_path, audio_value)
        if not audio.exists():
            continue
        records.append(ManifestRecord(
            idx=idx,
            audio=audio,
            text=text,
            score=score_f,
            video_id=row.get("video_id"),
            shabad_id=row.get("shabad_id"),
            duration_s=float(row["duration_s"]) if row.get("duration_s") is not None else None,
        ))
    return select_records(records, limit=limit, sample_strategy=sample_strategy)


def select_records(
    records: list[ManifestRecord],
    *,
    limit: int | None,
    sample_strategy: str,
) -> list[ManifestRecord]:
    if limit is None or len(records) <= limit:
        return records
    if sample_strategy == "first":
        return records[:limit]
    if sample_strategy != "round_robin_video":
        raise ValueError(f"unknown sample strategy: {sample_strategy}")

    groups: dict[str, list[ManifestRecord]] = {}
    order: list[str] = []
    for record in records:
        key = record.video_id or "(unknown)"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)

    out: list[ManifestRecord] = []
    cursor = 0
    while len(out) < limit:
        added = False
        for key in order:
            bucket = groups[key]
            if cursor < len(bucket):
                out.append(bucket[cursor])
                added = True
                if len(out) >= limit:
                    break
        if not added:
            break
        cursor += 1
    return out


def score_prediction(record: ManifestRecord, pred: str) -> SilverRow:
    norm_target = normalize(record.text)
    norm_pred = normalize(pred)
    return SilverRow(
        idx=record.idx,
        audio=str(record.audio),
        video_id=record.video_id,
        shabad_id=record.shabad_id,
        duration_s=record.duration_s,
        label_score=record.score,
        target=record.text,
        pred=pred,
        norm_target=norm_target,
        norm_pred=norm_pred,
        ratio=float(fuzz.ratio(norm_target, norm_pred)),
        token_sort_ratio=float(fuzz.token_sort_ratio(norm_target, norm_pred)),
        wratio=float(fuzz.WRatio(norm_target, norm_pred)),
        exact_norm=norm_target == norm_pred,
    )


def summarize(rows: list[SilverRow]) -> dict:
    if not rows:
        return {
            "n": 0,
            "exact_norm_rate": 0.0,
            "mean_ratio": 0.0,
            "median_ratio": 0.0,
            "mean_wratio": 0.0,
            "median_wratio": 0.0,
            "total_duration_s": 0.0,
            "unique_videos": 0,
            "unique_shabads": 0,
        }
    ratios = [r.ratio for r in rows]
    wratios = [r.wratio for r in rows]
    return {
        "n": len(rows),
        "exact_norm_rate": sum(1 for r in rows if r.exact_norm) / len(rows),
        "mean_ratio": statistics.fmean(ratios),
        "median_ratio": statistics.median(ratios),
        "mean_wratio": statistics.fmean(wratios),
        "median_wratio": statistics.median(wratios),
        "total_duration_s": sum(r.duration_s or 0.0 for r in rows),
        "unique_videos": len({r.video_id for r in rows if r.video_id}),
        "unique_shabads": len({r.shabad_id for r in rows if r.shabad_id}),
    }


def _resample_if_needed(audio, sr: int, target_sr: int = 16000):
    if sr == target_sr:
        return audio
    from scipy.signal import resample_poly
    gcd = math.gcd(sr, target_sr)
    return resample_poly(audio, target_sr // gcd, sr // gcd).astype("float32")


class WhisperRunner:
    """Single-process Whisper runner that keeps the model loaded across clips."""

    def __init__(self, model_id: str, *, adapter_dir: str | None, language: str):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        whisper_lang = {"pa": "punjabi", "hi": "hindi", "en": "english"}.get(language, language)
        self.processor = WhisperProcessor.from_pretrained(model_id, language=whisper_lang, task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        if adapter_dir:
            if not hasattr(torch.distributed, "tensor"):
                import torch.distributed.tensor as distributed_tensor
                torch.distributed.tensor = distributed_tensor
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_dir)
        model.generation_config.language = whisper_lang
        model.generation_config.task = "transcribe"
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch = torch
        self.model = model.to(self.device)
        self.model.eval()

    def transcribe(self, audio_path: pathlib.Path) -> str:
        import numpy as np
        import soundfile as sf

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        audio = _resample_if_needed(np.asarray(audio, dtype="float32"), int(sr))
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        with self.torch.no_grad():
            pred_ids = self.model.generate(input_features)
        pred = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        if self.device == "mps":
            self.torch.mps.empty_cache()
        return pred.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--model-id", default="surindersinghssj/surt-small-v3")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--language", default="pa")
    parser.add_argument("--min-score", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sample-strategy", default="round_robin_video",
                        choices=["first", "round_robin_video"],
                        help="How to pick --limit rows after filtering. "
                             "round_robin_video avoids a smoke run being one-video only.")
    args = parser.parse_args()

    records = load_manifest(
        args.manifest,
        min_score=args.min_score,
        limit=args.limit,
        sample_strategy=args.sample_strategy,
    )
    if not records:
        print("error: no eligible records in manifest", file=sys.stderr)
        return 1

    print(f"silver manifest: {args.manifest}")
    print(f"records: {len(records)} (limit={args.limit}, min_score={args.min_score})")
    print(f"model: {args.model_id}")
    print(f"adapter: {args.adapter_dir or '(none)'}")
    runner = WhisperRunner(args.model_id, adapter_dir=args.adapter_dir, language=args.language)
    print(f"device: {runner.device}")

    rows: list[SilverRow] = []
    t0 = time.perf_counter()
    for i, record in enumerate(records, start=1):
        pred = runner.transcribe(record.audio)
        row = score_prediction(record, pred)
        rows.append(row)
        print(
            f"  {i:>4}/{len(records)} idx={record.idx:<5} "
            f"ratio={row.ratio:5.1f} wratio={row.wratio:5.1f} "
            f"video={record.video_id} shabad={record.shabad_id}",
            flush=True,
        )

    elapsed = time.perf_counter() - t0
    payload = {
        "kind": "silver_manifest_asr_eval",
        "manifest": str(args.manifest),
        "model_id": args.model_id,
        "adapter_dir": args.adapter_dir,
        "language": args.language,
        "min_score": args.min_score,
        "limit": args.limit,
        "sample_strategy": args.sample_strategy,
        "elapsed_s": elapsed,
        "summary": summarize(rows),
        "rows": [asdict(r) for r in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    summary = payload["summary"]
    print()
    print(f"wrote: {args.out}")
    print(
        f"summary: n={summary['n']} mean_wratio={summary['mean_wratio']:.2f} "
        f"median_wratio={summary['median_wratio']:.2f} "
        f"exact={summary['exact_norm_rate']*100:.1f}% "
        f"videos={summary['unique_videos']} shabads={summary['unique_shabads']} "
        f"duration={summary['total_duration_s']/3600:.3f}h"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
