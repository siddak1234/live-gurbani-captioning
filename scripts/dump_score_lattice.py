#!/usr/bin/env python3
"""Dump Phase 2.9 score-lattice diagnostics for locked-shabad alignment."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.asr import transcribe  # noqa: E402
from src.score_lattice import build_score_lattice, score_lattice_summary  # noqa: E402

DEFAULT_GT_DIR = REPO_ROOT.parent / "live-gurbani-captioning-benchmark-v1" / "test"
DEFAULT_AUDIO_DIR = REPO_ROOT / "audio"
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus_cache"
DEFAULT_ASR_CACHE = REPO_ROOT / "asr_cache"
DEFAULT_OUT_DIR = REPO_ROOT / "diagnostics" / "phase2_9_score_lattice"


def _parse_blend(spec: str) -> dict[str, float] | None:
    if not spec.strip():
        return None
    out: dict[str, float] = {}
    for part in spec.split(","):
        name, weight = part.split(":")
        out[name.strip()] = float(weight)
    return out


def _load_corpora(corpus_dir: pathlib.Path) -> dict[int, list[dict]]:
    corpora: dict[int, list[dict]] = {}
    for path in sorted(corpus_dir.glob("*.json")):
        corpus = json.loads(path.read_text())
        corpora[int(corpus["shabad_id"])] = corpus["lines"]
    return corpora


def _row_to_json(row) -> dict:
    out = dataclasses.asdict(row)
    out["top_scores"] = [
        {"line_idx": top["line_idx"], "score": round(float(top["score"]), 3)}
        for top in out["top_scores"]
    ]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=pathlib.Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--audio-dir", type=pathlib.Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--corpus-dir", type=pathlib.Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--asr-cache-dir", type=pathlib.Path, default=DEFAULT_ASR_CACHE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--backend", default="huggingface_whisper",
                        choices=["faster_whisper", "mlx_whisper", "huggingface_whisper"])
    parser.add_argument("--model", default="surindersinghssj/surt-small-v3")
    parser.add_argument("--adapter-dir", default="lora_adapters/v5b_mac_diverse")
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--vad-filter", action="store_true")
    parser.add_argument("--no-speech-threshold", type=float, default=None)
    parser.add_argument("--blind-lookback", type=float, default=30.0)
    parser.add_argument("--ratio", default="WRatio")
    parser.add_argument("--blend", default="token_sort_ratio:0.5,WRatio:0.5")
    parser.add_argument("--stay-bias", type=float, default=6.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    corpora = _load_corpora(args.corpus_dir.resolve())
    if not corpora:
        print(f"error: no corpus files in {args.corpus_dir}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    for gt_path in sorted(args.gt_dir.glob("*.json")):
        gt = json.loads(gt_path.read_text())
        shabad_id = int(gt["shabad_id"])
        audio_path = args.audio_dir / f"{gt['video_id']}_16k.wav"
        if shabad_id not in corpora:
            print(f"error: missing corpus for shabad {shabad_id}", file=sys.stderr)
            return 1
        chunks = transcribe(
            audio_path,
            backend=args.backend,
            model_size=args.model,
            cache_dir=args.asr_cache_dir.resolve(),
            adapter_dir=args.adapter_dir,
            word_timestamps=args.word_timestamps,
            no_speech_threshold=args.no_speech_threshold,
            vad_filter=args.vad_filter,
        )
        commit_time = float(gt.get("uem", {}).get("start", 0.0)) + float(args.blind_lookback)
        post_lock_chunks = [chunk for chunk in chunks if float(chunk.end) >= commit_time]
        rows = build_score_lattice(
            post_lock_chunks,
            corpora[shabad_id],
            gt.get("segments", []),
            ratio=args.ratio,
            blend=_parse_blend(args.blend),
            stay_margin=args.stay_bias,
            score_threshold=args.threshold,
            top_k=args.top_k,
        )
        case_summary = score_lattice_summary(rows)
        case_summary["case"] = gt_path.stem
        case_summary["shabad_id"] = shabad_id
        case_summary["commit_time"] = round(commit_time, 3)
        summary_rows.append(case_summary)
        (args.out_dir / f"{gt_path.stem}.json").write_text(json.dumps({
            "case": gt_path.stem,
            "video_id": gt["video_id"],
            "shabad_id": shabad_id,
            "commit_time": commit_time,
            "config": {
                "backend": args.backend,
                "model": args.model,
                "adapter_dir": args.adapter_dir,
                "ratio": args.ratio,
                "blend": _parse_blend(args.blend),
                "stay_bias": args.stay_bias,
                "threshold": args.threshold,
            },
            "summary": case_summary,
            "rows": [_row_to_json(row) for row in rows],
        }, ensure_ascii=False, indent=2))
        print(
            f"  {gt_path.stem}: chunks={case_summary['chunks']} "
            f"with_gt={case_summary['chunks_with_gt']} "
            f"best_ok={case_summary['best_matches_gt']} "
            f"stay_ok={case_summary['stay_matches_gt']}",
            flush=True,
        )

    total = {
        "cases": len(summary_rows),
        "chunks": sum(row["chunks"] for row in summary_rows),
        "chunks_with_gt": sum(row["chunks_with_gt"] for row in summary_rows),
        "best_matches_gt": sum(row["best_matches_gt"] for row in summary_rows),
        "stay_matches_gt": sum(row["stay_matches_gt"] for row in summary_rows),
        "null_gt_chunks": sum(row["null_gt_chunks"] for row in summary_rows),
    }
    (args.out_dir / "summary.json").write_text(json.dumps({
        "total": total,
        "cases": summary_rows,
    }, ensure_ascii=False, indent=2))
    print(f"\nwrote diagnostics to {args.out_dir}")
    print(json.dumps(total, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
