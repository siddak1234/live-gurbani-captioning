#!/usr/bin/env python3
"""Pull kirtan training data from surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical.

Streams from HuggingFace, filters by confidence + holdout rules, writes audio
clips + a manifest.json in our existing training-pipeline format.

Holdout discipline (strict — these are the benchmark's 4 shabads/videos):
  - Drop any clip with canonical_shabad_id in {4377, 1821, 1341, 3712}
  - Drop any clip with video_id in {IZOsmkdmmcg, kZhIA8P6xWI, kchMJPK9Axs, zOtIpxMT9hU}

Quality filter: canonical_match_score >= --min-score (default 0.8).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

BENCHMARK_SHABADS = {"4377", "1821", "1341", "3712", 4377, 1821, 1341, 3712}
BENCHMARK_VIDEOS = {"IZOsmkdmmcg", "kZhIA8P6xWI", "kchMJPK9Axs", "zOtIpxMT9hU"}

DATASET_ID = "surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--num-samples", type=int, default=50,
                        help="How many qualifying samples to pull (default 50)")
    parser.add_argument("--min-score", type=float, default=0.8,
                        help="Minimum canonical_match_score (default 0.8)")
    parser.add_argument("--allow-simran", action="store_true",
                        help="Include simran clips (default: filter out — they're repetitive)")
    parser.add_argument("--max-scan", type=int, default=5000,
                        help="Stop scanning the stream after this many rows (default 5000)")
    args = parser.parse_args()

    import io

    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    out_dir = args.out_dir.resolve()
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Download one parquet shard directly. Each shard is roughly 1/129 of the
    # full 300h dataset (~2.3 hours of audio), small enough to handle locally
    # and gives us thousands of rows to filter from.
    print(f"downloading first parquet shard of {DATASET_ID} ...")
    parquet_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename="data/train-00000-of-00129.parquet",
        repo_type="dataset",
    )
    print(f"  downloaded to {parquet_path}")

    table = pq.read_table(parquet_path)
    print(f"  parquet rows: {table.num_rows}")
    ds = (dict(zip(table.column_names, row))
          for row in zip(*[table.column(c).to_pylist() for c in table.column_names]))

    manifest: list[dict] = []
    n_scanned = 0
    n_skipped_shabad = 0
    n_skipped_video = 0
    n_skipped_score = 0
    n_skipped_simran = 0
    n_kept = 0

    for row in ds:
        n_scanned += 1
        if n_scanned > args.max_scan:
            print(f"  hit --max-scan limit ({args.max_scan}); stopping")
            break
        if len(manifest) >= args.num_samples:
            break

        shabad_id = row.get("canonical_shabad_id")
        if shabad_id in BENCHMARK_SHABADS or str(shabad_id) in BENCHMARK_SHABADS:
            n_skipped_shabad += 1
            continue
        video_id = row.get("video_id", "")
        if video_id in BENCHMARK_VIDEOS:
            n_skipped_video += 1
            continue
        score = float(row.get("canonical_match_score") or 0.0)
        if score < args.min_score:
            n_skipped_score += 1
            continue
        if (not args.allow_simran) and bool(row.get("is_simran", False)):
            n_skipped_simran += 1
            continue

        text = (row.get("final_text") or row.get("text") or "").strip()
        if not text:
            continue
        # Audio is stored as a dict {bytes, path} in parquet (datasets Audio
        # feature format). We decode the bytes via soundfile.
        audio = row.get("audio") or {}
        audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
        if not audio_bytes:
            continue
        try:
            arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception:
            continue

        clip_id = row.get("clip_id") or f"clip_{len(manifest):06d}"
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(clip_id))
        clip_path = clips_dir / f"{safe}.wav"
        sf.write(str(clip_path), arr, int(sr))
        manifest.append({
            "audio": f"clips/{clip_path.name}",
            "text": text,
            "video_id": video_id,
            "shabad_id": str(shabad_id) if shabad_id is not None else "",
            "score": score,
            "duration_s": float(row.get("duration_s") or 0.0),
        })
        n_kept += 1
        if n_kept % 10 == 0:
            print(f"  kept {n_kept}/{args.num_samples} (scanned {n_scanned})")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"\nscanned: {n_scanned}")
    print(f"kept:    {n_kept}")
    print(f"skipped: shabad={n_skipped_shabad} video={n_skipped_video} "
          f"score={n_skipped_score} simran={n_skipped_simran}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
