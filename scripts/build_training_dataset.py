#!/usr/bin/env python3
"""Build a kirtan training dataset from YouTube URLs.

User provides a CSV of (youtube_id, shabad_id) pairs. For each row:
  1. Download audio via yt-dlp, convert to 16kHz mono WAV.
  2. Run Path A v3.2 in oracle mode (known shabad) → time-aligned line predictions.
  3. Filter to predictions inside Path A's high-confidence regions (segment durations
     above `--min-clip-sec`, lines actually in the corpus).
  4. Extract per-segment audio clips and pair each with its canonical Gurmukhi text
     from the corpus.
  5. Append rows to `<out-dir>/manifest.json` and write clips to `<out-dir>/clips/`.

Holdout protection: refuses to process any `shabad_id` in the benchmark's test set
(default: 4377, 1821, 1341, 3712). The Path A labels are noisy (we score ~86.5%),
so for cleaner training data, manually review the manifest output and drop bad
rows.

Input CSV format (header optional, one (youtube_id, shabad_id) per row):

    youtube_id,shabad_id,notes
    abc123XYZ,5621,Sukhmani Sahib reading by Bhai Surinder Singh
    def456ABC,12203,
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.asr import transcribe                                      # noqa: E402
from src.matcher import match_chunk, normalize, TfidfScorer         # noqa: E402
from src.shabad_id import identify_shabad                            # noqa: E402
from src.smoother import smooth, smooth_with_stay_bias               # noqa: E402

BENCHMARK_SHABADS = {4377, 1821, 1341, 3712}


def fetch_audio(video_id: str, audio_dir: pathlib.Path) -> pathlib.Path | None:
    """Download YouTube audio and convert to 16kHz mono WAV. Returns path or None."""
    out_path = audio_dir / f"{video_id}_16k.wav"
    if out_path.exists():
        return out_path
    audio_dir.mkdir(parents=True, exist_ok=True)
    raw_path = audio_dir / f"{video_id}.wav"
    url = f"https://youtube.com/watch?v={video_id}"
    try:
        subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "wav",
             "-o", str(audio_dir / "%(id)s.%(ext)s"), url],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw_path), "-ar", "16000", "-ac", "1", str(out_path)],
            check=True, capture_output=True, text=True,
        )
        raw_path.unlink(missing_ok=True)
        return out_path
    except subprocess.CalledProcessError as e:
        print(f"  error fetching {video_id}: {e.stderr[:200]}", file=sys.stderr)
        raw_path.unlink(missing_ok=True)
        return None


def fetch_corpus(shabad_id: int, corpus_dir: pathlib.Path) -> list[dict] | None:
    """Return cached corpus lines for `shabad_id`, fetching from BaniDB if missing."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    path = corpus_dir / f"{shabad_id}.json"
    if path.exists():
        return json.loads(path.read_text())["lines"]
    import urllib.error, urllib.request
    try:
        with urllib.request.urlopen(
            f"https://api.banidb.com/v2/shabads/{shabad_id}", timeout=15
        ) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  error fetching shabad {shabad_id}: {e}", file=sys.stderr)
        return None
    lines = []
    for i, v in enumerate(data.get("verses", [])):
        verse = v.get("verse", {}) or {}
        lines.append({
            "line_idx": i,
            "verse_id": v.get("verseId"),
            "banidb_gurmukhi": verse.get("unicode") or verse.get("gurmukhi") or "",
            "transliteration_english":
                (v.get("transliteration", {}) or {}).get("english", ""),
        })
    if not lines:
        return None
    path.write_text(json.dumps({"shabad_id": shabad_id, "lines": lines},
                                 ensure_ascii=False, indent=2) + "\n")
    return lines


def label_one(
    video_id: str,
    shabad_id: int,
    *,
    audio_dir: pathlib.Path,
    corpus_dir: pathlib.Path,
    asr_cache_dir: pathlib.Path,
    backend: str,
    model_size: str,
) -> list[dict] | None:
    """Run Path A v3.2 (oracle) on one (video, shabad) pair, return segments."""
    audio_path = fetch_audio(video_id, audio_dir)
    if audio_path is None:
        return None
    lines = fetch_corpus(shabad_id, corpus_dir)
    if lines is None:
        return None

    chunks = transcribe(audio_path, backend=backend, model_size=model_size,
                        cache_dir=asr_cache_dir)
    matches = [
        (c.start, c.end,
         match_chunk(c.text, lines, score_threshold=0.0, ratio="WRatio",
                     blend={"token_sort_ratio": 0.5, "WRatio": 0.5}).line_idx)
        for c in chunks
    ]
    # Use offline smoother + stay-bias on the score vector. For labeling we use
    # the post-hoc collapse (no live-mode tentative emission complexity).
    from src.matcher import score_chunk
    scored = [(c.start, c.end,
               score_chunk(c.text, lines, ratio="WRatio",
                           blend={"token_sort_ratio": 0.5, "WRatio": 0.5}))
              for c in chunks]
    segments = smooth_with_stay_bias(scored, stay_margin=6.0, score_threshold=0.0)
    by_idx = {l["line_idx"]: l for l in lines}
    return [{
        "start": s.start, "end": s.end, "line_idx": s.line_idx,
        "shabad_id": shabad_id,
        "text": by_idx[s.line_idx].get("banidb_gurmukhi", ""),
    } for s in segments if s.line_idx in by_idx and s.line_idx != 0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=pathlib.Path, required=True,
                        help="CSV with columns: youtube_id, shabad_id, [notes]")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True,
                        help="Output directory: <dir>/manifest.json + <dir>/clips/")
    parser.add_argument("--audio-dir", type=pathlib.Path,
                        default=REPO_ROOT / "training_data" / "_raw_audio",
                        help="Where to store full-length downloaded WAVs")
    parser.add_argument("--corpus-dir", type=pathlib.Path,
                        default=REPO_ROOT / "corpus_cache")
    parser.add_argument("--asr-cache-dir", type=pathlib.Path,
                        default=REPO_ROOT / "asr_cache")
    parser.add_argument("--backend", default="faster_whisper",
                        choices=["faster_whisper", "mlx_whisper"])
    parser.add_argument("--model", default="medium")
    parser.add_argument("--min-clip-sec", type=float, default=4.0,
                        help="Skip segments shorter than this (likely noise)")
    parser.add_argument("--allow-benchmark-shabads", action="store_true",
                        help="OVERRIDE: process benchmark shabads anyway. "
                             "Almost certainly NOT what you want — this contaminates eval.")
    args = parser.parse_args()

    if not shutil.which("yt-dlp") or not shutil.which("ffmpeg"):
        print("error: yt-dlp and ffmpeg must be on PATH", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = args.out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    manifest: list[dict] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"resuming with existing manifest ({len(manifest)} rows)")

    seen_keys = {(r["video_id"], r["line_idx"], r["start"]) for r in manifest}

    with open(args.input_csv) as f:
        rows = list(csv.DictReader(f))
    if rows and "youtube_id" not in rows[0]:
        print(f"error: input CSV must have a header row with at least "
              f"'youtube_id' and 'shabad_id' columns", file=sys.stderr)
        return 1

    import soundfile as sf
    n_skipped, n_labeled = 0, 0
    for row in rows:
        video_id = row["youtube_id"].strip()
        shabad_id = int(row["shabad_id"])
        if shabad_id in BENCHMARK_SHABADS and not args.allow_benchmark_shabads:
            print(f"skip {video_id}: shabad {shabad_id} is a benchmark shabad "
                  f"(would contaminate eval)")
            n_skipped += 1
            continue

        print(f"\n=== {video_id} (shabad {shabad_id}) ===")
        t0 = time.time()
        segments = label_one(
            video_id, shabad_id,
            audio_dir=args.audio_dir.resolve(),
            corpus_dir=args.corpus_dir.resolve(),
            asr_cache_dir=args.asr_cache_dir.resolve(),
            backend=args.backend, model_size=args.model,
        )
        if segments is None:
            n_skipped += 1
            continue

        audio_path = args.audio_dir.resolve() / f"{video_id}_16k.wav"
        audio, sr = sf.read(str(audio_path), dtype="float32")
        n_for_this = 0
        for s in segments:
            if (s["end"] - s["start"]) < args.min_clip_sec:
                continue
            key = (video_id, s["line_idx"], s["start"])
            if key in seen_keys:
                continue
            clip = audio[int(s["start"]*sr): int(s["end"]*sr)]
            clip_name = f"{video_id}_s{shabad_id}_l{s['line_idx']:02d}_{int(s['start']):04d}.wav"
            clip_path = clips_dir / clip_name
            sf.write(str(clip_path), clip, sr)
            manifest.append({
                "audio": f"clips/{clip_name}",
                "text": s["text"],
                "video_id": video_id,
                "shabad_id": shabad_id,
                "line_idx": s["line_idx"],
                "start": s["start"],
                "end": s["end"],
            })
            seen_keys.add(key)
            n_for_this += 1
            n_labeled += 1
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"  → {n_for_this} clip(s), {time.time()-t0:.1f}s elapsed, "
              f"running total: {len(manifest)} clips")

    print(f"\nfinished: {n_labeled} new clips, {n_skipped} sources skipped, "
          f"manifest at {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
