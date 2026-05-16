#!/usr/bin/env python3
"""Unified data-pull entrypoint for Path B fine-tuning.

Subcommands pull from different labeled sources and write a uniform manifest
(``manifest.json``) plus extracted audio (``clips/<id>.wav``) into ``--out-dir``.

Sources:

  kirtan       surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical (300h)
  sehaj        surindersinghssj/gurbani-sehajpath-yt-captions-canonical    (~160h)
  sehajpath    surindersinghssj/gurbani-sehajpath                         (~66h studio)
  indicvoices  ai4bharat/IndicVoices (Punjabi subset)                     [stub]
  commonvoice  mozilla-foundation/common_voice_12_0 (Punjabi)             [stub]
  archiveorg   archive.org bulk Gurbani Kirtan                            [stub]

Holdout discipline (enforced for Gurbani-specific sources):
  Per-source holdout policy lives in configs/datasets.yaml under
  ``sources.<key>.holdout``. Today every Gurbani source enforces the same
  4 benchmark shabads + 4 benchmark videos; the YAML lets us evolve that
  per-source without touching this script.

Uniform manifest schema (one record per clip):
  {"audio": "clips/...", "text": "...", "source": "<subcmd>", "duration_s": float,
   "shabad_id": "..."?, "video_id": "..."?, "score": float?}

Quality filter via --min-score (default 0.8) applies to all surindersinghssj sources.
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys
from typing import Iterable, Iterator

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASETS_CONFIG_PATH = REPO_ROOT / "configs" / "datasets.yaml"

# Known parquet-shard datasets on HF, single-shard schema "data/train-NNNNN-of-NNNNN.parquet".
SURT_DATASETS = {
    "kirtan":    "surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical",
    "sehaj":     "surindersinghssj/gurbani-sehajpath-yt-captions-canonical",
    "sehajpath": "surindersinghssj/gurbani-sehajpath",
}


# -----------------------------------------------------------------------------
# Dataset registry — single source of truth for what data exists + holdout policy.
# configs/datasets.yaml lists every source, its HF repo, quality columns, AND
# the per-source benchmark holdout (the 4 shabads + 4 videos we never train on).
# Previously the holdout was hardcoded here; now it loads from YAML so a single
# edit propagates to the script, the model card, and any other consumer.
# -----------------------------------------------------------------------------

_DATASETS_CONFIG_CACHE: dict | None = None


def load_dataset_config(path: pathlib.Path | None = None) -> dict:
    """Read configs/datasets.yaml once and memoize. Override ``path`` in tests.

    Caller can pass an explicit path to bypass the cache (used by the test
    suite to test malformed configs without polluting subsequent calls).
    """
    global _DATASETS_CONFIG_CACHE
    if path is not None:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    if _DATASETS_CONFIG_CACHE is None:
        import yaml
        with open(DATASETS_CONFIG_PATH) as f:
            _DATASETS_CONFIG_CACHE = yaml.safe_load(f) or {}
    return _DATASETS_CONFIG_CACHE


def get_holdout(source_key: str, cfg: dict | None = None) -> tuple[set, set, bool]:
    """Resolve (shabad_set, video_set, enforce) for one source from the registry.

    The shabad_set is populated with BOTH the str and int form of each id —
    HF parquet ``canonical_shabad_id`` columns can come back as either type
    depending on schema, so a single membership test handles both.

    Missing/unknown ``source_key`` returns empty sets + ``enforce=False`` so
    new sources that haven't been wired into the YAML yet pass through
    without holdout. The caller's contract: pass ``enforce_gurbani_holdout``
    only for sources that have shabad-level metadata; for unlabeled or
    non-Gurbani sources, no holdout applies.
    """
    cfg = cfg if cfg is not None else load_dataset_config()
    src = cfg.get("sources", {}).get(source_key, {})
    holdout = src.get("holdout", {}) or {}
    enforce = bool(holdout.get("enforce", False))

    shabad_set: set = set()
    for sid in holdout.get("shabad_ids") or []:
        shabad_set.add(str(sid))
        try:
            shabad_set.add(int(sid))
        except (TypeError, ValueError):
            pass

    video_set: set = set(holdout.get("video_ids") or [])
    return shabad_set, video_set, enforce


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

def _iter_first_parquet_shard(dataset_id: str, shard_idx: int = 0) -> Iterator[dict]:
    """Download the (shard_idx)-th parquet shard from an HF dataset and yield rows.

    Doesn't stream the full dataset — pulls one parquet file (~2-5h of audio for
    surindersinghssj sources). Caller stops early via --num-samples.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(repo_id=dataset_id, repo_type="dataset")
             if f.startswith("data/train-") and f.endswith(".parquet")]
    files.sort()
    if not files:
        raise RuntimeError(f"no parquet files found in dataset {dataset_id}")
    if shard_idx >= len(files):
        raise RuntimeError(f"shard {shard_idx} out of range (dataset has {len(files)} shards)")

    target = files[shard_idx]
    print(f"  downloading {target} from {dataset_id} ...")
    parquet_path = hf_hub_download(repo_id=dataset_id, filename=target, repo_type="dataset")
    print(f"  downloaded to {parquet_path}")

    table = pq.read_table(parquet_path)
    print(f"  parquet rows: {table.num_rows}")
    cols = table.column_names
    for row in zip(*[table.column(c).to_pylist() for c in cols]):
        yield dict(zip(cols, row))


def _is_held_out(shabad_id, video_id: str, *, shabads: set, videos: set) -> tuple[bool, str]:
    """Check holdout against per-source sets resolved from configs/datasets.yaml.

    Returns ``(True, reason)`` where reason is ``"shabad"`` or ``"video"`` so
    the puller can tally rejections by cause for the data card. Both str and
    int forms of the shabad_id are tested; ``shabads`` is expected to contain
    both forms (see get_holdout).
    """
    if shabad_id in shabads or str(shabad_id) in shabads:
        return True, "shabad"
    if video_id in videos:
        return True, "video"
    return False, ""


def _write_audio_clip(audio_field, clip_id: str, clips_dir: pathlib.Path) -> tuple[pathlib.Path | None, float]:
    """Write the audio dict (HF Audio feature format: {bytes, path}) to a wav file.

    Returns (path, duration_s) or (None, 0.0) on failure.
    """
    import soundfile as sf
    audio_bytes = audio_field.get("bytes") if isinstance(audio_field, dict) else None
    if not audio_bytes:
        return None, 0.0
    try:
        arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        return None, 0.0
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(clip_id))
    clip_path = clips_dir / f"{safe}.wav"
    sf.write(str(clip_path), arr, int(sr))
    duration = float(len(arr)) / float(sr) if sr else 0.0
    return clip_path, duration


def _write_manifest(out_dir: pathlib.Path, manifest: list[dict]) -> pathlib.Path:
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest_path


# -----------------------------------------------------------------------------
# Subcommand: kirtan / sehaj / sehajpath (shared SURT parquet schema)
# -----------------------------------------------------------------------------

def _run_surt_puller(args, source_key: str, *, enforce_gurbani_holdout: bool) -> int:
    dataset_id = SURT_DATASETS[source_key]
    out_dir = args.out_dir.resolve()
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Resolve per-source holdout from configs/datasets.yaml. The caller's
    # enforce_gurbani_holdout flag and the YAML's holdout.enforce both have
    # to be true for filtering to apply — caller wins if it says "don't".
    holdout_shabads, holdout_videos, yaml_enforce = get_holdout(source_key)
    apply_holdout = enforce_gurbani_holdout and yaml_enforce
    print(f"  holdout: enforce={apply_holdout} "
          f"(shabads={sorted(s for s in holdout_shabads if isinstance(s, str))}, "
          f"videos={sorted(holdout_videos)})")

    manifest: list[dict] = []
    n_scanned = n_kept = 0
    n_skipped_shabad = n_skipped_video = n_skipped_score = n_skipped_simran = 0

    for row in _iter_first_parquet_shard(dataset_id, shard_idx=args.shard):
        n_scanned += 1
        if n_scanned > args.max_scan:
            print(f"  hit --max-scan limit ({args.max_scan}); stopping")
            break
        if len(manifest) >= args.num_samples:
            break

        shabad_id = row.get("canonical_shabad_id")
        video_id = row.get("video_id", "")

        if apply_holdout:
            held, reason = _is_held_out(shabad_id, video_id,
                                        shabads=holdout_shabads, videos=holdout_videos)
            if held:
                if reason == "shabad": n_skipped_shabad += 1
                else:                  n_skipped_video  += 1
                continue

        score = float(row.get("canonical_match_score") or 0.0)
        if score < args.min_score:
            n_skipped_score += 1
            continue

        if (not args.allow_simran) and bool(row.get("is_simran", False)):
            n_skipped_simran += 1
            continue

        text = (row.get("final_text") or row.get("text") or row.get("gurmukhi_text") or "").strip()
        if not text:
            continue

        clip_id = row.get("clip_id") or f"{source_key}_{len(manifest):06d}"
        clip_path, duration = _write_audio_clip(row.get("audio"), clip_id, clips_dir)
        if not clip_path:
            continue

        record = {
            "audio": f"clips/{clip_path.name}",
            "text": text,
            "source": source_key,
            "shabad_id": str(shabad_id) if shabad_id is not None else "",
            "video_id": video_id,
            "score": score,
            "duration_s": duration,
        }
        manifest.append(record)
        n_kept += 1
        if n_kept % 10 == 0:
            print(f"  kept {n_kept}/{args.num_samples} (scanned {n_scanned})")

    manifest_path = _write_manifest(out_dir, manifest)
    print(f"\nscanned: {n_scanned}")
    print(f"kept:    {n_kept}")
    print(f"skipped: shabad={n_skipped_shabad} video={n_skipped_video} "
          f"score={n_skipped_score} simran={n_skipped_simran}")
    print(f"manifest: {manifest_path}")
    return 0


def cmd_kirtan(args):
    return _run_surt_puller(args, "kirtan", enforce_gurbani_holdout=True)


def cmd_sehaj(args):
    return _run_surt_puller(args, "sehaj", enforce_gurbani_holdout=True)


def cmd_sehajpath(args):
    return _run_surt_puller(args, "sehajpath", enforce_gurbani_holdout=True)


# -----------------------------------------------------------------------------
# Subcommand stubs — to flesh out as we expand data coverage
# -----------------------------------------------------------------------------

def cmd_indicvoices(args):
    print(
        "indicvoices subcommand is a stub. Planned: stream ai4bharat/IndicVoices "
        "Punjabi subset via datasets.load_dataset, write uniform manifest. "
        "Not implemented yet — will land alongside the OCI training milestone "
        "where we actually need it.",
        file=sys.stderr,
    )
    return 2


def cmd_commonvoice(args):
    print(
        "commonvoice subcommand is a stub. Planned: stream "
        "mozilla-foundation/common_voice_12_0 Punjabi (pa-IN) subset, filter to "
        "validated clips, write uniform manifest. Not implemented yet.",
        file=sys.stderr,
    )
    return 2


def cmd_archiveorg(args):
    print(
        "archiveorg subcommand is a stub. Planned: scrape archive.org bulk "
        "GurbaniKirtan collection, auto-label via surt + BaniDB forced alignment, "
        "confidence-filter, write uniform manifest. Not implemented yet.",
        file=sys.stderr,
    )
    return 2


# -----------------------------------------------------------------------------
# Argparse setup
# -----------------------------------------------------------------------------

def _add_surt_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--num-samples", type=int, default=50,
                   help="How many qualifying samples to keep (default 50)")
    p.add_argument("--min-score", type=float, default=0.8,
                   help="Minimum canonical_match_score (default 0.8)")
    p.add_argument("--allow-simran", action="store_true",
                   help="Include simran clips (default: filter out — repetitive)")
    p.add_argument("--max-scan", type=int, default=5000,
                   help="Stop scanning the stream after this many rows (default 5000)")
    p.add_argument("--shard", type=int, default=0,
                   help="Which parquet shard to pull (default 0 = first)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_k = sub.add_parser("kirtan", help="surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical")
    _add_surt_args(p_k); p_k.set_defaults(func=cmd_kirtan)

    p_s = sub.add_parser("sehaj", help="surindersinghssj/gurbani-sehajpath-yt-captions-canonical")
    _add_surt_args(p_s); p_s.set_defaults(func=cmd_sehaj)

    p_sp = sub.add_parser("sehajpath", help="surindersinghssj/gurbani-sehajpath (old studio)")
    _add_surt_args(p_sp); p_sp.set_defaults(func=cmd_sehajpath)

    p_iv = sub.add_parser("indicvoices", help="ai4bharat/IndicVoices Punjabi subset [stub]")
    p_iv.add_argument("--out-dir", type=pathlib.Path, required=True)
    p_iv.add_argument("--num-samples", type=int, default=50)
    p_iv.set_defaults(func=cmd_indicvoices)

    p_cv = sub.add_parser("commonvoice", help="mozilla-foundation/common_voice_12_0 Punjabi [stub]")
    p_cv.add_argument("--out-dir", type=pathlib.Path, required=True)
    p_cv.add_argument("--num-samples", type=int, default=50)
    p_cv.set_defaults(func=cmd_commonvoice)

    p_ao = sub.add_parser("archiveorg", help="archive.org Gurbani Kirtan bulk [stub]")
    p_ao.add_argument("--out-dir", type=pathlib.Path, required=True)
    p_ao.set_defaults(func=cmd_archiveorg)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
