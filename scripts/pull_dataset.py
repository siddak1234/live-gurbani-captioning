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

# Make `from src.matcher import normalize` work when this file is invoked
# directly via `python scripts/pull_dataset.py` (the Makefile's pattern).
# Without this, only the scripts/ directory is on sys.path, and src/ — at
# the repo root — isn't importable. Other top-of-repo scripts
# (finetune_path_b.py, build_training_dataset.py) do the same.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def load_benchmark_lines(benchmark_dir: pathlib.Path | None = None) -> set:
    """Load normalized canonical Gurmukhi lines from the paired benchmark.

    Why this exists: the kirtan dataset uses its own ``canonical_shabad_id``
    scheme (3-char tokens like ``"79E"``) that doesn't map to BaniDB's
    integer IDs (``4377``, ``1821``, ...). The shabad-ID holdout in
    ``configs/datasets.yaml`` is therefore a no-op for content-level
    contamination — a clip from one of the 4 benchmark shabads could
    slip through under a different ``canonical_shabad_id`` and different
    ``video_id``. The only reliable filter at the row level is to check
    the line text against the canonical Gurmukhi strings the benchmark
    publishes per segment.

    Returns a set of normalized line texts (normalize() from src.matcher
    — same canonicalization used by the inference matcher). Empty set if
    the benchmark dir is unavailable; the puller prints a warning rather
    than blocking, so the training-machine workflow (no benchmark repo
    required, per CLAUDE.md) still functions — at the cost of losing
    the content-level holdout signal for that run.
    """
    if benchmark_dir is None:
        benchmark_dir = REPO_ROOT.parent / "live-gurbani-captioning-benchmark-v1" / "test"
    if not benchmark_dir.exists():
        print(
            f"  warning: benchmark dir {benchmark_dir} not found — content-level holdout "
            f"is a no-op for this pull (video-id + shabad-id holdout still applied if configured)",
            file=sys.stderr,
        )
        return set()

    # Lazy-import normalize so the puller's startup cost doesn't bake in
    # rapidfuzz/unidecode unless content holdout is actually requested.
    from src.matcher import normalize

    lines: set = set()
    for gt_path in sorted(benchmark_dir.glob("*.json")):
        try:
            gt = json.loads(gt_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for seg in gt.get("segments", []) or []:
            text = seg.get("banidb_gurmukhi") or seg.get("text") or ""
            if not text:
                continue
            norm = normalize(text)
            if norm:
                lines.add(norm)
    return lines


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

def _parse_shards(spec: str) -> tuple[int, ...]:
    """Parse a shard spec like ``"0"``, ``"0,2,5"``, or ``"0-3,7"``.

    Returned tuple is deduped while preserving order. Used for Phase 2.5
    diversity pulls, where the first shard alone proved too narrow
    (v5_mac_baseline kept 200 clips from only 2 source videos).
    """
    out: list[int] = []
    seen: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start, end = int(start_s), int(end_s)
                if end < start:
                    raise ValueError
                vals = range(start, end + 1)
            else:
                vals = (int(part),)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"--shards must be a comma list/range of non-negative integers; got {spec!r}"
            ) from e
        for val in vals:
            if val < 0:
                raise argparse.ArgumentTypeError(
                    f"--shards must be non-negative; got {val} in {spec!r}"
                )
            if val not in seen:
                out.append(val)
                seen.add(val)
    if not out:
        raise argparse.ArgumentTypeError(f"--shards parsed to no shard indexes: {spec!r}")
    return tuple(out)


def _iter_parquet_shards(dataset_id: str, shard_idxs: Iterable[int]) -> Iterator[dict]:
    """Download selected parquet shards from an HF dataset and yield rows.

    Pulls only the requested shards, not the full dataset. Caller stops early
    via --num-samples / --max-scan. Each yielded row gets a private
    ``_source_shard`` key for data-card lineage.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(repo_id=dataset_id, repo_type="dataset")
             if f.startswith("data/train-") and f.endswith(".parquet")]
    files.sort()
    if not files:
        raise RuntimeError(f"no parquet files found in dataset {dataset_id}")

    for shard_idx in shard_idxs:
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
            rec = dict(zip(cols, row))
            rec["_source_shard"] = shard_idx
            yield rec


def _iter_first_parquet_shard(dataset_id: str, shard_idx: int = 0) -> Iterator[dict]:
    """Back-compat wrapper for tests / older callers."""
    yield from _iter_parquet_shards(dataset_id, (shard_idx,))


def _diversity_counts(manifest: list[dict]) -> dict[str, int]:
    """Count unique source-video and shabad-token diversity in a manifest."""
    return {
        "unique_videos": len({r.get("video_id") for r in manifest if r.get("video_id")}),
        "unique_shabads": len({str(r.get("shabad_id")) for r in manifest if r.get("shabad_id")}),
    }


def _check_diversity_floors(
    manifest: list[dict],
    *,
    min_unique_videos: int = 0,
    min_unique_shabads: int = 0,
) -> tuple[dict[str, int], list[str]]:
    """Return diversity counts plus human-readable floor failures.

    This is a guardrail for Phase 2.5. It intentionally runs after the pull
    and writes the data card even on failure, so the user can inspect what the
    attempted pull actually contained and decide whether to widen shards,
    raise max_scan, or change source mix.
    """
    counts = _diversity_counts(manifest)
    failures: list[str] = []
    if min_unique_videos and counts["unique_videos"] < min_unique_videos:
        failures.append(
            f"unique videos {counts['unique_videos']} < required {min_unique_videos}"
        )
    if min_unique_shabads and counts["unique_shabads"] < min_unique_shabads:
        failures.append(
            f"unique shabads {counts['unique_shabads']} < required {min_unique_shabads}"
        )
    return counts, failures


def _pull_target_met(
    manifest: list[dict],
    *,
    num_samples: int,
    min_unique_videos: int = 0,
    min_unique_shabads: int = 0,
) -> bool:
    """True when the pull has enough samples AND diversity floors pass.

    If any diversity floor is active, ``num_samples`` is a minimum, not a hard
    cap. This matters for Phase 2.5: the first failed `data-v5b` attempt hit
    1000 clips before reaching the required source-video/shabad diversity.
    Stopping there produced exactly the low-diversity failure we were trying
    to prevent.
    """
    if len(manifest) < num_samples:
        return False
    _, failures = _check_diversity_floors(
        manifest,
        min_unique_videos=min_unique_videos,
        min_unique_shabads=min_unique_shabads,
    )
    return not failures


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


def _decode_audio(audio_field) -> tuple:
    """Decode an HF Audio dict to (array, sample_rate, duration_s).

    Splitting decode from write lets us cheaply filter on duration BEFORE
    paying for the wav file write — avoids orphaned files in clips/ when
    duration bounds reject a clip. Returns ``(None, 0, 0.0)`` on any
    decode failure so the caller drops the row uniformly.
    """
    import soundfile as sf
    audio_bytes = audio_field.get("bytes") if isinstance(audio_field, dict) else None
    if not audio_bytes:
        return None, 0, 0.0
    try:
        arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        return None, 0, 0.0
    duration = float(len(arr)) / float(sr) if sr else 0.0
    return arr, int(sr), duration


def _write_audio_clip(arr, sr: int, clip_id: str, clips_dir: pathlib.Path) -> pathlib.Path:
    """Write a decoded audio array to ``clips_dir/<sanitized_id>.wav`` and
    return the path. Caller has already validated duration."""
    import soundfile as sf
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(clip_id))
    clip_path = clips_dir / f"{safe}.wav"
    sf.write(str(clip_path), arr, sr)
    return clip_path


def _write_manifest(out_dir: pathlib.Path, manifest: list[dict]) -> pathlib.Path:
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest_path


def _write_split_manifests(out_dir: pathlib.Path, splits: dict[str, list[dict]]) -> dict[str, pathlib.Path]:
    """Write ``manifest_<split>.json`` for each split, plus ``manifest.json``
    as a back-compat alias for the train split. Existing callers
    (Makefile train target, finetune script with --manifest) keep working."""
    paths: dict[str, pathlib.Path] = {}
    for split_name, records in splits.items():
        p = out_dir / f"manifest_{split_name}.json"
        p.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        paths[split_name] = p
    # Back-compat: manifest.json mirrors the train split.
    if "train" in splits:
        legacy = out_dir / "manifest.json"
        legacy.write_text(json.dumps(splits["train"], ensure_ascii=False, indent=2))
        paths["_legacy_manifest"] = legacy
    return paths


def _split_by_shabad(
    manifest: list[dict],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    diversity_warn_threshold: int = 3,
) -> dict[str, list[dict]]:
    """Zero-leakage train/val/test split with shabads (not clips) as the unit.

    Each shabad lives in exactly ONE split — clips of the same shabad never
    straddle. This is the hygiene required so the validation loss reflects
    generalization to unseen shabads, not memorization of seen lines.

    Allocation is greedy by hours, not by clip count: shabads vary widely
    in length (the longest shabad in the kirtan corpus is ~10× the median),
    so a clip-count split would be substantially off-ratio in practice.
    Per-shabad hours are summed once, shabads shuffled with ``seed``, and
    each shabad assigned to whichever split is most under its hour target.

    Returns a dict with keys "train", "val", "test" mapped to lists of
    records (subset of ``manifest``). Empty manifest → empty splits.

    Raises ``ValueError`` if any record has an empty ``shabad_id`` — the
    caller asked for shabad-level splits on data without shabad metadata,
    which is incoherent; switch to ``--split-by none`` or pull from a
    source whose records carry shabad_id.

    Prints a ``Warning:`` to stderr (does NOT raise) if val or test has
    fewer than ``diversity_warn_threshold`` unique shabads — small pulls
    are legitimately small, but a single-shabad eval set is statistically
    weak and the user should know.
    """
    import random

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    if not manifest:
        return splits

    # Validate ratios — sane defensive default even when called from Python
    # bypassing the CLI's _parse_split_ratios.
    if abs(sum(ratios) - 1.0) > 1e-3:
        raise ValueError(f"ratios must sum to 1.0; got {ratios} (sum={sum(ratios)})")

    # Validate that every record has a shabad_id — shabad-level splitting
    # is incoherent otherwise.
    missing = [i for i, r in enumerate(manifest) if not r.get("shabad_id")]
    if missing:
        raise ValueError(
            f"--split-by shabad: {len(missing)} record(s) have empty shabad_id "
            f"(first index: {missing[0]}). Use --split-by none for sources "
            f"without shabad metadata."
        )

    # Sum hours per shabad.
    shabad_hours: dict[str, float] = {}
    shabad_records: dict[str, list[dict]] = {}
    for rec in manifest:
        sid = str(rec["shabad_id"])
        shabad_hours[sid] = shabad_hours.get(sid, 0.0) + float(rec.get("duration_s") or 0.0) / 3600.0
        shabad_records.setdefault(sid, []).append(rec)

    total_hours = sum(shabad_hours.values())
    if total_hours == 0:
        # All clips have duration_s == 0 (unlikely in real data — usually means
        # the metadata wasn't populated). Fall back to equal-weight by count.
        for sid in shabad_hours:
            shabad_hours[sid] = float(len(shabad_records[sid]))
        total_hours = sum(shabad_hours.values())

    targets = {
        "train": ratios[0] * total_hours,
        "val":   ratios[1] * total_hours,
        "test":  ratios[2] * total_hours,
    }
    current = {"train": 0.0, "val": 0.0, "test": 0.0}

    # Deterministic shuffle: sort to canonicalize input order, then shuffle.
    rng = random.Random(seed)
    shabad_list = sorted(shabad_hours.keys())
    rng.shuffle(shabad_list)

    # Greedy: each shabad goes to the split with the biggest deficit.
    for sid in shabad_list:
        deficits = {s: targets[s] - current[s] for s in ("train", "val", "test")}
        chosen = max(deficits, key=lambda s: deficits[s])
        splits[chosen].extend(shabad_records[sid])
        current[chosen] += shabad_hours[sid]

    # Diversity guardrail — does NOT block, just warns.
    val_unique = {str(r["shabad_id"]) for r in splits["val"]}
    test_unique = {str(r["shabad_id"]) for r in splits["test"]}
    if len(val_unique) < diversity_warn_threshold:
        print(f"Warning: val split has only {len(val_unique)} unique shabad(s); "
              f"eval loss will be noisy. Consider increasing --num-samples.", file=sys.stderr)
    if len(test_unique) < diversity_warn_threshold:
        print(f"Warning: test split has only {len(test_unique)} unique shabad(s); "
              f"held-out score will be noisy. Consider increasing --num-samples.", file=sys.stderr)

    return splits


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
    shard_idxs = args.shards if getattr(args, "shards", None) is not None else (args.shard,)
    print(f"  shards: {','.join(str(s) for s in shard_idxs)}")

    # Content-level holdout: the kirtan dataset's canonical_shabad_id namespace
    # doesn't align with BaniDB's integer IDs, so the shabad-ID filter is a
    # no-op in practice. The only reliable contamination check at the row level
    # is to test the line text against the benchmark's canonical Gurmukhi lines.
    # Loads once per pull; empty set when benchmark dir is unavailable (training
    # machine workflow keeps functioning, with a warning).
    benchmark_lines: set = load_benchmark_lines() if apply_holdout else set()
    if apply_holdout:
        print(f"  content-holdout: {len(benchmark_lines)} canonical benchmark lines loaded")

    manifest: list[dict] = []
    n_scanned = n_kept = 0
    n_skipped_shabad = n_skipped_video = n_skipped_score = n_skipped_simran = 0
    n_skipped_dur_short = n_skipped_dur_long = 0
    n_skipped_content = 0

    for row in _iter_parquet_shards(dataset_id, shard_idxs):
        n_scanned += 1
        if n_scanned > args.max_scan:
            print(f"  hit --max-scan limit ({args.max_scan}); stopping")
            break
        if _pull_target_met(
            manifest,
            num_samples=args.num_samples,
            min_unique_videos=getattr(args, "min_unique_videos", 0),
            min_unique_shabads=getattr(args, "min_unique_shabads", 0),
        ):
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

        # Content-level holdout: drop the row if its canonical line text
        # exactly matches any line from one of the 4 benchmark shabads.
        # See load_benchmark_lines() for why this is the only correct holdout.
        if benchmark_lines:
            from src.matcher import normalize
            if normalize(text) in benchmark_lines:
                n_skipped_content += 1
                continue

        # Decode → duration filter → write. The split avoids orphaned wav
        # files when duration bounds reject a clip.
        arr, sr, duration = _decode_audio(row.get("audio"))
        if arr is None:
            continue
        if duration < args.min_duration_s:
            n_skipped_dur_short += 1
            continue
        if duration > args.max_duration_s:
            n_skipped_dur_long += 1
            continue

        clip_id = row.get("clip_id") or f"{source_key}_{len(manifest):06d}"
        clip_path = _write_audio_clip(arr, sr, clip_id, clips_dir)

        record = {
            "audio": f"clips/{clip_path.name}",
            "text": text,
            "source": source_key,
            "shabad_id": str(shabad_id) if shabad_id is not None else "",
            "video_id": video_id,
            "score": score,
            "duration_s": duration,
            "source_shard": row.get("_source_shard"),
        }
        manifest.append(record)
        n_kept += 1
        if n_kept % 10 == 0:
            print(f"  kept {n_kept}/{args.num_samples} (scanned {n_scanned})")

    print(f"\nscanned: {n_scanned}")
    print(f"kept:    {n_kept}")
    print(f"skipped: shabad={n_skipped_shabad} video={n_skipped_video} content={n_skipped_content} "
          f"score={n_skipped_score} simran={n_skipped_simran} "
          f"dur_short={n_skipped_dur_short} dur_long={n_skipped_dur_long}")

    rejections = {
        "holdout_shabad":  n_skipped_shabad,
        "holdout_video":   n_skipped_video,
        "holdout_content": n_skipped_content,
        "score_low":       n_skipped_score,
        "simran":          n_skipped_simran,
        "dur_short":       n_skipped_dur_short,
        "dur_long":        n_skipped_dur_long,
    }
    diversity_counts, diversity_failures = _check_diversity_floors(
        manifest,
        min_unique_videos=getattr(args, "min_unique_videos", 0),
        min_unique_shabads=getattr(args, "min_unique_shabads", 0),
    )
    print(f"diversity: videos={diversity_counts['unique_videos']} "
          f"shabads={diversity_counts['unique_shabads']}")

    splits = None
    if getattr(args, "split_by", "none") == "shabad":
        # argparse already parsed via type=_parse_split_ratios → tuple of 3 floats.
        ratios = args.split_ratios
        splits = _split_by_shabad(manifest, ratios=ratios, seed=args.split_seed)
        paths = _write_split_manifests(out_dir, splits)
        print(f"split ({ratios[0]}/{ratios[1]}/{ratios[2]}, seed={args.split_seed}):")
        for split_name in ("train", "val", "test"):
            hrs = sum(float(r.get("duration_s") or 0.0) for r in splits[split_name]) / 3600.0
            n_shabads = len({str(r["shabad_id"]) for r in splits[split_name]})
            print(f"  {split_name}: {len(splits[split_name])} clips, {hrs:.2f}h, "
                  f"{n_shabads} unique shabads -> {paths[split_name].name}")
        print(f"manifest: {paths['_legacy_manifest']} (= manifest_train.json, back-compat)")
    else:
        manifest_path = _write_manifest(out_dir, manifest)
        print(f"manifest: {manifest_path}")

    # Data card lives alongside the manifest — human-readable lineage doc.
    card_md = _build_data_card(
        out_dir=out_dir,
        source_key=source_key,
        source_id=dataset_id,
        args=args,
        splits=splits,
        manifest=manifest,
        rejections=rejections,
        holdout_shabads=holdout_shabads,
        holdout_videos=holdout_videos,
        apply_holdout=apply_holdout,
        diversity_counts=diversity_counts,
        diversity_failures=diversity_failures,
    )
    card_path = out_dir / "data_card.md"
    card_path.write_text(card_md, encoding="utf-8")
    print(f"data_card: {card_path}")
    if diversity_failures:
        for failure in diversity_failures:
            print(f"error: diversity floor failed: {failure}", file=sys.stderr)
        print("hint: widen --shards, raise --max-scan, or lower the floor for a diagnostic pull.",
              file=sys.stderr)
        return 2
    return 0


def _build_data_card(
    *,
    out_dir: pathlib.Path,
    source_key: str,
    source_id: str,
    args,
    splits: dict | None,
    manifest: list[dict],
    rejections: dict[str, int],
    holdout_shabads: set,
    holdout_videos: set,
    apply_holdout: bool,
    diversity_counts: dict[str, int] | None = None,
    diversity_failures: list[str] | None = None,
) -> str:
    """Generate ``data_card.md`` content for a pull. Pure: caller writes to disk.

    The data card is the human-readable lineage doc for a pull — what we
    scraped, what we kept, what we dropped, and how the splits look. It
    lives alongside ``manifest.json`` so anyone inspecting an adapter can
    answer "what did this train on?" without re-running the pull.

    Sections:
      * Frontmatter: source HF repo, pull config, holdout policy
      * Split summary table (when --split-by shabad) OR single-manifest stats
      * Rejection counts table (every reason a row could be dropped)
      * Top shabads in train (by hours) — sanity-check class imbalance
      * Diversity guardrail (warns when val/test has <3 unique shabads)
    """
    import datetime as _dt

    generated = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # --- header ---
    lines = [
        f"# Data card — {out_dir.name}",
        "",
        f"**Generated:** {generated}  ",
        f"**Source:** `{source_id}` (`{source_key}` in configs/datasets.yaml)  ",
        f"**Pull config:** num_samples={args.num_samples}, min_score={args.min_score}, "
        f"min_duration_s={args.min_duration_s}, max_duration_s={args.max_duration_s}, "
        f"shards={getattr(args, 'shards', None) or (getattr(args, 'shard', 0),)}, "
        f"min_unique_videos={getattr(args, 'min_unique_videos', 0)}, "
        f"min_unique_shabads={getattr(args, 'min_unique_shabads', 0)}, "
        f"split_by={getattr(args, 'split_by', 'none')}, "
        f"split_seed={getattr(args, 'split_seed', 'n/a')}  ",
        "",
        "**Holdout (configs/datasets.yaml):**",
    ]
    if apply_holdout:
        shabad_strs = sorted(s for s in holdout_shabads if isinstance(s, str))
        lines.append(f"- Shabad IDs: {', '.join(shabad_strs) or '(none)'}")
        lines.append(f"- Video IDs: {', '.join(sorted(holdout_videos)) or '(none)'}")
    else:
        lines.append("- Not enforced for this source (general Punjabi or unlabeled audio)")
    lines.append("")

    # --- split summary OR single-manifest summary ---
    if splits is not None:
        lines += [
            "## Split summary",
            "",
            "| Split | Clips | Hours | Unique shabads | Unique videos |",
            "|---|---|---|---|---|",
        ]
        for split_name in ("train", "val", "test"):
            recs = splits[split_name]
            hrs = sum(float(r.get("duration_s") or 0.0) for r in recs) / 3600.0
            n_shabads = len({str(r["shabad_id"]) for r in recs if r.get("shabad_id")})
            n_videos = len({r.get("video_id") for r in recs if r.get("video_id")})
            lines.append(f"| {split_name} | {len(recs)} | {hrs:.3f} | {n_shabads} | {n_videos} |")
        lines.append("")
    else:
        hrs = sum(float(r.get("duration_s") or 0.0) for r in manifest) / 3600.0
        n_shabads = len({str(r["shabad_id"]) for r in manifest if r.get("shabad_id")})
        n_videos = len({r.get("video_id") for r in manifest if r.get("video_id")})
        lines += [
            "## Manifest summary",
            "",
            f"- Clips: {len(manifest)}",
            f"- Hours: {hrs:.3f}",
            f"- Unique shabads: {n_shabads}",
            f"- Unique videos: {n_videos}",
            "",
        ]

    # --- rejections ---
    lines += [
        "## Rejection counts",
        "",
        "| Reason | Count |",
        "|---|---|",
    ]
    for reason in ("score_low", "dur_short", "dur_long",
                   "holdout_shabad", "holdout_video", "holdout_content", "simran"):
        lines.append(f"| {reason} | {rejections.get(reason, 0)} |")
    lines.append("")

    # --- top shabads (in train if split, else in manifest) ---
    pool = splits["train"] if splits is not None else manifest
    if pool:
        per_shabad_hours: dict[str, float] = {}
        per_shabad_count: dict[str, int] = {}
        for r in pool:
            sid = str(r.get("shabad_id") or "")
            if not sid:
                continue
            per_shabad_hours[sid] = per_shabad_hours.get(sid, 0.0) + float(r.get("duration_s") or 0.0) / 3600.0
            per_shabad_count[sid] = per_shabad_count.get(sid, 0) + 1
        top = sorted(per_shabad_hours.items(), key=lambda kv: kv[1], reverse=True)[:10]
        section_title = "Top shabads in train (by hours)" if splits else "Top shabads (by hours)"
        lines += [f"## {section_title}", ""]
        for i, (sid, hrs) in enumerate(top, 1):
            lines.append(f"{i}. shabad `{sid}` — {hrs:.3f} h ({per_shabad_count[sid]} clips)")
        lines.append("")

    # --- training-pull diversity floor (Phase 2.5) ---
    min_videos = int(getattr(args, "min_unique_videos", 0) or 0)
    min_shabads = int(getattr(args, "min_unique_shabads", 0) or 0)
    if min_videos or min_shabads:
        counts = diversity_counts or _diversity_counts(manifest)
        failures = diversity_failures or []
        lines += ["## Training diversity gate", ""]
        lines.append(f"- Unique videos: {counts['unique_videos']} (floor: {min_videos})")
        lines.append(f"- Unique shabads: {counts['unique_shabads']} (floor: {min_shabads})")
        if failures:
            lines.append("- Status: FAIL")
            for failure in failures:
                lines.append(f"  - {failure}")
        else:
            lines.append("- Status: PASS")
        lines.append("")

    # --- diversity guardrail status ---
    if splits is not None:
        val_unique = len({str(r["shabad_id"]) for r in splits["val"] if r.get("shabad_id")})
        test_unique = len({str(r["shabad_id"]) for r in splits["test"] if r.get("shabad_id")})
        lines += ["## Diversity guardrail", ""]
        flag = "⚠️ " if val_unique < 3 else ""
        lines.append(f"- {flag}val unique shabads: {val_unique} (warn if < 3)")
        flag = "⚠️ " if test_unique < 3 else ""
        lines.append(f"- {flag}test unique shabads: {test_unique} (warn if < 3)")
        lines.append("")

    return "\n".join(lines)


def _parse_split_ratios(s: str) -> tuple[float, float, float]:
    """Parse "0.8,0.1,0.1" → (0.8, 0.1, 0.1). Tolerates whitespace."""
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--split-ratios must be 3 comma-separated floats; got {len(parts)}: {s!r}"
        )
    total = sum(parts)
    if abs(total - 1.0) > 1e-3:
        raise argparse.ArgumentTypeError(
            f"--split-ratios must sum to 1.0 (±1e-3); got {total}: {s!r}"
        )
    return parts[0], parts[1], parts[2]


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
    p.add_argument("--shards", type=_parse_shards, default=None,
                   help="Comma/range shard list to scan, e.g. 0,2,4 or 0-9. "
                        "Overrides --shard. Use for diversity pulls.")
    p.add_argument("--min-unique-videos", type=int, default=0,
                   help="Fail after writing the data card unless the kept manifest has at least "
                        "this many source videos (default 0 = no floor).")
    p.add_argument("--min-unique-shabads", type=int, default=0,
                   help="Fail after writing the data card unless the kept manifest has at least "
                        "this many shabad tokens (default 0 = no floor).")
    # -- Duration filter (Phase 1.C) --
    # Whisper's encoder is fixed at 30s; longer clips get truncated, which
    # corrupts the trailing label. Floor at 1s drops near-silent fragments.
    p.add_argument("--min-duration-s", type=float, default=1.0,
                   help="Drop clips shorter than this (default 1.0s).")
    p.add_argument("--max-duration-s", type=float, default=30.0,
                   help="Drop clips longer than this (default 30.0s — Whisper's encoder ceiling).")
    # -- Split (Phase 1.B) --
    # Default is "none" for back-compat: existing Makefile train target reads
    # manifest.json directly. Switch to "shabad" for zero-leakage train/val/test.
    p.add_argument("--split-by", choices=["none", "shabad"], default="none",
                   help="If 'shabad', emit manifest_{train,val,test}.json with no shabad "
                        "appearing in more than one split. 'none' keeps the single manifest.json. "
                        "Default: none.")
    p.add_argument("--split-ratios", type=_parse_split_ratios, default=(0.8, 0.1, 0.1),
                   help="Comma-separated train,val,test ratios (must sum to 1.0). Default 80/10/10.")
    p.add_argument("--split-seed", type=int, default=42,
                   help="Seed for the shabad shuffle that drives split assignment.")


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
