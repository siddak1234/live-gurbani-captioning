#!/usr/bin/env python3
"""Pull user corrections from Supabase into a labeled training dataset.

Phase 6b, option 1 — *collect-and-label*. Pulls correction rows from the
`gurbani-captioning` Supabase project, downloads the referenced audio clips,
resolves the canonical Gurmukhi text for each corrected shabad, enforces the
benchmark-shabad holdout, and writes a reviewable dataset under
`training_data/<batch>/` (manifest + clips + data_card.md).

IMPORTANT — what this dataset is (and isn't):
  Each correction gives (audio clip ~30s, corrected SHABAD). `hardNegPos`
  corrections carry no line index, so the manifest `text` is the **full
  canonical shabad** (`text_granularity: "shabad"`), NOT a line-aligned ASR
  transcript. Treat this as labeled audio for later use (forced alignment to
  derive per-line text, or a shabad-ID signal) behind a human review gate —
  do not feed it straight into a line-level ASR fine-tune.

Read-only by default; `--mark-exported` flips server `export_status` to
`exported` for the pulled rows (run only after review).

Environment:
  SUPABASE_SERVICE_ROLE_KEY   (required) — the project's service_role/secret
                              key from the Supabase dashboard. NEVER commit it.
  SUPABASE_URL                (optional) — defaults to the project URL.

Usage:
  SUPABASE_SERVICE_ROLE_KEY=... python scripts/pull_corrections.py \
      --out-dir training_data/corrections_v1
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://mlovvqoiuuihmymkypfm.supabase.co"
DATASETS_CONFIG = REPO_ROOT / "configs" / "datasets.yaml"
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus_cache"
DEFAULT_OUT_DIR = REPO_ROOT / "training_data" / "corrections_v1"
BUCKET = "correction-audio"


# --------------------------------------------------------------------------
# Pure helpers (unit-tested; no network / no env)
# --------------------------------------------------------------------------

def load_holdout_shabad_ids(config_path: pathlib.Path) -> set[str]:
    """Union of benchmark-holdout shabad ids across all enforced sources."""
    import yaml
    data = yaml.safe_load(config_path.read_text()) or {}
    ids: set[str] = set()
    for src in (data.get("sources") or {}).values():
        holdout = (src or {}).get("holdout") or {}
        if holdout.get("enforce"):
            ids |= {str(x) for x in (holdout.get("shabad_ids") or [])}
    return ids


def is_held_out(row: dict, holdout_ids: set[str]) -> bool:
    """True if a correction touches a benchmark shabad (never train on those)."""
    gt = row.get("ground_truth_shabad_id")
    pred = row.get("predicted_shabad_id")
    return (gt is not None and str(gt) in holdout_ids) or \
           (pred is not None and str(pred) in holdout_ids)


def shabad_text_from_lines(lines: list[dict]) -> str:
    """Join a shabad's canonical Gurmukhi lines into one text blob."""
    parts = [(ln.get("banidb_gurmukhi") or "").strip() for ln in lines]
    return " ".join(p for p in parts if p)


def build_record(row: dict, audio_relpath: str, text: str) -> dict:
    """One manifest record. `text` is the FULL shabad (see module docstring)."""
    return {
        "audio": audio_relpath,
        "text": text,
        "text_granularity": "shabad",       # not line-aligned — review/align before ASR
        "shabad_id": row.get("ground_truth_shabad_id"),
        "line_idx": row.get("ground_truth_line_idx"),
        "predicted_shabad_id": row.get("predicted_shabad_id"),
        "kind": row.get("kind"),
        "correction_id": row.get("id"),
        "device_id": row.get("device_id"),
        "model_version": row.get("model_version"),
        "created_at": row.get("created_at"),
        "audio_start": row.get("audio_start"),
        "audio_end": row.get("audio_end"),
        "source": "correction",
        "review_status": "unreviewed",
    }


def data_card(manifest: list[dict], dropped_holdout: int, missing_text: int) -> str:
    """Human-readable summary of a pull (mirrors the repo's data_card convention)."""
    per_shabad: dict[str, int] = {}
    total_seconds = 0.0
    for r in manifest:
        per_shabad[str(r.get("shabad_id"))] = per_shabad.get(str(r.get("shabad_id")), 0) + 1
        start, end = r.get("audio_start") or 0.0, r.get("audio_end") or 0.0
        if end > start:
            total_seconds += (end - start)
    lines = [
        "# Corrections pull — data card",
        "",
        f"- records: **{len(manifest)}**",
        f"- approx audio: **{total_seconds / 3600:.3f} h** (from clip windows)",
        f"- dropped (benchmark holdout): {dropped_holdout}",
        f"- missing canonical text: {missing_text}",
        f"- unique shabads: {len(per_shabad)}",
        "",
        "## per-shabad record counts",
    ]
    for sid, n in sorted(per_shabad.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {sid}: {n}")
    lines += [
        "",
        "> `text` is the full canonical shabad (`text_granularity: shabad`), not a",
        "> line-aligned transcript. Review + align before line-level ASR training.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def _request(url: str, key: str, method: str = "GET",
             data: bytes | None = None, extra_headers: dict | None = None) -> tuple[bytes, dict]:
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, method=method, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read(), dict(r.headers)


def fetch_corrections(base_url: str, key: str, only_new: bool = True) -> list[dict]:
    query = "select=*&order=created_at.asc"
    if only_new:
        query += "&export_status=eq.new"
    body, _ = _request(f"{base_url}/rest/v1/corrections?{query}", key)
    return json.loads(body)


def download_audio(base_url: str, key: str, storage_key: str, dest: pathlib.Path) -> bool:
    url = f"{base_url}/storage/v1/object/{BUCKET}/{urllib.parse.quote(storage_key)}"
    try:
        body, _ = _request(url, key)
    except urllib.error.HTTPError as e:
        print(f"  warn: audio download failed for {storage_key}: {e}", file=sys.stderr)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return True


def resolve_text(shabad_id, corpus_dir: pathlib.Path, allow_fetch: bool = True) -> str | None:
    if shabad_id is None:
        return None
    cache = corpus_dir / f"{shabad_id}.json"
    if not cache.exists() and allow_fetch:
        try:
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
            from build_corpus import build_one
            build_one(int(shabad_id), corpus_dir)
        except Exception as e:  # noqa: BLE001 — best-effort enrichment
            print(f"  warn: corpus fetch failed for {shabad_id}: {e}", file=sys.stderr)
    if cache.exists():
        return shabad_text_from_lines(json.loads(cache.read_text()).get("lines") or [])
    return None


def mark_exported(base_url: str, key: str, ids: list[str]) -> None:
    for cid in ids:
        url = f"{base_url}/rest/v1/corrections?id=eq.{urllib.parse.quote(str(cid))}"
        data = json.dumps({"export_status": "exported"}).encode("utf-8")
        _request(url, key, method="PATCH", data=data,
                 extra_headers={"Content-Type": "application/json", "Prefer": "return=minimal"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--url", default=os.environ.get("SUPABASE_URL", DEFAULT_URL))
    parser.add_argument("--corpus-dir", type=pathlib.Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--all", action="store_true",
                        help="include already-exported rows (default: only export_status=new)")
    parser.add_argument("--no-fetch-corpus", action="store_true",
                        help="don't hit BaniDB for missing shabad text")
    parser.add_argument("--mark-exported", action="store_true",
                        help="after review: flip export_status=exported for the pulled rows")
    args = parser.parse_args()

    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not key:
        print("error: SUPABASE_SERVICE_ROLE_KEY not set. Get the service_role key from the "
              "Supabase dashboard (Settings → API) and export it; never commit it.", file=sys.stderr)
        return 2

    holdout = load_holdout_shabad_ids(DATASETS_CONFIG)
    rows = fetch_corrections(args.url, key, only_new=not args.all)
    print(f"pulled {len(rows)} correction(s) from {args.url}")

    out_dir = args.out_dir.resolve()
    manifest: list[dict] = []
    dropped = missing_text = 0
    pulled_ids: list[str] = []

    for row in rows:
        if is_held_out(row, holdout):
            dropped += 1
            continue
        audio_key = row.get("audio_path")
        if not audio_key:           # only audio-bearing corrections are training data
            continue
        cid = row.get("id")
        ext = pathlib.Path(audio_key).suffix.lstrip(".") or "m4a"
        rel = f"clips/{cid}.{ext}"
        if not download_audio(args.url, key, audio_key, out_dir / rel):
            continue
        text = resolve_text(row.get("ground_truth_shabad_id"), args.corpus_dir,
                            allow_fetch=not args.no_fetch_corpus)
        if not text:
            missing_text += 1
        manifest.append(build_record(row, rel, text or ""))
        pulled_ids.append(cid)

    leaked = [r for r in manifest if str(r.get("shabad_id")) in holdout]
    if leaked:
        print(f"error: {len(leaked)} holdout shabad(s) leaked into the manifest — aborting",
              file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (out_dir / "data_card.md").write_text(data_card(manifest, dropped, missing_text))
    print(f"wrote {len(manifest)} record(s) → {out_dir / 'manifest.json'} "
          f"(dropped {dropped} holdout, {missing_text} missing text)")

    if args.mark_exported and pulled_ids:
        mark_exported(args.url, key, pulled_ids)
        print(f"marked {len(pulled_ids)} correction(s) export_status=exported")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
