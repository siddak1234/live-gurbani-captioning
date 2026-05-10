#!/usr/bin/env python3
"""Stage 1b: build the BaniDB corpus cache.

Reads unique `shabad_id`s from the paired benchmark's test/*.json, fetches
each shabad's lines from `api.banidb.com/v2/shabads/{id}`, and writes a
parsed line list to `corpus_cache/<shabad_id>.json`. Idempotent — skips
shabads already cached.

Output shape per file:

  {
    "shabad_id": 4377,
    "lines": [
      {
        "line_idx": 0,
        "verse_id": 52521,
        "banidb_gurmukhi": "...",
        "banidb_larivaar": "...",
        "transliteration_english": "..."
      }, ...
    ]
  }
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_GT_DIR = REPO_ROOT.parent / "live-gurbani-captioning-benchmark-v1" / "test"
DEFAULT_CACHE_DIR = REPO_ROOT / "corpus_cache"
BANIDB_API = "https://api.banidb.com/v2"


def collect_shabad_ids(gt_dir: pathlib.Path) -> list[int]:
    ids: set[int] = set()
    for f in sorted(gt_dir.glob("*.json")):
        gt = json.loads(f.read_text())
        if "shabad_id" in gt:
            ids.add(int(gt["shabad_id"]))
    return sorted(ids)


def fetch_shabad(shabad_id: int, timeout: int = 15) -> dict | None:
    url = f"{BANIDB_API}/shabads/{shabad_id}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  error: fetch shabad {shabad_id} failed: {e}", file=sys.stderr)
        return None


def parse_verses(data: dict) -> list[dict]:
    lines: list[dict] = []
    for i, v in enumerate(data.get("verses", [])):
        verse = v.get("verse", {}) or {}
        larivaar = v.get("larivaar", {}) or {}
        translit = v.get("transliteration", {}) or {}
        lines.append({
            "line_idx": i,
            "verse_id": v.get("verseId"),
            "banidb_gurmukhi": verse.get("unicode") or verse.get("gurmukhi") or "",
            "banidb_larivaar": larivaar.get("unicode") or larivaar.get("gurmukhi") or "",
            "transliteration_english": translit.get("english") or translit.get("en") or "",
        })
    return lines


def build_one(shabad_id: int, cache_dir: pathlib.Path) -> bool:
    out_path = cache_dir / f"{shabad_id}.json"
    if out_path.exists():
        print(f"skip: {out_path.name} already exists")
        return True
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"fetching shabad {shabad_id}...")
    data = fetch_shabad(shabad_id)
    if data is None:
        return False

    lines = parse_verses(data)
    if not lines:
        print(f"  error: shabad {shabad_id} returned no verses", file=sys.stderr)
        return False

    payload = {"shabad_id": shabad_id, "lines": lines}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"done: {out_path.relative_to(REPO_ROOT)} ({len(lines)} lines)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=pathlib.Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--cache-dir", type=pathlib.Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    shabad_ids = collect_shabad_ids(args.gt_dir.resolve())
    if not shabad_ids:
        print(f"error: no shabad_ids found in {args.gt_dir}", file=sys.stderr)
        return 1

    print(f"found {len(shabad_ids)} unique shabad_id(s): {shabad_ids}\n")
    cache_dir = args.cache_dir.resolve()

    failures: list[int] = []
    for i, sid in enumerate(shabad_ids):
        if not build_one(sid, cache_dir):
            failures.append(sid)
        if i < len(shabad_ids) - 1:
            time.sleep(0.2)  # be polite to BaniDB

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1
    print(f"\nall {len(shabad_ids)} shabad(s) cached in {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
