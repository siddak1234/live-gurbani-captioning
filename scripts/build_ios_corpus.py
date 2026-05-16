#!/usr/bin/env python3
"""Generate the on-device SGGS corpus JSON for the iOS bundle.

Reads from ``corpus_cache/*.json`` (cached BaniDB data, populated by
``scripts/build_corpus.py``) and emits a single condensed JSON array at
``ios/Sources/GurbaniCaptioning/Resources/shabads.json``. The Swift package
ships this file inside the app bundle; ``ShabadCorpus.loadFromBundle()``
decodes it at launch.

Why we don't commit the JSON
----------------------------
The full corpus is ~40-60 MB. Keeping it as a generated artifact (and ignored
by git) keeps the repo small, avoids merge conflicts on data updates, and
lets the build always pick up the freshest BaniDB cache.

Schema
------
The output is a JSON array of shabads. Each shabad mirrors corpus_cache:

    [
      {
        "shabad_id": 4377,
        "lines": [
          {
            "line_idx": 0,
            "verse_id": "ABCD",
            "banidb_gurmukhi": "...",
            "transliteration_english": "..."
          },
          ...
        ]
      },
      ...
    ]

Usage:

    # Refresh the BaniDB cache first if needed:
    python scripts/build_corpus.py

    # Then materialize the iOS bundle resource:
    python scripts/build_ios_corpus.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus_cache"
DEFAULT_OUT = REPO_ROOT / "ios" / "Sources" / "GurbaniCaptioning" / "Resources" / "shabads.json"

# Keys we keep in the on-device JSON. Anything else gets stripped to shrink
# the bundle. If iOS adds a feature that needs more fields, extend this list
# AND the Swift ShabadLine struct in Models.swift.
LINE_KEYS_KEPT = {"line_idx", "verse_id", "banidb_gurmukhi", "transliteration_english"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=pathlib.Path, default=DEFAULT_CORPUS_DIR,
                        help=f"BaniDB corpus cache (default: {DEFAULT_CORPUS_DIR.relative_to(REPO_ROOT)})")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUT,
                        help="Output JSON path")
    parser.add_argument("--pretty", action="store_true",
                        help="Indent the JSON output for human inspection (larger bundle)")
    args = parser.parse_args()

    corpus_dir = args.corpus_dir.resolve()
    if not corpus_dir.exists():
        print(f"error: corpus directory not found: {corpus_dir}", file=sys.stderr)
        print("       run scripts/build_corpus.py first to populate it.", file=sys.stderr)
        return 1

    shabad_files = sorted(corpus_dir.glob("*.json"))
    if not shabad_files:
        print(f"error: no shabad files in {corpus_dir}", file=sys.stderr)
        return 1

    out: list[dict] = []
    n_lines = 0
    for f in shabad_files:
        try:
            shabad = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            print(f"  warning: skipping malformed {f.name}: {e}", file=sys.stderr)
            continue

        condensed_lines = []
        for ln in shabad.get("lines", []):
            condensed_lines.append({k: ln.get(k) for k in LINE_KEYS_KEPT if k in ln})
        out.append({
            "shabad_id": int(shabad["shabad_id"]),
            "lines": condensed_lines,
        })
        n_lines += len(condensed_lines)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.pretty:
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        # Compact form: no whitespace, ~40% smaller bundle.
        args.output.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))

    size_mb = args.output.stat().st_size / 1_000_000
    print(f"wrote {len(out)} shabads / {n_lines} lines → {args.output} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
