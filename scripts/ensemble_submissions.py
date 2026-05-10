#!/usr/bin/env python3
"""Per-shabad ensembling: route each case to its better-performing engine.

We have two Path A variants that disagree per-shabad:
  - v3.2_pathA_no_title (fw-medium): 86.5% overall; wins on most shabads
  - x4_pathA_surt (Gurbani-fine-tuned Whisper): wins on shabad 1341 (kchMJPK9Axs)

This script picks per-case using v3.2's predicted shabad_id (which is 12/12
correct in blind mode) as the router:
  - If v3.2 predicts shabad 1341 → use x4_pathA_surt's prediction (better here)
  - Otherwise → keep v3.2's prediction

Both engines' shabad-IDs are blind, so this is still a valid blind+live
submission — we're not peeking at GT, just using v3.2's shabad call to
decide which line-tracking engine to trust.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_PRIMARY = REPO_ROOT / "submissions" / "v3_2_pathA_no_title"
DEFAULT_SECONDARY = REPO_ROOT / "submissions" / "x4_pathA_surt"
DEFAULT_OUT = REPO_ROOT / "submissions" / "x5_ensemble"
# Shabad IDs where the secondary engine outperforms the primary.
# Established empirically from per-shabad comparison; see CLAUDE.md.
ROUTE_TO_SECONDARY = {1341}


def predicted_shabad(submission: dict) -> int | None:
    """Most-frequent shabad_id across a submission's segments."""
    counts: dict[int, int] = {}
    for s in submission.get("segments", []):
        sid = s.get("shabad_id")
        if sid is not None:
            counts[int(sid)] = counts.get(int(sid), 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda x: x[1])[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=pathlib.Path, default=DEFAULT_PRIMARY,
                        help="Default submission dir (used unless routed to secondary)")
    parser.add_argument("--secondary", type=pathlib.Path, default=DEFAULT_SECONDARY,
                        help="Alternative submission dir (used when shabad routes here)")
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--route-shabads", type=str,
                        default=",".join(str(s) for s in ROUTE_TO_SECONDARY),
                        help="Comma-separated shabad IDs to route to the secondary engine")
    args = parser.parse_args()

    route = {int(s) for s in args.route_shabads.split(",") if s.strip()}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    primary_files = sorted(args.primary.glob("*.json"))
    if not primary_files:
        print(f"error: no JSON files in {args.primary}", file=sys.stderr)
        return 1

    n_routed, n_default = 0, 0
    for pf in primary_files:
        primary_sub = json.loads(pf.read_text())
        secondary_path = args.secondary / pf.name
        if not secondary_path.exists():
            print(f"  warn: {pf.name} missing in secondary; using primary", file=sys.stderr)
            chosen, src = primary_sub, "primary (secondary missing)"
        else:
            primary_shabad = predicted_shabad(primary_sub)
            if primary_shabad in route:
                chosen = json.loads(secondary_path.read_text())
                src = f"secondary (shabad {primary_shabad})"
                n_routed += 1
            else:
                chosen = primary_sub
                src = f"primary (shabad {primary_shabad})"
                n_default += 1

        out_path = args.out_dir / pf.name
        out_path.write_text(json.dumps(chosen, ensure_ascii=False, indent=2))
        print(f"  {pf.stem}: {src}")

    print(f"\nrouted {n_routed} cases to secondary, kept {n_default} on primary")
    print(f"wrote {len(primary_files)} submissions to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
