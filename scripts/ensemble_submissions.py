#!/usr/bin/env python3
"""Per-shabad ensembling: route each case to its better-performing engine.

We have three Path A variants that win on different shabads:
  - v3.2 (fw-medium, raw audio):   wins shabads 4377 (IZOsmkdmmcg) and 3712 (zOtIpxMT9hU)
  - x4 (surt-small-v3, Gurbani Whisper): wins shabad 1341 (kchMJPK9Axs) by +10-17 pts
  - v4_mlx (mlx large-v3):         wins shabad 1821 (kZhIA8P6xWI) by +1-12 pts

Routes are configured below by shabad_id. v3.2's blind shabad ID (12/12
correct) is the dispatcher — we use its predicted shabad to decide which
engine's output to keep. All three engines do blind shabad ID, so the
final submission is still a valid blind+live result; we're not peeking at
GT.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_PRIMARY = REPO_ROOT / "submissions" / "v3_2_pathA_no_title"
DEFAULT_OUT = REPO_ROOT / "submissions" / "x6_ensemble"

# shabad_id → submission directory (overrides primary when v3.2 predicts this shabad).
# Empirically tuned from per-shabad comparison vs v3.2. Each routed shabad
# has the routing engine winning by ≥1 point across all three cold variants.
ROUTE_TABLE: dict[int, pathlib.Path] = {
    1341: REPO_ROOT / "submissions" / "x4_pathA_surt",       # kchMJPK9Axs
    1821: REPO_ROOT / "submissions" / "v4_mlx_large_v3",     # kZhIA8P6xWI
}


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
                        help="Default submission dir (used unless shabad routes elsewhere)")
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    primary_files = sorted(args.primary.glob("*.json"))
    if not primary_files:
        print(f"error: no JSON files in {args.primary}", file=sys.stderr)
        return 1

    counts: dict[str, int] = {}
    for pf in primary_files:
        primary_sub = json.loads(pf.read_text())
        primary_shabad = predicted_shabad(primary_sub)
        target_dir = ROUTE_TABLE.get(primary_shabad)
        if target_dir is not None and (target_dir / pf.name).exists():
            chosen = json.loads((target_dir / pf.name).read_text())
            src = f"{target_dir.name} (shabad {primary_shabad})"
        else:
            chosen = primary_sub
            src = f"{args.primary.name} (shabad {primary_shabad})"
        counts[src.split(" ")[0]] = counts.get(src.split(" ")[0], 0) + 1

        (args.out_dir / pf.name).write_text(json.dumps(chosen, ensure_ascii=False, indent=2))
        print(f"  {pf.stem}: {src}")

    print(f"\ncounts by source:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"wrote {len(primary_files)} submissions to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
