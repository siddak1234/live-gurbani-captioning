#!/usr/bin/env python3
"""Out-of-set (OOS) evaluation harness.

Runs the engine on a directory of (audio, GT) pairs that are NOT in the paired
benchmark, then invokes the benchmark's own ``eval.py`` for scoring — so the
math is identical and we can't accidentally drift.

Why this exists: the paired benchmark is 4 shabads. A model that overfits to
those 4 looks great on the leaderboard and falls apart in deployment. OOS eval
is the honest-accuracy gate.

Data directory layout (benchmark-shaped):

    <data-dir>/
        audio/                <case_id>_16k.wav       (gitignored, large)
        test/                 <case_id>.json          (GT, committed)

GT JSON shape — same as the benchmark:
    {
        "video_id": "<case_id>",
        "shabad_id": <int>,
        "uem": {"start": <float>, "end": <float>},
        "segments": [
            {"start": <float>, "end": <float>, "line_idx": <int>,
             "verse_id": "<str>", "banidb_gurmukhi": "<str>"}, ...
        ]
    }

Usage:

    # Replay v3.2 canonical config on the paired benchmark (regression check):
    python scripts/eval_oos.py \\
        --data-dir ../live-gurbani-captioning-benchmark-v1 \\
        --pred-dir /tmp/v32_oos_replay \\
        --engine-config configs/inference/v3_2.yaml

    # Real OOS run on curated held-out shabads:
    python scripts/eval_oos.py \\
        --data-dir eval_data/oos_v1 \\
        --pred-dir submissions/oos_v1_v3_2 \\
        --engine-config configs/inference/v3_2.yaml

The audit gate: the replay-on-benchmark invocation above must score the same
number as ``benchmark/eval.py --pred submissions/v3_2_pathA_no_title``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.engine import predict, EngineConfig  # noqa: E402

BENCHMARK_EVAL = REPO_ROOT.parent / "live-gurbani-captioning-benchmark-v1" / "eval.py"


def _parse_blend(spec) -> dict[str, float] | None:
    """Accept either a dict (already parsed) or a comma:weight string."""
    if not spec:
        return None
    if isinstance(spec, dict):
        return spec
    out: dict[str, float] = {}
    for part in str(spec).split(","):
        name, w = part.split(":")
        out[name.strip()] = float(w)
    return out


def _load_engine_config(path: pathlib.Path | None) -> EngineConfig:
    """Build an EngineConfig from a YAML file (or return defaults)."""
    if path is None:
        return EngineConfig()
    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw["blend"] = _parse_blend(raw.get("blend"))
    raw["blind_blend"] = _parse_blend(raw.get("blind_blend"))
    if "asr_cache_dir" in raw and raw["asr_cache_dir"]:
        raw["asr_cache_dir"] = pathlib.Path(raw["asr_cache_dir"])
    known = set(EngineConfig().__dataclass_fields__.keys())
    unknown = set(raw) - known
    if unknown:
        print(f"Warning: ignoring unknown engine-config keys: {sorted(unknown)}", file=sys.stderr)
    return EngineConfig(**{k: v for k, v in raw.items() if k in known})


def process_one(
    gt_path: pathlib.Path,
    *,
    audio_dir: pathlib.Path,
    corpora: dict[int, list[dict]],
    pred_dir: pathlib.Path,
    blind: bool,
    engine_config: EngineConfig,
) -> bool:
    gt = json.loads(gt_path.read_text())
    video_id = gt["video_id"]
    gt_shabad_id = gt["shabad_id"]

    audio_path = audio_dir / f"{video_id}_16k.wav"
    if not audio_path.exists():
        print(f"  error: audio missing {audio_path}", file=sys.stderr)
        return False

    uem_start = float(gt.get("uem", {}).get("start", 0.0))

    try:
        result = predict(
            audio_path, corpora,
            shabad_id=None if blind else gt_shabad_id,
            uem_start=uem_start,
            config=engine_config,
        )
    except ValueError as e:
        print(f"  error: {e}", file=sys.stderr)
        return False

    if blind:
        ok = "✓" if result.shabad_id == gt_shabad_id else "✗"
        print(f"  {gt_path.stem}: blind ID predicts {result.shabad_id} "
              f"(GT {gt_shabad_id}) {ok}  top={result.blind_id_score:.1f}  "
              f"runner_up={result.blind_runner_up_score:.1f}")

    submission_segments = [
        {
            "start": s.start, "end": s.end,
            "line_idx": s.line_idx, "shabad_id": s.shabad_id,
            "verse_id": s.verse_id, "banidb_gurmukhi": s.banidb_gurmukhi,
        }
        for s in result.segments
    ]
    (pred_dir / gt_path.name).write_text(json.dumps(
        {"video_id": video_id, "segments": submission_segments},
        ensure_ascii=False, indent=2,
    ))
    print(f"  {gt_path.stem}: {result.n_chunks} chunks → {len(result.segments)} segments")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True,
                        help="Benchmark-shaped directory with audio/ and test/ subdirs")
    parser.add_argument("--pred-dir", type=pathlib.Path, required=True,
                        help="Where to write per-case prediction JSONs")
    parser.add_argument("--engine-config", type=pathlib.Path, default=None,
                        help="YAML engine config (e.g. configs/inference/v3_2.yaml). "
                             "If omitted, uses EngineConfig() defaults.")
    parser.add_argument("--corpus-dir", type=pathlib.Path,
                        default=REPO_ROOT / "corpus_cache",
                        help="BaniDB corpus cache (default: corpus_cache/)")
    parser.add_argument("--asr-cache-dir", type=pathlib.Path,
                        default=REPO_ROOT / "asr_cache",
                        help="ASR cache dir (default: asr_cache/)")
    parser.add_argument("--blind", action="store_true", default=True,
                        help="Run blind shabad ID (default: True for OOS — that's the point)")
    parser.add_argument("--oracle", action="store_true",
                        help="Use GT shabad_id (regression-check vs benchmark only)")
    parser.add_argument("--skip-scoring", action="store_true",
                        help="Generate predictions but skip the eval.py call")
    args = parser.parse_args()

    if args.oracle:
        args.blind = False

    audio_dir = (args.data_dir / "audio").resolve()
    gt_dir = (args.data_dir / "test").resolve()
    if not audio_dir.exists():
        print(f"error: {audio_dir} not found", file=sys.stderr)
        return 1
    if not gt_dir.exists():
        print(f"error: {gt_dir} not found", file=sys.stderr)
        return 1

    pred_dir = args.pred_dir.resolve()
    pred_dir.mkdir(parents=True, exist_ok=True)

    engine_cfg = _load_engine_config(args.engine_config)
    # asr_cache_dir from CLI wins over YAML default.
    if engine_cfg.asr_cache_dir is None:
        engine_cfg.asr_cache_dir = args.asr_cache_dir.resolve()

    # Load corpora.
    corpora: dict[int, list[dict]] = {}
    for cf in sorted(args.corpus_dir.resolve().glob("*.json")):
        c = json.loads(cf.read_text())
        corpora[int(c["shabad_id"])] = c["lines"]
    if not corpora:
        print(f"error: no corpus files in {args.corpus_dir}", file=sys.stderr)
        return 1

    gt_files = sorted(gt_dir.glob("*.json"))
    if not gt_files:
        print(f"error: no GT files in {gt_dir}", file=sys.stderr)
        return 1

    mode = "BLIND" if args.blind else "ORACLE"
    print(f"OOS eval: {len(gt_files)} cases in {args.data_dir} "
          f"(mode={mode}, config={args.engine_config or 'default'})\n")

    failures: list[str] = []
    for gt_file in gt_files:
        ok = process_one(
            gt_file,
            audio_dir=audio_dir, corpora=corpora,
            pred_dir=pred_dir, blind=args.blind,
            engine_config=engine_cfg,
        )
        if not ok:
            failures.append(gt_file.stem)

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        # Don't bail — still score what landed.

    print(f"\nwrote {len(gt_files) - len(failures)}/{len(gt_files)} predictions to {pred_dir}")

    if args.skip_scoring:
        return 0

    # Hand off to the benchmark scorer — single source of truth for the metric.
    if not BENCHMARK_EVAL.exists():
        print(f"\nwarning: benchmark eval.py not found at {BENCHMARK_EVAL}", file=sys.stderr)
        print(f"score it yourself with: python <path-to-benchmark>/eval.py "
              f"--pred {pred_dir} --gt {gt_dir}", file=sys.stderr)
        return 0

    print(f"\nscoring via {BENCHMARK_EVAL.name}...")
    r = subprocess.run(
        [sys.executable, str(BENCHMARK_EVAL), "--pred", str(pred_dir), "--gt", str(gt_dir)],
        capture_output=False,
    )
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
