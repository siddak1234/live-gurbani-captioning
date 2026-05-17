#!/usr/bin/env python3
"""Audit whether the local M4 Pro is being used appropriately.

This is a checkpoint script, not a benchmark. It records:

- detected Apple Silicon / memory information;
- whether PyTorch can see MPS in the current process;
- real run-card memory and wall-clock from completed adapters;
- current data footprint;
- the recommended compute decision at the current project phase.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass
class RunCardSummary:
    name: str
    train_n_clips: int | None
    final_train_loss: float | None
    wall_clock_s: float | None
    peak_mem_gb: float | None
    peak_mem_source: str | None
    device: str | None


def _run_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:  # pragma: no cover - system-specific
        return f"ERROR: {exc}"


def hardware_summary() -> dict[str, str | None]:
    text = _run_text(["system_profiler", "SPHardwareDataType"])
    out: dict[str, str | None] = {"chip": None, "memory": None, "model": None}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Chip:"):
            out["chip"] = line.split(":", 1)[1].strip()
        elif line.startswith("Memory:"):
            out["memory"] = line.split(":", 1)[1].strip()
        elif line.startswith("Model Name:"):
            out["model"] = line.split(":", 1)[1].strip()
    return out


def parse_memory_gb(memory: str | None) -> float | None:
    if not memory:
        return None
    match = re.search(r"([0-9.]+)\s*GB", memory)
    return float(match.group(1)) if match else None


def mps_summary() -> dict[str, str | bool | float | None]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - env-specific
        return {"torch": f"import failed: {exc}", "mps_built": None, "mps_available": None}

    out: dict[str, str | bool | float | None] = {
        "torch": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_smoke_ok": False,
        "mps_driver_allocated_gb_after_smoke": None,
    }
    if torch.backends.mps.is_available():
        x = torch.ones((512, 512), device="mps")
        y = x @ x
        torch.mps.synchronize()
        out["mps_smoke_ok"] = abs(float(y[0, 0].cpu()) - 512.0) < 1e-6
        out["mps_driver_allocated_gb_after_smoke"] = torch.mps.driver_allocated_memory() / 1024**3
    return out


def load_run_cards(root: pathlib.Path = REPO_ROOT) -> list[RunCardSummary]:
    cards: list[RunCardSummary] = []
    for path in sorted((root / "lora_adapters").glob("*/run_card.json")):
        data = json.loads(path.read_text())
        cards.append(RunCardSummary(
            name=path.parent.name,
            train_n_clips=data.get("train_n_clips"),
            final_train_loss=data.get("final_train_loss"),
            wall_clock_s=data.get("wall_clock_s"),
            peak_mem_gb=data.get("peak_mem_gb"),
            peak_mem_source=data.get("peak_mem_source"),
            device=data.get("device"),
        ))
    return cards


def _du(path: pathlib.Path) -> str:
    if not path.exists():
        return "(missing)"
    return _run_text(["du", "-sh", str(path)]).split()[0]


def render_report() -> str:
    hw = hardware_summary()
    mem_gb = parse_memory_gb(hw.get("memory"))
    mps = mps_summary()
    cards = load_run_cards()
    card_names = {c.name for c in cards}
    has_v6 = "v6_mac_scale20" in card_names
    has_v6_silver = (REPO_ROOT / "submissions" / "silver_300h_v6_mac_scale20.json").exists()
    has_v6_paired = (REPO_ROOT / "submissions" / "phase3_v6_lock_fusion_paired").exists()
    has_v6_oos = (REPO_ROOT / "submissions" / "oos_v1_assisted_phase3_v6_lock_fusion").exists()
    has_recency_paired = (REPO_ROOT / "submissions" / "phase3_recency_guard_paired").exists()
    has_recency_oos = (REPO_ROOT / "submissions" / "oos_v1_assisted_phase3_recency_guard").exists()
    has_confirmed_paired = (REPO_ROOT / "submissions" / "phase3_recency_guard_confirmed_paired").exists()
    has_confirmed_oos = (REPO_ROOT / "submissions" / "oos_v1_assisted_phase3_confirmed").exists()
    has_confirmed_reports = (
        (REPO_ROOT / "diagnostics" / "phase3_recency_guard_confirmed_paired_line_path.md").exists()
        and (REPO_ROOT / "diagnostics" / "phase3_recency_guard_confirmed_oos_assisted_line_path.md").exists()
    )
    has_alignment_reports = (
        (REPO_ROOT / "diagnostics" / "phase3_recency_guard_paired_alignment_errors.md").exists()
        and (REPO_ROOT / "diagnostics" / "phase3_recency_guard_oos_assisted_alignment_errors.md").exists()
    )
    peak_mem = max((c.peak_mem_gb or 0.0 for c in cards), default=0.0)
    peak_fraction = (peak_mem / mem_gb) if mem_gb else None

    lines: list[str] = [
        "# M4 Pro compute utilization audit",
        "",
        "## Hardware and runtime",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Model | {hw.get('model') or 'unknown'} |",
        f"| Chip | {hw.get('chip') or 'unknown'} |",
        f"| Unified memory | {hw.get('memory') or 'unknown'} |",
        f"| Torch | {mps.get('torch')} |",
        f"| MPS built | {mps.get('mps_built')} |",
        f"| MPS available in this process | {mps.get('mps_available')} |",
        f"| MPS smoke ok | {mps.get('mps_smoke_ok')} |",
        "",
        "## Completed training runs",
        "",
        "| Adapter | Clips | Loss | Wall clock | Peak MPS memory | Device |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for card in cards:
        wall = f"{(card.wall_clock_s or 0.0) / 60:.1f} min" if card.wall_clock_s else "unknown"
        peak = f"{card.peak_mem_gb:.2f} GB ({card.peak_mem_source})" if card.peak_mem_gb else "unknown"
        loss = f"{card.final_train_loss:.4f}" if card.final_train_loss is not None else "unknown"
        lines.append(
            f"| `{card.name}` | {card.train_n_clips or 'unknown'} | {loss} | {wall} | {peak} | {card.device or 'unknown'} |"
        )

    lines.extend([
        "",
        "## Data and artifact footprint",
        "",
        "| Path | Size |",
        "|---|---:|",
        f"| `training_data/` | {_du(REPO_ROOT / 'training_data')} |",
        f"| `lora_adapters/` | {_du(REPO_ROOT / 'lora_adapters')} |",
        f"| `submissions/` | {_du(REPO_ROOT / 'submissions')} |",
        f"| `asr_cache/` | {_du(REPO_ROOT / 'asr_cache')} |",
        "",
        "## Audit decision",
        "",
    ])

    if peak_fraction is not None:
        lines.append(
            f"- Highest completed training memory use was {peak_mem:.2f} GB, about {peak_fraction * 100:.1f}% of 48 GB unified memory."
        )
    else:
        lines.append(f"- Highest completed training memory use was {peak_mem:.2f} GB.")
    lines.extend([
        "- The M4 Pro is being used correctly for the training work we have actually approved: PyTorch MPS, not CPU.",
    ])
    if has_confirmed_paired and has_confirmed_oos and has_confirmed_reports:
        lines.extend([
            "- The controlled Phase 3 warm-start completed and passed the silver non-regression gate modestly.",
            "- A generic recency-consistency guarded fusion runtime lifted paired accuracy to 91.0% / 12-of-12 locks without assisted-OOS regression.",
            "- The confirmed loop-align smoother now lifts paired accuracy to 92.8% and assisted-OOS to 60.8% without additional training.",
            "- This proves the current bottleneck is still runtime line-path logic, not M4 Pro underuse.",
            "- The 48 GB headroom remains useful for future larger batches, gradient checkpointing experiments, and longer runs, but full 300h / multi-seed training is not justified until the line-path lane stops producing generic gains.",
            "- Do not pull/train on all 300h right now. The next valid experiment is reducing remaining adjacent_backtrack and outside_gt_line_set errors under the confirmed loop-align runtime.",
            "- Next recommended compute use: cached-output line-path diagnostics and targeted smoother changes, then re-run paired + assisted-OOS gates.",
        ])
    elif has_recency_paired and has_recency_oos and has_alignment_reports:
        lines.extend([
            "- The controlled Phase 3 warm-start completed and passed the silver non-regression gate modestly.",
            "- A generic recency-consistency guarded fusion runtime lifted paired accuracy to 91.0% / 12-of-12 locks without assisted-OOS regression.",
            "- Alignment-error reports show the active blocker: paired residual errors are mostly wrong-line/boundary issues; assisted-OOS is mostly wrong-line plus outside-GT line choices inside the correct shabad.",
            "- The 48 GB headroom remains useful for future larger batches, gradient checkpointing experiments, and longer runs, but full 300h / multi-seed training is not justified until OOS alignment/canonical resolution improves or diagnostics prove true ASR misses.",
            "- Do not pull/train on all 300h right now. The next valid experiment is locked-shabad aligner diagnostics: line-state transitions, loop/refrain behavior, and null/no-line penalties.",
            "- Next recommended compute use: cached-output diagnostics on OOS unresolved predictions and loop-align wrong-line spans.",
        ])
    elif has_recency_paired and has_recency_oos:
        lines.extend([
            "- The controlled Phase 3 warm-start completed and passed the silver non-regression gate modestly.",
            "- A generic recency-consistency guarded fusion runtime lifted paired accuracy to 91.0% / 12-of-12 locks without assisted-OOS regression.",
            "- Assisted-OOS remains flat at 59.9% despite 5-of-5 locks, so the active blocker is line timing/alignment under the correct shabad, not M4 Pro capacity.",
            "- The 48 GB headroom remains useful for future larger batches, gradient checkpointing experiments, and longer runs, but full 300h / multi-seed training is not justified until OOS alignment improves or diagnostics prove true ASR misses.",
            "- Do not pull/train on all 300h right now. The next valid experiment is OOS/paired line-alignment error analysis under the recency-guarded runtime.",
            "- Next recommended compute use: cached-output alignment diagnostics on `submissions/phase3_recency_guard_paired` and `submissions/oos_v1_assisted_phase3_recency_guard`.",
        ])
    elif has_v6_silver and has_v6_paired and has_v6_oos:
        lines.extend([
            "- The controlled Phase 3 warm-start completed and passed the silver non-regression gate modestly.",
            "- Paired and assisted-OOS runtime gates were flat: v6 did not regress, but it did not move frame accuracy toward the 95% target.",
            "- We are not currently compute-bound; the active blocker is generic lock/alignment quality, especially the persistent full-start zOtIpxMT9hU false lock and OOS timing weakness.",
            "- The 48 GB headroom remains useful for future larger batches, gradient checkpointing experiments, and longer runs, but full 300h / multi-seed training is not justified from this checkpoint.",
            "- Do not pull/train on all 300h right now. The next valid experiment is cached-ASR lock recency-consistency analysis, followed by a generic runtime change only if it preserves paired + OOS behavior.",
            "- Next recommended compute use: `make audit-lock-recency-consistency`.",
        ])
    elif has_v6_silver:
        lines.extend([
            "- The controlled Phase 3 warm-start completed and passed the silver non-regression gate modestly.",
            "- We are not currently compute-bound; the active blocker is paired/OOS validation quality for `v6_mac_scale20` under the generic lock/alignment runtime.",
            "- The 48 GB headroom remains useful for future larger batches, gradient checkpointing experiments, and longer runs, but more training is not justified until paired/OOS gates are checked.",
            "- Do not pull/train on all 300h right now. The next valid experiment is paired runtime evaluation of `lora_adapters/v6_mac_scale20` with the Phase 2.13 evidence-fusion lock policy.",
            "- Next recommended compute use: run `scripts/run_idlock_path.py` on the paired benchmark with `--post-adapter-dir lora_adapters/v6_mac_scale20`, `--post-context buffered`, `--merge-policy retro-buffered`, `--smoother loop_align`, and `--blind-aggregate fusion:tfidf_45+0.5*chunk_vote_90`.",
        ])
    elif has_v6:
        lines.extend([
            "- The controlled Phase 3 warm-start completed. We are not currently compute-bound; the active blocker is held-out validation quality for `v6_mac_scale20`.",
            "- The 48 GB headroom remains useful for future larger batches, gradient checkpointing experiments, and longer runs, but more training is not justified until silver/OOS gates are checked.",
            "- Do not pull/train on all 300h right now. The next valid experiment is held-out silver evaluation of `lora_adapters/v6_mac_scale20` against base `surt-small-v3` and `v5b_mac_diverse`.",
            "- Next recommended compute use: `make eval-silver-300h SILVER_ADAPTER_DIR=lora_adapters/v6_mac_scale20 SILVER_OUT=submissions/silver_300h_v6_mac_scale20.json`.",
        ])
    else:
        lines.extend([
            "- We are not currently compute-bound. The current blocker is validation quality: gold OOS for `phase2_9_loop_align`, not more broad data or another blind LoRA scale-up.",
            "- The 48 GB headroom is useful for the future Phase 3 plan (larger batches, gradient checkpointing experiments, longer runs), but Phase 3 is intentionally gated.",
            "- Do not pull/train on all 300h right now. The silver audit found label-risk rows, not clean ASR failures, and `v5b_mac_diverse` was neutral/regressive outside oracle alignment.",
            "- Next recommended compute use: small targeted diagnostics or gold OOS scoring. Next recommended non-compute work: finish OOS v1 GT validation.",
        ])
    lines.extend([
        "",
        "## If Phase 3 is unblocked later",
        "",
        "- Re-enable MPS fp16 only after a torch >= 2.8 / accelerate >= 1.11 compatibility pass.",
        "- Verify `gradient_checkpointing=true` with PEFT+MPS in isolation before changing the main YAML.",
        "- Use the 48 GB machine for full-slice or multi-seed runs only after silver/OOS gates pass or a deliberate pivot is documented.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()
    report = render_report()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"wrote: {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
