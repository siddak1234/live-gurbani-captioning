#!/usr/bin/env python3
"""Summarize frame-level alignment errors for a prediction directory.

This is a companion diagnostic to the benchmark scorer. The benchmark gives the
headline percentage; this report asks what kind of frames are still wrong:

- missing prediction over a labeled GT span;
- wrong resolved line_idx;
- unresolved prediction (`NO_MATCH`);
- boundary/collar errors.

The script imports the benchmark's own `score_video` implementation so the
frame details are identical to the official metric. It does not rerun ASR.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_EVAL = REPO_ROOT.parent / "live-gurbani-captioning-benchmark-v1" / "eval.py"
DEFAULT_OUT = REPO_ROOT / "diagnostics" / "alignment_errors.md"


@dataclass(frozen=True)
class ErrorSpan:
    start: int
    end: int
    gt: object
    pred: object
    kind: str

    @property
    def duration_s(self) -> int:
        return int(self.end) - int(self.start)


def load_benchmark_eval(path: pathlib.Path = DEFAULT_BENCHMARK_EVAL):
    spec = importlib.util.spec_from_file_location("benchmark_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import benchmark eval from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_error(detail: dict[str, Any]) -> str:
    pred = detail.get("pred")
    if pred is None:
        return "missing_pred"
    if pred == "__no_match__":
        return "unresolved_pred"
    if detail.get("type") == "boundary_error":
        return "boundary_wrong"
    return "wrong_line"


def error_spans(details: list[dict[str, Any]]) -> list[ErrorSpan]:
    spans: list[ErrorSpan] = []
    active: ErrorSpan | None = None
    for detail in details:
        if detail.get("correct"):
            if active is not None:
                spans.append(active)
                active = None
            continue
        t = int(detail["t"])
        key = (detail.get("gt"), detail.get("pred"), classify_error(detail))
        if (
            active is not None
            and active.end == t
            and (active.gt, active.pred, active.kind) == key
        ):
            active = ErrorSpan(active.start, t + 1, active.gt, active.pred, active.kind)
        else:
            if active is not None:
                spans.append(active)
            active = ErrorSpan(t, t + 1, key[0], key[1], key[2])
    if active is not None:
        spans.append(active)
    return spans


def summarize_case(score: dict[str, Any], *, case_id: str | None = None) -> dict[str, Any]:
    spans = error_spans(score["details"])
    by_kind = Counter()
    for span in spans:
        by_kind[span.kind] += span.duration_s
    return {
        "video_id": score["video_id"],
        "case_id": case_id or score["video_id"],
        "shabad_id": score.get("shabad_id"),
        "accuracy": float(score["frame_accuracy"]),
        "correct": int(score["correct"]),
        "total": int(score["total"]),
        "error_frames": int(score["total"]) - int(score["correct"]),
        "n_pred_segments": int(score["n_pred_segments"]),
        "n_gt_segments": int(score["n_gt_segments"]),
        "by_kind": dict(sorted(by_kind.items())),
        "spans": sorted(spans, key=lambda s: (-s.duration_s, s.start)),
    }


def _label(value: object) -> str:
    return "null" if value is None else str(value)


def render_markdown(
    *,
    pred_dir: pathlib.Path,
    gt_dir: pathlib.Path,
    summaries: list[dict[str, Any]],
    collar: int,
) -> str:
    total_correct = sum(row["correct"] for row in summaries)
    total_frames = sum(row["total"] for row in summaries)
    overall = 100.0 * total_correct / total_frames if total_frames else 0.0
    kind_totals = Counter()
    for row in summaries:
        kind_totals.update(row["by_kind"])

    lines = [
        "# Alignment error report",
        "",
        "**Diagnostic only.** Uses the benchmark scorer's per-frame details and",
        "groups incorrect frames into contiguous spans. This tells us whether the",
        "next move should be aligner/timing work or acoustic training.",
        "",
        "## Inputs",
        "",
        f"- Predictions: `{pred_dir}`",
        f"- Ground truth: `{gt_dir}`",
        f"- Collar: `{collar}s`",
        "",
        "## Summary",
        "",
        f"- Overall frame accuracy: `{overall:.1f}%` ({total_correct}/{total_frames})",
        f"- Error frames: `{total_frames - total_correct}`",
        "",
        "| Error kind | Frames | Share of errors |",
        "|---|---:|---:|",
    ]
    total_errors = max(1, total_frames - total_correct)
    for kind, count in sorted(kind_totals.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{kind}` | {count} | {100 * count / total_errors:.1f}% |")

    lines.extend([
        "",
        "## Per Case",
        "",
        "| Case | Accuracy | Error frames | Pred segs | GT segs | Error mix |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in sorted(summaries, key=lambda r: r["accuracy"]):
        mix = ", ".join(f"{k}={v}" for k, v in row["by_kind"].items()) or "none"
        lines.append(
            f"| `{row['case_id']}` | {row['accuracy']:.1f}% | {row['error_frames']} | "
            f"{row['n_pred_segments']} | {row['n_gt_segments']} | {mix} |"
        )

    lines.extend([
        "",
        "## Longest Error Spans",
        "",
        "| Case | Start | End | Dur | Kind | GT | Pred |",
        "|---|---:|---:|---:|---|---:|---:|",
    ])
    all_spans: list[tuple[str, ErrorSpan]] = []
    for row in summaries:
        all_spans.extend((row["case_id"], span) for span in row["spans"])
    for case_id, span in sorted(all_spans, key=lambda item: (-item[1].duration_s, item[0], item[1].start))[:25]:
        lines.append(
            f"| `{case_id}` | {span.start} | {span.end} | {span.duration_s} | "
            f"`{span.kind}` | {_label(span.gt)} | {_label(span.pred)} |"
        )

    lines.extend([
        "",
        "## Interpretation Guide",
        "",
        "- High `missing_pred`: aligner/no-line behavior is too conservative or ASR chunks are sparse.",
        "- High `wrong_line`: locked-shabad line smoother is choosing the wrong pangti, often a loop/refrain issue.",
        "- High `boundary_wrong`: segment boundaries need timing smoothing; label identity may be right nearby.",
        "- High `unresolved_pred`: predictions are not resolving to the GT shabad's canonical line IDs/text.",
        "",
    ])
    return "\n".join(lines)


def run_report(*, pred_dir: pathlib.Path, gt_dir: pathlib.Path, collar: int, benchmark_eval: pathlib.Path) -> list[dict[str, Any]]:
    scorer = load_benchmark_eval(benchmark_eval)
    summaries: list[dict[str, Any]] = []
    for gt_path in sorted(gt_dir.glob("*.json")):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            raise FileNotFoundError(f"missing prediction for {gt_path.name}: {pred_path}")
        gt = json.loads(gt_path.read_text())
        pred = json.loads(pred_path.read_text())
        score = scorer.score_video(gt, pred, collar=collar)
        summaries.append(summarize_case(score, case_id=gt_path.stem))
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=pathlib.Path, required=True)
    parser.add_argument("--gt-dir", type=pathlib.Path, required=True)
    parser.add_argument("--collar", type=int, default=1)
    parser.add_argument("--benchmark-eval", type=pathlib.Path, default=DEFAULT_BENCHMARK_EVAL)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    summaries = run_report(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        collar=args.collar,
        benchmark_eval=args.benchmark_eval,
    )
    report = render_markdown(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        summaries=summaries,
        collar=args.collar,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
