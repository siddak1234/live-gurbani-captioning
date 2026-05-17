#!/usr/bin/env python3
"""Audit locked-shabad line-path errors.

This is the next diagnostic after shabad lock accuracy is mostly solved. It
uses the benchmark scorer for frame correctness, then enriches incorrect spans
with corpus text and path relations:

- adjacent forward/backward line errors;
- larger jumps/backtracks;
- correct-shabad predictions outside the GT-labeled clip lines;
- wrong-shabad lines and missing predictions.

It does not rerun ASR or modify predictions.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.report_alignment_errors import (  # noqa: E402
    DEFAULT_BENCHMARK_EVAL,
    ErrorSpan,
    error_spans,
    load_benchmark_eval,
    raw_pred_segment_frames,
)


DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus_cache"
DEFAULT_OUT = REPO_ROOT / "diagnostics" / "line_path_errors.md"


def load_corpora(corpus_dir: pathlib.Path) -> dict[int, dict[int, dict[str, Any]]]:
    corpora: dict[int, dict[int, dict[str, Any]]] = {}
    for path in sorted(corpus_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        shabad_id = int(payload["shabad_id"])
        corpora[shabad_id] = {
            int(line["line_idx"]): line
            for line in payload.get("lines", [])
        }
    return corpora


def _gt_line_set(gt: dict[str, Any]) -> set[int]:
    return {
        int(segment["line_idx"])
        for segment in gt.get("segments", [])
        if segment.get("line_idx") is not None
    }


def _line_text(corpora: dict[int, dict[int, dict[str, Any]]], shabad_id: int | None, line_idx: int | None) -> str:
    if shabad_id is None or line_idx is None:
        return ""
    line = corpora.get(int(shabad_id), {}).get(int(line_idx), {})
    return str(line.get("banidb_gurmukhi") or line.get("transliteration_english") or "")


def _short(text: str, limit: int = 54) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def _as_int(value: object) -> int | None:
    if value is None or value == "__no_match__":
        return None
    return int(value)


def relation_for_span(
    span: ErrorSpan,
    *,
    raw_pred_segment: dict | None,
    gt_shabad_id: int,
    gt_lines: set[int],
) -> str:
    """Return the line-path relation for an incorrect span."""
    if raw_pred_segment is None:
        return "missing_prediction"

    pred_shabad = raw_pred_segment.get("shabad_id")
    if pred_shabad is not None and int(pred_shabad) != int(gt_shabad_id):
        return "wrong_shabad"

    pred_line = _as_int(raw_pred_segment.get("line_idx"))
    gt_line = _as_int(span.gt)
    if gt_line is None:
        return "predicted_during_unlabeled_gt"
    if pred_line is None:
        return "missing_line_idx"
    if pred_line == gt_line:
        return "boundary_or_resolution"
    if pred_line not in gt_lines:
        return "outside_gt_line_set"

    delta = pred_line - gt_line
    if delta == 1:
        return "adjacent_future"
    if delta == -1:
        return "adjacent_backtrack"
    if delta > 1:
        return "future_jump"
    return "backtrack_jump"


def span_record(
    *,
    case_id: str,
    span: ErrorSpan,
    raw_pred_segment: dict | None,
    gt: dict[str, Any],
    corpora: dict[int, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    gt_shabad = int(gt["shabad_id"])
    gt_line = _as_int(span.gt)
    pred_shabad = None
    pred_line = None
    if raw_pred_segment is not None:
        pred_shabad = _as_int(raw_pred_segment.get("shabad_id"))
        pred_line = _as_int(raw_pred_segment.get("line_idx"))
    relation = relation_for_span(
        span,
        raw_pred_segment=raw_pred_segment,
        gt_shabad_id=gt_shabad,
        gt_lines=_gt_line_set(gt),
    )
    return {
        "case_id": case_id,
        "start": int(span.start),
        "end": int(span.end),
        "duration_s": int(span.duration_s),
        "error_kind": span.kind,
        "relation": relation,
        "gt_shabad_id": gt_shabad,
        "gt_line_idx": gt_line,
        "pred_shabad_id": pred_shabad,
        "pred_line_idx": pred_line,
        "line_delta": (pred_line - gt_line) if pred_line is not None and gt_line is not None else None,
        "gt_text": _line_text(corpora, gt_shabad, gt_line),
        "pred_text": _line_text(corpora, pred_shabad, pred_line),
    }


def audit_case(
    *,
    gt_path: pathlib.Path,
    pred_path: pathlib.Path,
    scorer: Any,
    corpora: dict[int, dict[int, dict[str, Any]]],
    collar: int,
) -> dict[str, Any]:
    gt = json.loads(gt_path.read_text())
    pred = json.loads(pred_path.read_text())
    score = scorer.score_video(gt, pred, collar=collar)
    total_seconds = int(gt.get("total_duration", score.get("uem_end", 0)))
    raw_frames = raw_pred_segment_frames(pred.get("segments", []), total_seconds)
    spans = error_spans(score["details"], raw_pred_frames=raw_frames, gt_shabad_id=int(gt["shabad_id"]))
    records = []
    for span in spans:
        raw_seg = raw_frames[span.start] if span.start < len(raw_frames) else None
        records.append(span_record(
            case_id=gt_path.stem,
            span=span,
            raw_pred_segment=raw_seg,
            gt=gt,
            corpora=corpora,
        ))
    return {
        "case_id": gt_path.stem,
        "accuracy": float(score["frame_accuracy"]),
        "correct": int(score["correct"]),
        "total": int(score["total"]),
        "error_frames": int(score["total"]) - int(score["correct"]),
        "n_gt_segments": int(score["n_gt_segments"]),
        "n_pred_segments": int(score["n_pred_segments"]),
        "spans": records,
    }


def render_markdown(
    *,
    pred_dir: pathlib.Path,
    gt_dir: pathlib.Path,
    corpus_dir: pathlib.Path,
    cases: list[dict[str, Any]],
    collar: int,
) -> str:
    total_correct = sum(case["correct"] for case in cases)
    total_frames = sum(case["total"] for case in cases)
    total_errors = max(1, total_frames - total_correct)
    overall = 100.0 * total_correct / total_frames if total_frames else 0.0

    relation_totals = Counter()
    kind_totals = Counter()
    pair_totals = Counter()
    for case in cases:
        for span in case["spans"]:
            relation_totals[span["relation"]] += span["duration_s"]
            kind_totals[span["error_kind"]] += span["duration_s"]
            pair = (span["gt_line_idx"], span["pred_line_idx"], span["relation"])
            pair_totals[pair] += span["duration_s"]

    lines = [
        "# Locked-shabad line-path audit",
        "",
        "**Diagnostic only.** Shabad lock is treated as mostly solved here; this",
        "report asks how the line tracker fails *inside* the locked shabad. The",
        "goal is to decide whether the next high-accuracy move is aligner logic or",
        "another broad acoustic training run.",
        "",
        "## Inputs",
        "",
        f"- Predictions: `{pred_dir}`",
        f"- Ground truth: `{gt_dir}`",
        f"- Corpus: `{corpus_dir}`",
        f"- Scorer collar: `{collar}s`",
        "",
        "## Summary",
        "",
        f"- Overall frame accuracy: `{overall:.1f}%` ({total_correct}/{total_frames})",
        f"- Error frames: `{total_frames - total_correct}`",
        "",
        "### Error Kinds",
        "",
        "| Kind | Frames | Share of errors |",
        "|---|---:|---:|",
    ]
    for kind, count in sorted(kind_totals.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{kind}` | {count} | {100 * count / total_errors:.1f}% |")

    lines.extend([
        "",
        "### Line-Path Relations",
        "",
        "| Relation | Frames | Share of errors |",
        "|---|---:|---:|",
    ])
    for relation, count in sorted(relation_totals.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{relation}` | {count} | {100 * count / total_errors:.1f}% |")

    lines.extend([
        "",
        "## Per Case",
        "",
        "| Case | Accuracy | Error frames | Pred segs | GT segs | Dominant relations |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for case in sorted(cases, key=lambda row: row["accuracy"]):
        rel = Counter()
        for span in case["spans"]:
            rel[span["relation"]] += span["duration_s"]
        rel_s = ", ".join(f"{k}={v}" for k, v in rel.most_common(4)) or "none"
        lines.append(
            f"| `{case['case_id']}` | {case['accuracy']:.1f}% | {case['error_frames']} | "
            f"{case['n_pred_segments']} | {case['n_gt_segments']} | {rel_s} |"
        )

    lines.extend([
        "",
        "## Top Line Confusions",
        "",
        "| GT line | Pred line | Relation | Frames |",
        "|---:|---:|---|---:|",
    ])
    for (gt_line, pred_line, relation), count in pair_totals.most_common(15):
        gt_s = "null" if gt_line is None else str(gt_line)
        pred_s = "null" if pred_line is None else str(pred_line)
        lines.append(f"| {gt_s} | {pred_s} | `{relation}` | {count} |")

    lines.extend([
        "",
        "## Longest Error Spans",
        "",
        "| Case | Start | End | Dur | Relation | Kind | GT | Pred |",
        "|---|---:|---:|---:|---|---|---|---|",
    ])
    all_spans = [span for case in cases for span in case["spans"]]
    for span in sorted(all_spans, key=lambda row: (-row["duration_s"], row["case_id"], row["start"]))[:30]:
        gt_label = "null" if span["gt_line_idx"] is None else f"{span['gt_line_idx']}: {_short(span['gt_text'])}"
        pred_label = "null" if span["pred_line_idx"] is None else f"{span['pred_line_idx']}: {_short(span['pred_text'])}"
        lines.append(
            f"| `{span['case_id']}` | {span['start']} | {span['end']} | {span['duration_s']} | "
            f"`{span['relation']}` | `{span['error_kind']}` | {gt_label} | {pred_label} |"
        )

    lines.extend([
        "",
        "## Decision Use",
        "",
        "- High `adjacent_future` / `adjacent_backtrack`: tune transition penalties and boundary smoothing before larger training.",
        "- High `future_jump` / `backtrack_jump`: line-state path is unstable; Viterbi/loop-align constraints should be tightened.",
        "- High `outside_gt_line_set`: inspect whether the clip labels omit sung repeats; if labels are acceptable, add a no-line/end-of-clip guard.",
        "- High `wrong_shabad`: return to shabad-lock evidence; this report is no longer in the locked-shabad regime.",
        "",
    ])
    return "\n".join(lines)


def run_audit(
    *,
    pred_dir: pathlib.Path,
    gt_dir: pathlib.Path,
    corpus_dir: pathlib.Path,
    benchmark_eval: pathlib.Path,
    collar: int,
) -> list[dict[str, Any]]:
    scorer = load_benchmark_eval(benchmark_eval)
    corpora = load_corpora(corpus_dir)
    cases = []
    for gt_path in sorted(gt_dir.glob("*.json")):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            raise FileNotFoundError(f"missing prediction for {gt_path.name}: {pred_path}")
        cases.append(audit_case(
            gt_path=gt_path,
            pred_path=pred_path,
            scorer=scorer,
            corpora=corpora,
            collar=collar,
        ))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=pathlib.Path, required=True)
    parser.add_argument("--gt-dir", type=pathlib.Path, required=True)
    parser.add_argument("--corpus-dir", type=pathlib.Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--benchmark-eval", type=pathlib.Path, default=DEFAULT_BENCHMARK_EVAL)
    parser.add_argument("--collar", type=int, default=1)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cases = run_audit(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        corpus_dir=args.corpus_dir,
        benchmark_eval=args.benchmark_eval,
        collar=args.collar,
    )
    report = render_markdown(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        corpus_dir=args.corpus_dir,
        cases=cases,
        collar=args.collar,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
