#!/usr/bin/env python3
"""Audit weak silver rows against the original 300h parquet metadata.

The silver ASR score compares model text to canonicalized ``final_text``. For
weak rows, we also need to know whether the model is wrong acoustically or
whether the silver label is a risky canonical replacement. This script scans the
cached Hugging Face parquet rows for weak clip IDs and compares:

  - raw caption text
  - canonical final text
  - base / adapter predictions
  - dataset decision, retrieval margin, and canonical op counts
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import sys
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rapidfuzz import fuzz  # noqa: E402
from src.matcher import normalize  # noqa: E402
from scripts.report_silver_weak_slices import PairedRow, load_paired  # noqa: E402


HF_DATASET_CACHE = (
    pathlib.Path.home()
    / ".cache/huggingface/hub"
    / "datasets--surindersinghssj--gurbani-kirtan-yt-captions-300h-canonical"
    / "snapshots"
)


@dataclass
class SourceAuditRow:
    paired: PairedRow
    clip_id: str
    raw_text: str
    text: str
    final_text: str
    sggs_line: str
    decision: str
    canonical_match_score: float
    canonical_retrieval_margin: float
    canonical_op_counts: str
    start_s: float
    end_s: float
    base_raw_wratio: float
    v5b_raw_wratio: float
    base_final_wratio: float
    v5b_final_wratio: float
    assessment: str


def clip_id_from_audio(audio_path: str) -> str:
    return pathlib.Path(audio_path).stem


def find_parquet_dir() -> pathlib.Path:
    candidates = sorted(HF_DATASET_CACHE.glob("*/data"))
    if not candidates:
        raise FileNotFoundError(f"no cached parquet data found under {HF_DATASET_CACHE}")
    return candidates[-1]


def parse_shards(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    return out


def _norm_wratio(a: str, b: str) -> float:
    return float(fuzz.WRatio(normalize(a or ""), normalize(b or "")))


def _assessment(
    *,
    raw_best: float,
    final_best: float,
    decision: str,
    margin: float,
    op_counts: str,
) -> str:
    if raw_best >= final_best + 10 and final_best < 90:
        return "silver-label-risk: prediction matches raw caption better than canonical final"
    if decision in {"review", "replaced"} and margin < 0.30 and final_best < 90:
        return "silver-label-risk: low-margin canonical replacement"
    if '"fix": 3' in op_counts and final_best < 90:
        return "silver-label-risk: heavy canonical fixes"
    if final_best < 90:
        return "true-or-ambiguous-asr-miss"
    return "ok"


def _read_source_rows(parquet_dir: pathlib.Path, clip_ids: set[str], shards: list[int]) -> dict[str, dict]:
    import pyarrow.parquet as pq

    cols = [
        "clip_id",
        "raw_text",
        "text",
        "sggs_line",
        "final_text",
        "decision",
        "canonical_match_score",
        "canonical_op_counts",
        "canonical_retrieval_margin",
        "start_s",
        "end_s",
        "duration_s",
    ]
    found: dict[str, dict] = {}
    for shard in shards:
        path = parquet_dir / f"train-{shard:05d}-of-00129.parquet"
        if not path.exists():
            continue
        table = pq.read_table(path, columns=cols)
        for row in table.to_pylist():
            clip_id = row["clip_id"]
            if clip_id in clip_ids:
                found[clip_id] = row
        if len(found) == len(clip_ids):
            break
    return found


def audit_rows(
    paired_rows: list[PairedRow],
    *,
    parquet_dir: pathlib.Path,
    shards: list[int],
    threshold: float,
) -> list[SourceAuditRow]:
    weak = [r for r in paired_rows if r.best_wratio < threshold]
    clip_ids = {clip_id_from_audio(r.audio) for r in weak if r.audio}
    source = _read_source_rows(parquet_dir, clip_ids, shards)

    out: list[SourceAuditRow] = []
    for row in weak:
        clip_id = clip_id_from_audio(row.audio)
        if not clip_id or clip_id not in source:
            continue
        meta = source[clip_id]
        raw_text = str(meta.get("text") or meta.get("raw_text") or "")
        final_text = str(meta.get("final_text") or "")
        base_raw = _norm_wratio(raw_text, row.base_pred)
        v5b_raw = _norm_wratio(raw_text, row.v5b_pred)
        base_final = _norm_wratio(final_text, row.base_pred)
        v5b_final = _norm_wratio(final_text, row.v5b_pred)
        margin = float(meta.get("canonical_retrieval_margin") or 0.0)
        decision = str(meta.get("decision") or "")
        op_counts = str(meta.get("canonical_op_counts") or "")
        out.append(SourceAuditRow(
            paired=row,
            clip_id=clip_id,
            raw_text=raw_text,
            text=str(meta.get("text") or ""),
            final_text=final_text,
            sggs_line=str(meta.get("sggs_line") or ""),
            decision=decision,
            canonical_match_score=float(meta.get("canonical_match_score") or 0.0),
            canonical_retrieval_margin=margin,
            canonical_op_counts=op_counts,
            start_s=float(meta.get("start_s") or 0.0),
            end_s=float(meta.get("end_s") or 0.0),
            base_raw_wratio=base_raw,
            v5b_raw_wratio=v5b_raw,
            base_final_wratio=base_final,
            v5b_final_wratio=v5b_final,
            assessment=_assessment(
                raw_best=max(base_raw, v5b_raw),
                final_best=max(base_final, v5b_final),
                decision=decision,
                margin=margin,
                op_counts=op_counts,
            ),
        ))
    return out


def _fmt(text: str, max_len: int = 70) -> str:
    text = re.sub(r"\s+", " ", text or "").strip().replace("|", "\\|")
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def render_report(rows: list[SourceAuditRow], *, parquet_dir: pathlib.Path, threshold: float) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.assessment] = counts.get(row.assessment, 0) + 1

    return "\n".join([
        "# Phase 2.10 silver source-row audit",
        "",
        f"Parquet source: `{parquet_dir}`",
        f"Weak threshold: best(base, v5b) WRatio < `{threshold:.1f}`",
        "",
        "## Summary",
        "",
        _table(
            ["Assessment", "Rows"],
            [[k, str(v)] for k, v in sorted(counts.items())],
        ),
        "",
        "## Weak-row source metadata",
        "",
        _table(
            [
                "idx",
                "clip",
                "video",
                "decision",
                "margin",
                "raw->pred",
                "final->pred",
                "assessment",
                "raw text",
                "canonical final",
                "best pred",
            ],
            [
                [
                    str(r.paired.idx),
                    r.clip_id,
                    r.paired.video_id,
                    r.decision,
                    f"{r.canonical_retrieval_margin:.3f}",
                    f"{max(r.base_raw_wratio, r.v5b_raw_wratio):.1f}",
                    f"{max(r.base_final_wratio, r.v5b_final_wratio):.1f}",
                    r.assessment,
                    _fmt(r.raw_text),
                    _fmt(r.final_text),
                    _fmt(r.paired.v5b_pred if r.v5b_final_wratio >= r.base_final_wratio else r.paired.base_pred),
                ]
                for r in rows
            ],
        ),
        "",
        "## Read",
        "",
        "- Most weak rows are silver-label risks, not evidence that the model cannot hear the line.",
        "- The model often matches the raw caption / audible phrase better than the canonical replacement.",
        "- This confirms the current protocol: use silver for diagnostics, but keep gold OOS for promotion.",
        "- Next experiment should not be broad adapter scaling. Inspect weak audio/labels or improve runtime alignment.",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=pathlib.Path, default=pathlib.Path("submissions/silver_300h_surt_base_100.json"))
    parser.add_argument("--v5b", type=pathlib.Path, default=pathlib.Path("submissions/silver_300h_v5b_100.json"))
    parser.add_argument("--parquet-dir", type=pathlib.Path, default=None)
    parser.add_argument("--shards", default="10-19")
    parser.add_argument("--threshold", type=float, default=90.0)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("diagnostics/phase2_10_silver_source_audit.md"))
    args = parser.parse_args()

    parquet_dir = args.parquet_dir or find_parquet_dir()
    rows = audit_rows(
        load_paired(args.base, args.v5b),
        parquet_dir=parquet_dir,
        shards=parse_shards(args.shards),
        threshold=args.threshold,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(rows, parquet_dir=parquet_dir, threshold=args.threshold))
    print(f"wrote: {args.out}")
    print(f"audited weak rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
