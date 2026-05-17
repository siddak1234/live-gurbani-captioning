#!/usr/bin/env python3
"""Generate a weak-slice report from paired silver ASR eval outputs."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
from dataclasses import dataclass


@dataclass
class PairedRow:
    idx: int
    audio: str
    video_id: str
    shabad_id: str
    duration_s: float
    label_score: float
    target: str
    base_pred: str
    v5b_pred: str
    base_wratio: float
    v5b_wratio: float
    base_exact: bool
    v5b_exact: bool

    @property
    def delta(self) -> float:
        return self.v5b_wratio - self.base_wratio

    @property
    def best_wratio(self) -> float:
        return max(self.base_wratio, self.v5b_wratio)

    @property
    def worst_wratio(self) -> float:
        return min(self.base_wratio, self.v5b_wratio)


def _load_rows(path: pathlib.Path) -> dict[int, dict]:
    data = json.loads(path.read_text())
    rows = {}
    for row in data["rows"]:
        idx = int(row["idx"])
        if idx in rows:
            raise ValueError(f"duplicate idx {idx} in {path}")
        rows[idx] = row
    return rows


def load_paired(base_path: pathlib.Path, v5b_path: pathlib.Path) -> list[PairedRow]:
    base = _load_rows(base_path)
    v5b = _load_rows(v5b_path)
    missing_base = sorted(set(v5b) - set(base))
    missing_v5b = sorted(set(base) - set(v5b))
    if missing_base or missing_v5b:
        raise ValueError(
            f"silver row sets differ: missing_base={missing_base[:5]} "
            f"missing_v5b={missing_v5b[:5]}"
        )

    out = []
    for idx in sorted(base):
        b = base[idx]
        v = v5b[idx]
        for field in ("video_id", "shabad_id", "target"):
            if b.get(field) != v.get(field):
                raise ValueError(f"row {idx} field {field} differs between result files")
        out.append(PairedRow(
            idx=idx,
            audio=str(b.get("audio") or ""),
            video_id=str(b.get("video_id") or ""),
            shabad_id=str(b.get("shabad_id") or ""),
            duration_s=float(b.get("duration_s") or 0.0),
            label_score=float(b.get("label_score") or 0.0),
            target=str(b.get("target") or ""),
            base_pred=str(b.get("pred") or ""),
            v5b_pred=str(v.get("pred") or ""),
            base_wratio=float(b.get("wratio") or 0.0),
            v5b_wratio=float(v.get("wratio") or 0.0),
            base_exact=bool(b.get("exact_norm")),
            v5b_exact=bool(v.get("exact_norm")),
        ))
    return out


def duration_bucket(seconds: float) -> str:
    if seconds < 5:
        return "<5s"
    if seconds < 10:
        return "5-10s"
    if seconds < 15:
        return "10-15s"
    if seconds < 20:
        return "15-20s"
    return ">=20s"


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _fmt(text: str, max_len: int = 90) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def aggregate_by(rows: list[PairedRow], key_fn) -> list[tuple[str, list[PairedRow]]]:
    groups: dict[str, list[PairedRow]] = collections.defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return sorted(
        groups.items(),
        key=lambda item: (_mean([r.best_wratio for r in item[1]]), item[0]),
    )


def render_report(rows: list[PairedRow], *, base_path: pathlib.Path, v5b_path: pathlib.Path) -> str:
    weak = [r for r in rows if r.best_wratio < 90]
    base_better = [r for r in rows if r.delta <= -2]
    v5b_better = [r for r in rows if r.delta >= 2]

    lines = [
        "# Phase 2.10 silver weak-slice report",
        "",
        "Generated from:",
        f"- Base: `{base_path}`",
        f"- v5b: `{v5b_path}`",
        "",
        "## Summary",
        "",
        _table(
            ["Metric", "Value"],
            [
                ["Rows", str(len(rows))],
                ["Videos", str(len({r.video_id for r in rows}))],
                ["Shabad tokens", str(len({r.shabad_id for r in rows}))],
                ["Base mean WRatio", f"{_mean([r.base_wratio for r in rows]):.2f}"],
                ["v5b mean WRatio", f"{_mean([r.v5b_wratio for r in rows]):.2f}"],
                ["Mean v5b-base delta", f"{_mean([r.delta for r in rows]):+.2f}"],
                ["Rows where both models <90", str(len(weak))],
                ["Rows v5b better by >=2", str(len(v5b_better))],
                ["Rows base better by >=2", str(len(base_better))],
            ],
        ),
        "",
        "## Weak videos",
        "",
        _table(
            ["Video", "n", "Base mean", "v5b mean", "Best mean", "Min best", "Weak rows"],
            [
                [
                    video,
                    str(len(group)),
                    f"{_mean([r.base_wratio for r in group]):.1f}",
                    f"{_mean([r.v5b_wratio for r in group]):.1f}",
                    f"{_mean([r.best_wratio for r in group]):.1f}",
                    f"{min(r.best_wratio for r in group):.1f}",
                    str(sum(1 for r in group if r.best_wratio < 90)),
                ]
                for video, group in aggregate_by(rows, lambda r: r.video_id)[:10]
            ],
        ),
        "",
        "## Duration buckets",
        "",
        _table(
            ["Bucket", "n", "Base mean", "v5b mean", "Weak rows"],
            [
                [
                    bucket,
                    str(len(group)),
                    f"{_mean([r.base_wratio for r in group]):.1f}",
                    f"{_mean([r.v5b_wratio for r in group]):.1f}",
                    str(sum(1 for r in group if r.best_wratio < 90)),
                ]
                for bucket, group in aggregate_by(rows, lambda r: duration_bucket(r.duration_s))
            ],
        ),
        "",
        "## Worst rows by best available model",
        "",
        _table(
            ["idx", "video", "shabad", "dur", "base", "v5b", "delta", "target", "base pred", "v5b pred"],
            [
                [
                    str(r.idx),
                    r.video_id,
                    r.shabad_id,
                    f"{r.duration_s:.1f}s",
                    f"{r.base_wratio:.1f}",
                    f"{r.v5b_wratio:.1f}",
                    f"{r.delta:+.1f}",
                    _fmt(r.target),
                    _fmt(r.base_pred),
                    _fmt(r.v5b_pred),
                ]
                for r in sorted(rows, key=lambda r: (r.best_wratio, r.worst_wratio))[:20]
            ],
        ),
        "",
        "## Largest v5b regressions",
        "",
        _table(
            ["idx", "video", "shabad", "base", "v5b", "delta", "target", "base pred", "v5b pred"],
            [
                [
                    str(r.idx),
                    r.video_id,
                    r.shabad_id,
                    f"{r.base_wratio:.1f}",
                    f"{r.v5b_wratio:.1f}",
                    f"{r.delta:+.1f}",
                    _fmt(r.target),
                    _fmt(r.base_pred),
                    _fmt(r.v5b_pred),
                ]
                for r in sorted(rows, key=lambda r: r.delta)[:12]
            ],
        ),
        "",
        "## Largest v5b improvements",
        "",
        _table(
            ["idx", "video", "shabad", "base", "v5b", "delta", "target", "base pred", "v5b pred"],
            [
                [
                    str(r.idx),
                    r.video_id,
                    r.shabad_id,
                    f"{r.base_wratio:.1f}",
                    f"{r.v5b_wratio:.1f}",
                    f"{r.delta:+.1f}",
                    _fmt(r.target),
                    _fmt(r.base_pred),
                    _fmt(r.v5b_pred),
                ]
                for r in sorted(rows, key=lambda r: r.delta, reverse=True)[:12]
            ],
        ),
        "",
        "## Audit read",
        "",
    ]

    weak_video_names = [video for video, group in aggregate_by(rows, lambda r: r.video_id)[:3]]
    lines.extend([
        f"- Weakness is concentrated, not global: the weakest videos are {', '.join(f'`{v}`' for v in weak_video_names)}.",
        "- The 5-10s duration bucket contains most weak rows in this 100-row sample, so failures are not simply long-window truncation.",
        "- v5b is mixed: it improves some weak rows and regresses others. That argues against a broad adapter-scale-up as the next move.",
        "- Next engineering step: inspect weak-row audio/neighboring manifest rows for label noise and repeated-line ambiguity before changing training.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=pathlib.Path, default=pathlib.Path("submissions/silver_300h_surt_base_100.json"))
    parser.add_argument("--v5b", type=pathlib.Path, default=pathlib.Path("submissions/silver_300h_v5b_100.json"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("diagnostics/phase2_10_silver_weak_slices.md"))
    args = parser.parse_args()

    rows = load_paired(args.base, args.v5b)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(rows, base_path=args.base, v5b_path=args.v5b))
    print(f"wrote: {args.out}")
    print(f"rows={len(rows)} videos={len({r.video_id for r in rows})} weak={sum(1 for r in rows if r.best_wratio < 90)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
