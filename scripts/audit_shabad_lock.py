#!/usr/bin/env python3
"""Audit blind shabad-lock behavior from cached ASR transcripts.

This is a Phase 2.11 diagnostic tool. It does not run ASR and it does not
score line timing. It answers a narrower question:

    Given the cached pre-lock ASR text and the current candidate corpus set,
    which shabad would each lock policy choose?

That keeps the failure mode separate from expensive model inference.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.asr import AsrChunk  # noqa: E402
from src.shabad_id import ShabadIdResult, identify_shabad  # noqa: E402


DEFAULT_GT_DIR = REPO_ROOT / "eval_data" / "oos_v1" / "assisted_test"
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus_cache"
DEFAULT_ASR_CACHE = REPO_ROOT / "asr_cache"


@dataclass(frozen=True)
class Case:
    case_id: str
    video_id: str
    shabad_id: int


@dataclass(frozen=True)
class AuditRow:
    case_id: str
    gt_shabad_id: int
    predicted_shabad_id: int | None
    score: float | None
    runner_up_id: int | None
    runner_up_score: float | None
    mode: str
    ok: bool
    missing_cache: bool = False


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_lookbacks(raw: str) -> list[float]:
    return [float(part) for part in parse_csv(raw)]


def load_cases(gt_dir: pathlib.Path) -> list[Case]:
    cases: list[Case] = []
    for path in sorted(gt_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        cases.append(Case(
            case_id=path.stem,
            video_id=str(payload["video_id"]),
            shabad_id=int(payload["shabad_id"]),
        ))
    return cases


def load_corpora(corpus_dir: pathlib.Path) -> dict[int, list[dict]]:
    corpora: dict[int, list[dict]] = {}
    for path in sorted(corpus_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        corpora[int(payload["shabad_id"])] = payload["lines"]
    return corpora


def load_cached_chunks(asr_cache_dir: pathlib.Path, video_id: str, asr_tag: str) -> list[AsrChunk] | None:
    path = asr_cache_dir / f"{video_id}_16k__{asr_tag}__pa.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    return [
        AsrChunk(start=float(row["start"]), end=float(row["end"]), text=str(row["text"]))
        for row in rows
    ]


def identify_hybrid(
    chunks: list[AsrChunk],
    corpora: dict[int, list[dict]],
    *,
    lookback_seconds: float,
    tfidf_min_score: float,
    tfidf_min_margin: float,
) -> tuple[ShabadIdResult, str]:
    """Prefer clear TF-IDF document evidence, else fall back to top-3 line evidence."""
    tfidf = identify_shabad(
        chunks,
        corpora,
        start_t=0.0,
        lookback_seconds=lookback_seconds,
        aggregate="tfidf",
        ratio="WRatio",
    )
    margin = float(tfidf.score) - float(tfidf.runner_up_score or 0.0)
    if tfidf.score >= tfidf_min_score and margin >= tfidf_min_margin:
        return tfidf, "tfidf"

    topk = identify_shabad(
        chunks,
        corpora,
        start_t=0.0,
        lookback_seconds=lookback_seconds,
        aggregate="topk:3",
        ratio="WRatio",
    )
    return topk, "topk:3"


def run_variant(
    cases: list[Case],
    corpora: dict[int, list[dict]],
    *,
    asr_cache_dir: pathlib.Path,
    asr_tag: str,
    lookback_seconds: float,
    aggregate: str,
    tfidf_min_score: float,
    tfidf_min_margin: float,
) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for case in cases:
        chunks = load_cached_chunks(asr_cache_dir, case.video_id, asr_tag)
        if chunks is None:
            rows.append(AuditRow(
                case_id=case.case_id,
                gt_shabad_id=case.shabad_id,
                predicted_shabad_id=None,
                score=None,
                runner_up_id=None,
                runner_up_score=None,
                mode=aggregate,
                ok=False,
                missing_cache=True,
            ))
            continue
        if aggregate == "tfidf_then_topk3":
            result, mode = identify_hybrid(
                chunks,
                corpora,
                lookback_seconds=lookback_seconds,
                tfidf_min_score=tfidf_min_score,
                tfidf_min_margin=tfidf_min_margin,
            )
        else:
            result = identify_shabad(
                chunks,
                corpora,
                start_t=0.0,
                lookback_seconds=lookback_seconds,
                aggregate=aggregate,
                ratio="WRatio",
            )
            mode = aggregate
        rows.append(AuditRow(
            case_id=case.case_id,
            gt_shabad_id=case.shabad_id,
            predicted_shabad_id=int(result.shabad_id),
            score=float(result.score),
            runner_up_id=None if result.runner_up_id is None else int(result.runner_up_id),
            runner_up_score=float(result.runner_up_score or 0.0),
            mode=mode,
            ok=int(result.shabad_id) == int(case.shabad_id),
        ))
    return rows


def render_markdown(
    *,
    gt_dir: pathlib.Path,
    corpus_dir: pathlib.Path,
    asr_cache_dir: pathlib.Path,
    asr_tag: str,
    lookbacks: list[float],
    aggregates: list[str],
    results: dict[tuple[float, str], list[AuditRow]],
) -> str:
    n_cases = len(next(iter(results.values()))) if results else 0
    corpus_count = len(list(corpus_dir.glob("*.json")))
    def display(path: pathlib.Path) -> str:
        try:
            return str(path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    lines: list[str] = [
        "# Shabad-lock audit",
        "",
        "Diagnostic only. This report scores blind shabad-ID choices from cached",
        "ASR transcripts; it does not run ASR and does not evaluate line timing.",
        "",
        "## Inputs",
        "",
        f"- GT dir: `{display(gt_dir)}`",
        f"- ASR cache: `{display(asr_cache_dir)}`",
        f"- ASR tag: `{asr_tag}`",
        f"- Corpus dir: `{display(corpus_dir)}` ({corpus_count} cached shabads)",
        f"- Cases: {n_cases}",
        "",
        "## Variant summary",
        "",
        "| Lookback | Aggregate | Correct | Missing cache |",
        "|---:|---|---:|---:|",
    ]
    for lookback in lookbacks:
        for aggregate in aggregates:
            rows = results[(lookback, aggregate)]
            ok = sum(1 for row in rows if row.ok)
            missing = sum(1 for row in rows if row.missing_cache)
            lines.append(f"| {lookback:g}s | `{aggregate}` | {ok}/{len(rows) - missing} | {missing} |")

    lines.extend([
        "",
        "## Per-case details",
        "",
    ])
    for lookback in lookbacks:
        for aggregate in aggregates:
            lines.extend([
                f"### {lookback:g}s / `{aggregate}`",
                "",
                "| Case | GT | Pred | Mode | Score | Runner-up | Result |",
                "|---|---:|---:|---|---:|---|---|",
            ])
            for row in results[(lookback, aggregate)]:
                if row.missing_cache:
                    lines.append(
                        f"| {row.case_id} | {row.gt_shabad_id} | — | {row.mode} | — | — | MISSING_CACHE |"
                    )
                    continue
                runner = "—"
                if row.runner_up_id is not None and row.runner_up_score is not None:
                    runner = f"{row.runner_up_id} ({row.runner_up_score:.1f})"
                status = "OK" if row.ok else "BAD"
                lines.append(
                    f"| {row.case_id} | {row.gt_shabad_id} | {row.predicted_shabad_id} | "
                    f"{row.mode} | {row.score:.1f} | {runner} | {status} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=pathlib.Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--corpus-dir", type=pathlib.Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--asr-cache-dir", type=pathlib.Path, default=DEFAULT_ASR_CACHE)
    parser.add_argument("--asr-tag", default="medium_word",
                        help="Cache tag in <video>_16k__<tag>__pa.json")
    parser.add_argument("--lookbacks", default="30,45,60,90")
    parser.add_argument("--aggregates", default="chunk_vote,tfidf,topk:3,tfidf_then_topk3")
    parser.add_argument("--tfidf-min-score", type=float, default=10.0)
    parser.add_argument("--tfidf-min-margin", type=float, default=1.0)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    cases = load_cases(args.gt_dir)
    if not cases:
        print(f"error: no GT files in {args.gt_dir}", file=sys.stderr)
        return 1
    corpora = load_corpora(args.corpus_dir)
    if not corpora:
        print(f"error: no corpus files in {args.corpus_dir}", file=sys.stderr)
        return 1

    lookbacks = parse_lookbacks(args.lookbacks)
    aggregates = parse_csv(args.aggregates)
    results: dict[tuple[float, str], list[AuditRow]] = {}
    for lookback in lookbacks:
        for aggregate in aggregates:
            results[(lookback, aggregate)] = run_variant(
                cases,
                corpora,
                asr_cache_dir=args.asr_cache_dir,
                asr_tag=args.asr_tag,
                lookback_seconds=lookback,
                aggregate=aggregate,
                tfidf_min_score=args.tfidf_min_score,
                tfidf_min_margin=args.tfidf_min_margin,
            )

    markdown = render_markdown(
        gt_dir=args.gt_dir,
        corpus_dir=args.corpus_dir,
        asr_cache_dir=args.asr_cache_dir,
        asr_tag=args.asr_tag,
        lookbacks=lookbacks,
        aggregates=aggregates,
        results=results,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown)
        print(f"wrote: {args.out}")
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
