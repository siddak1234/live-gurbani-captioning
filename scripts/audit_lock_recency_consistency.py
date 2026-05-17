#!/usr/bin/env python3
"""Audit early shabad-lock decisions against later evidence windows.

The current best generic lock policy is intentionally early:

    tfidf_45 + 0.5*chunk_vote_90

That is good for live UX, but Phase 2.13 and Phase 3 found one persistent
paired failure: full-start zOtIpxMT9hU locks to shabad 4892 even though later
audio points to the GT shabad 3712. This diagnostic does not change runtime
behavior. It asks a narrower question:

    Does a later validation window disagree with the early lock, and does the
    early winner have unusually low support in that later window?

The output is a guardrail report for deciding whether a generic recency-
consistency veto/delay policy is worth implementing. It is not an accuracy
claim and it is not a case-specific route table.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_shabad_lock import (  # noqa: E402
    Case,
    load_cached_chunks,
    load_cases,
    load_corpora,
)
from scripts.tune_lock_evidence_fusion import (  # noqa: E402
    FeatureSpec,
    FusionPolicy,
    build_feature_table,
    rank_candidates,
)
from src.shabad_id import ShabadDocTfidf  # noqa: E402


DEFAULT_PAIRED_GT = REPO_ROOT.parent / "live-gurbani-captioning-benchmark-v1" / "test"
DEFAULT_OOS_GT = REPO_ROOT / "eval_data" / "oos_v1" / "assisted_test"
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus_cache"
DEFAULT_ASR_CACHE = REPO_ROOT / "asr_cache"
DEFAULT_OUT = REPO_ROOT / "diagnostics" / "phase3_v6_lock_recency_consistency.md"
DEFAULT_POLICY = "tfidf_45+0.5*chunk_vote_90"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    gt_dir: pathlib.Path
    role: str


@dataclass(frozen=True)
class RecencyRow:
    dataset: str
    case_id: str
    gt_shabad_id: int
    prefix_pred: int | None
    prefix_score: float | None
    prefix_runner_up: int | None
    validation_offset: float
    validation_pred: int | None
    validation_score: float | None
    prefix_validation_score: float | None
    validation_gt_rank: int | None
    missing_cache: bool = False

    @property
    def prefix_ok(self) -> bool:
        return self.prefix_pred == self.gt_shabad_id

    @property
    def validation_ok(self) -> bool:
        return self.validation_pred == self.gt_shabad_id


def parse_fusion_policy(raw: str) -> FusionPolicy:
    """Parse a sparse fusion expression like ``tfidf_45+0.5*chunk_vote_90``."""
    parts = [part.strip() for part in raw.split("+") if part.strip()]
    features: list[tuple[str, float]] = []
    for part in parts:
        if "*" in part:
            weight_s, name = part.split("*", 1)
            weight = float(weight_s.strip())
            name = name.strip()
        else:
            weight = 1.0
            name = part
        if not name:
            raise ValueError(f"empty feature in policy: {raw!r}")
        features.append((name, weight))
    if not features:
        raise ValueError("policy must contain at least one feature")
    return FusionPolicy(tuple(features))


def feature_specs_for_policy(policy: FusionPolicy, *, offset: float = 0.0) -> list[FeatureSpec]:
    """Build FeatureSpecs needed by a policy, optionally shifted later."""
    specs: list[FeatureSpec] = []
    seen: set[str] = set()
    for name, _weight in policy.features:
        if name in seen:
            continue
        seen.add(name)
        if match := re.fullmatch(r"chunk_vote_([0-9]+(?:\.[0-9]+)?)", name):
            specs.append(FeatureSpec(name, "chunk_vote", float(match.group(1)), offset=offset))
        elif match := re.fullmatch(r"tfidf_([0-9]+(?:\.[0-9]+)?)", name):
            specs.append(FeatureSpec(name, "tfidf", float(match.group(1)), offset=offset))
        elif match := re.fullmatch(r"topk3_([0-9]+(?:\.[0-9]+)?)", name):
            specs.append(FeatureSpec(name, "topk:3", float(match.group(1)), offset=offset))
        else:
            raise ValueError(f"unsupported feature in policy: {name!r}")
    return specs


def _rank_lookup(ranked: list[tuple[int, float]]) -> dict[int, int]:
    return {sid: index + 1 for index, (sid, _score) in enumerate(ranked)}


def audit_case(
    *,
    dataset: str,
    case: Case,
    corpora: dict[int, list[dict]],
    chunks,
    policy: FusionPolicy,
    validation_offset: float,
    tfidf_scorer: ShabadDocTfidf,
) -> RecencyRow:
    prefix_table = build_feature_table(
        chunks,
        corpora,
        feature_specs_for_policy(policy, offset=0.0),
        start_t=case.uem_start,
        tfidf_scorer=tfidf_scorer,
    )
    prefix_ranked = rank_candidates(prefix_table, policy)
    prefix_pred, prefix_score = prefix_ranked[0]
    prefix_runner_up = prefix_ranked[1][0] if len(prefix_ranked) > 1 else None

    validation_table = build_feature_table(
        chunks,
        corpora,
        feature_specs_for_policy(policy, offset=validation_offset),
        start_t=case.uem_start,
        tfidf_scorer=tfidf_scorer,
    )
    validation_ranked = rank_candidates(validation_table, policy)
    validation_pred, validation_score = validation_ranked[0]
    validation_scores = dict(validation_ranked)
    validation_ranks = _rank_lookup(validation_ranked)

    return RecencyRow(
        dataset=dataset,
        case_id=case.case_id,
        gt_shabad_id=case.shabad_id,
        prefix_pred=prefix_pred,
        prefix_score=prefix_score,
        prefix_runner_up=prefix_runner_up,
        validation_offset=validation_offset,
        validation_pred=validation_pred,
        validation_score=validation_score,
        prefix_validation_score=validation_scores.get(prefix_pred),
        validation_gt_rank=validation_ranks.get(case.shabad_id),
    )


def should_flag(row: RecencyRow, *, low_support_threshold: float, min_validation_score: float) -> bool:
    if row.missing_cache:
        return False
    if row.prefix_pred is None or row.validation_pred is None:
        return False
    if row.prefix_pred == row.validation_pred:
        return False
    if row.prefix_validation_score is None or row.validation_score is None:
        return False
    return (
        row.prefix_validation_score <= low_support_threshold
        and row.validation_score >= min_validation_score
    )


def run_audit(
    *,
    datasets: list[DatasetSpec],
    corpora: dict[int, list[dict]],
    asr_cache_dir: pathlib.Path,
    asr_tag: str,
    policy: FusionPolicy,
    validation_offset: float,
) -> list[RecencyRow]:
    tfidf_scorer = ShabadDocTfidf(corpora)
    rows: list[RecencyRow] = []
    for dataset in datasets:
        for case in load_cases(dataset.gt_dir):
            chunks = load_cached_chunks(asr_cache_dir, case.video_id, asr_tag)
            if chunks is None:
                rows.append(RecencyRow(
                    dataset=dataset.name,
                    case_id=case.case_id,
                    gt_shabad_id=case.shabad_id,
                    prefix_pred=None,
                    prefix_score=None,
                    prefix_runner_up=None,
                    validation_offset=validation_offset,
                    validation_pred=None,
                    validation_score=None,
                    prefix_validation_score=None,
                    validation_gt_rank=None,
                    missing_cache=True,
                ))
                continue
            rows.append(audit_case(
                dataset=dataset.name,
                case=case,
                corpora=corpora,
                chunks=chunks,
                policy=policy,
                validation_offset=validation_offset,
                tfidf_scorer=tfidf_scorer,
            ))
    return rows


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_markdown(
    *,
    datasets: list[DatasetSpec],
    corpus_dir: pathlib.Path,
    asr_cache_dir: pathlib.Path,
    asr_tag: str,
    policy: FusionPolicy,
    rows: list[RecencyRow],
    validation_offset: float,
    low_support_threshold: float,
    min_validation_score: float,
) -> str:
    flagged = [
        row for row in rows
        if should_flag(
            row,
            low_support_threshold=low_support_threshold,
            min_validation_score=min_validation_score,
        )
    ]
    prefix_correct = sum(1 for row in rows if not row.missing_cache and row.prefix_ok)
    validation_correct = sum(1 for row in rows if not row.missing_cache and row.validation_ok)
    total = sum(1 for row in rows if not row.missing_cache)

    lines = [
        "# Phase 3 lock recency-consistency audit",
        "",
        "**Diagnostic only.** This report compares the current early shabad",
        "lock against a later validation window. It is meant to guide the next",
        "generic architecture step; it is not a promotion-grade accuracy claim",
        "and it must not be turned into a case-specific route table.",
        "",
        "## Inputs",
        "",
        f"- ASR cache: `{asr_cache_dir}`",
        f"- ASR tag: `{asr_tag}`",
        f"- Corpus dir: `{corpus_dir}` ({len(corpora_hint(corpus_dir))} cached shabads)",
        f"- Prefix policy: `{policy.label}`",
        f"- Validation offset: `{validation_offset:g}s` after each case's `uem.start`",
        f"- Flag rule: prefix winner differs from validation winner, prefix winner",
        f"  late-window score <= `{low_support_threshold:g}`, validation winner",
        f"  score >= `{min_validation_score:g}`",
    ]
    for dataset in datasets:
        lines.append(f"- {dataset.name}: `{dataset.gt_dir}` — {dataset.role}")

    lines.extend([
        "",
        "## Summary",
        "",
        f"- Prefix lock accuracy: `{prefix_correct}/{total}`",
        f"- Validation-window lock accuracy: `{validation_correct}/{total}`",
        f"- Flagged recency disagreements: `{len(flagged)}`",
        "",
    ])
    if flagged:
        lines.extend([
            "Flagged rows are not automatic fixes. They are candidates for a",
            "generic delayed/veto policy that must preserve paired + OOS behavior.",
            "",
            "| Dataset | Case | GT | Prefix pred | Prefix score | Late pred | Late score | Prefix late support | Late GT rank |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in flagged:
            lines.append(
                f"| {row.dataset} | {row.case_id} | {row.gt_shabad_id} | "
                f"{row.prefix_pred} | {_fmt(row.prefix_score)} | {row.validation_pred} | "
                f"{_fmt(row.validation_score)} | {_fmt(row.prefix_validation_score)} | "
                f"{row.validation_gt_rank or '—'} |"
            )
    else:
        lines.append("No cases met the recency-disagreement flag rule.")

    lines.extend([
        "",
        "## All Rows",
        "",
        "| Dataset | Case | GT | Prefix pred | Prefix OK | Prefix score | Runner-up | Late pred | Late OK | Late score | Prefix late support | Late GT rank | Flag |",
        "|---|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---|",
    ])
    for row in rows:
        if row.missing_cache:
            lines.append(
                f"| {row.dataset} | {row.case_id} | {row.gt_shabad_id} | — | — | — | — | — | — | — | — | — | MISSING_CACHE |"
            )
            continue
        flag = "FLAG" if row in flagged else ""
        lines.append(
            f"| {row.dataset} | {row.case_id} | {row.gt_shabad_id} | "
            f"{row.prefix_pred} | {'yes' if row.prefix_ok else 'no'} | {_fmt(row.prefix_score)} | "
            f"{row.prefix_runner_up or '—'} | {row.validation_pred} | "
            f"{'yes' if row.validation_ok else 'no'} | {_fmt(row.validation_score)} | "
            f"{_fmt(row.prefix_validation_score)} | {row.validation_gt_rank or '—'} | {flag} |"
        )

    lines.extend([
        "",
        "## Decision",
        "",
        "Do not start full 300h / multi-seed training from this checkpoint.",
        "The completed v6 warm-start improved held-out ASR slightly, but paired",
        "and assisted-OOS frame accuracy stayed flat. The next highest-leverage",
        "step is a generic lock/alignment change that addresses flagged",
        "recency disagreements without weakening OOS behavior.",
        "",
    ])
    return "\n".join(lines)


def corpora_hint(corpus_dir: pathlib.Path) -> list[pathlib.Path]:
    return list(corpus_dir.glob("*.json")) if corpus_dir.exists() else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-gt-dir", type=pathlib.Path, default=DEFAULT_PAIRED_GT)
    parser.add_argument("--oos-gt-dir", type=pathlib.Path, default=DEFAULT_OOS_GT)
    parser.add_argument("--corpus-dir", type=pathlib.Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--asr-cache-dir", type=pathlib.Path, default=DEFAULT_ASR_CACHE)
    parser.add_argument("--asr-tag", default="medium_word")
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--validation-offset", type=float, default=90.0)
    parser.add_argument("--low-support-threshold", type=float, default=0.15)
    parser.add_argument("--min-validation-score", type=float, default=0.5)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    datasets = [
        DatasetSpec("paired", args.paired_gt_dir, "paired benchmark regression/dev labels"),
        DatasetSpec("assisted_oos", args.oos_gt_dir, "machine-assisted OOS silver labels"),
    ]
    corpora = load_corpora(args.corpus_dir)
    if not corpora:
        print(f"error: no corpus files in {args.corpus_dir}", file=sys.stderr)
        return 1
    policy = parse_fusion_policy(args.policy)
    rows = run_audit(
        datasets=datasets,
        corpora=corpora,
        asr_cache_dir=args.asr_cache_dir,
        asr_tag=args.asr_tag,
        policy=policy,
        validation_offset=args.validation_offset,
    )
    report = render_markdown(
        datasets=datasets,
        corpus_dir=args.corpus_dir,
        asr_cache_dir=args.asr_cache_dir,
        asr_tag=args.asr_tag,
        policy=policy,
        rows=rows,
        validation_offset=args.validation_offset,
        low_support_threshold=args.low_support_threshold,
        min_validation_score=args.min_validation_score,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    flagged = [
        row for row in rows
        if should_flag(
            row,
            low_support_threshold=args.low_support_threshold,
            min_validation_score=args.min_validation_score,
        )
    ]
    print(f"wrote: {args.out}")
    print(f"rows={len(rows)} flagged={len(flagged)} policy={policy.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
