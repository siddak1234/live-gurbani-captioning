#!/usr/bin/env python3
"""Tune sparse shabad-lock evidence fusion policies.

Phase 2.12 showed that choosing one scorer/window is not enough. This Phase
2.13 diagnostic asks a stronger candidate-retrieval question:

    If every candidate shabad gets multiple evidence features across windows
    and scoring families, can a small generic fusion recover the correct lock?

This script is silver-learning only. It uses paired benchmark labels plus
machine-assisted OOS labels as development targets; it does not produce a
promotion-grade OOS claim.
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import sys
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_shabad_lock import (  # noqa: E402
    Case,
    load_cached_chunks,
    load_cases,
    load_corpora,
    parse_lookbacks,
)
from scripts.tune_shabad_lock_policy import DatasetScore, DatasetSpec  # noqa: E402
from src.asr import AsrChunk  # noqa: E402
from src.matcher import score_chunk  # noqa: E402
from src.shabad_id import ShabadDocTfidf, buffer_text  # noqa: E402


DEFAULT_PAIRED_GT = REPO_ROOT.parent / "live-gurbani-captioning-benchmark-v1" / "test"
DEFAULT_OOS_GT = REPO_ROOT / "eval_data" / "oos_v1" / "assisted_test"
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus_cache"
DEFAULT_ASR_CACHE = REPO_ROOT / "asr_cache"
DEFAULT_OUT = REPO_ROOT / "diagnostics" / "phase2_13_lock_evidence_fusion.md"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    aggregate: str
    window: float


@dataclass(frozen=True)
class FusionPolicy:
    features: tuple[tuple[str, float], ...]

    @property
    def label(self) -> str:
        return " + ".join(
            f"{weight:g}*{name}" if weight != 1.0 else name
            for name, weight in self.features
        )


@dataclass(frozen=True)
class CandidateDecision:
    dataset: str
    case_id: str
    gt_shabad_id: int
    predicted_shabad_id: int | None
    score: float | None
    runner_up_id: int | None
    runner_up_score: float | None
    gt_rank: int | None
    ok: bool
    missing_cache: bool = False


@dataclass(frozen=True)
class FusionResult:
    policy: FusionPolicy
    by_dataset: dict[str, DatasetScore]
    decisions: list[CandidateDecision]

    @property
    def macro_accuracy(self) -> float:
        if not self.by_dataset:
            return 0.0
        return sum(score.accuracy for score in self.by_dataset.values()) / len(self.by_dataset)

    @property
    def paired_accuracy(self) -> float:
        return self.by_dataset.get("paired", DatasetScore(0, 0, 0)).accuracy

    @property
    def oos_accuracy(self) -> float:
        return self.by_dataset.get("assisted_oos", DatasetScore(0, 0, 0)).accuracy


def make_feature_specs(lookbacks: list[float]) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for window in lookbacks:
        specs.extend([
            FeatureSpec(f"chunk_vote_{window:g}", "chunk_vote", window),
            FeatureSpec(f"tfidf_{window:g}", "tfidf", window),
            FeatureSpec(f"topk3_{window:g}", "topk:3", window),
        ])
    return specs


def chunk_vote_scores(
    chunks: list[AsrChunk],
    corpora: dict[int, list[dict]],
    *,
    window: float,
) -> dict[int, float]:
    end_t = float(window)
    scores: dict[int, float] = {sid: 0.0 for sid in corpora}
    for chunk in chunks:
        if float(chunk.start) >= end_t:
            break
        if float(chunk.end) <= 0.0:
            continue
        best_sid: int | None = None
        best_score = 0.0
        for sid, lines in corpora.items():
            line_scores = score_chunk(chunk.text, lines, ratio="WRatio")
            score = max(line_scores) if line_scores else 0.0
            if score > best_score:
                best_sid = sid
                best_score = score
        if best_sid is not None:
            scores[best_sid] += best_score
    return scores


def topk_scores(
    chunks: list[AsrChunk],
    corpora: dict[int, list[dict]],
    *,
    window: float,
    k: int = 3,
) -> dict[int, float]:
    buf = buffer_text(chunks, start_t=0.0, lookback_seconds=window)
    out: dict[int, float] = {}
    for sid, lines in corpora.items():
        line_scores = score_chunk(buf, lines, ratio="WRatio")
        out[sid] = sum(sorted(line_scores, reverse=True)[:k]) if line_scores else 0.0
    return out


def tfidf_scores(
    chunks: list[AsrChunk],
    tfidf_scorer: ShabadDocTfidf,
    *,
    window: float,
) -> dict[int, float]:
    buf = buffer_text(chunks, start_t=0.0, lookback_seconds=window)
    return tfidf_scorer.score(buf)


def build_feature_table(
    chunks: list[AsrChunk],
    corpora: dict[int, list[dict]],
    specs: list[FeatureSpec],
    *,
    tfidf_scorer: ShabadDocTfidf | None = None,
) -> dict[int, dict[str, float]]:
    """Return sid -> feature_name -> per-case normalized score."""
    raw_by_feature: dict[str, dict[int, float]] = {}
    tfidf_scorer = tfidf_scorer or ShabadDocTfidf(corpora)
    for spec in specs:
        if spec.aggregate == "chunk_vote":
            raw = chunk_vote_scores(chunks, corpora, window=spec.window)
        elif spec.aggregate == "tfidf":
            raw = tfidf_scores(chunks, tfidf_scorer, window=spec.window)
        elif spec.aggregate == "topk:3":
            raw = topk_scores(chunks, corpora, window=spec.window, k=3)
        else:
            raise ValueError(f"unknown aggregate: {spec.aggregate}")
        max_score = max(raw.values()) if raw else 0.0
        if max_score > 0:
            raw_by_feature[spec.name] = {sid: score / max_score for sid, score in raw.items()}
        else:
            raw_by_feature[spec.name] = {sid: 0.0 for sid in corpora}

    table: dict[int, dict[str, float]] = {sid: {} for sid in corpora}
    for feature_name, scores in raw_by_feature.items():
        for sid in corpora:
            table[sid][feature_name] = float(scores.get(sid, 0.0))
    return table


def rank_candidates(
    feature_table: dict[int, dict[str, float]],
    policy: FusionPolicy,
) -> list[tuple[int, float]]:
    ranked: list[tuple[int, float]] = []
    for sid, features in feature_table.items():
        score = sum(float(features.get(name, 0.0)) * float(weight) for name, weight in policy.features)
        ranked.append((sid, score))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked


def make_policy_grid(
    feature_specs: list[FeatureSpec],
    *,
    max_features: int,
    weight_values: tuple[float, ...] = (1.0, 0.5, 2.0),
) -> list[FusionPolicy]:
    policies: list[FusionPolicy] = []
    names = [spec.name for spec in feature_specs]
    for size in range(1, max_features + 1):
        for combo in itertools.combinations(names, size):
            if size == 1:
                policies.append(FusionPolicy(((combo[0], 1.0),)))
                continue
            for weights in itertools.product(weight_values, repeat=size):
                # Collapse all-equal variants to one representative so the
                # report is easier to read.
                if len(set(weights)) == 1 and weights[0] != 1.0:
                    continue
                policies.append(FusionPolicy(tuple(zip(combo, weights))))
    return policies


def evaluate_policy(
    *,
    policy: FusionPolicy,
    datasets: list[DatasetSpec],
    cases_by_dataset: dict[str, list[Case]],
    feature_tables: dict[tuple[str, str], dict[int, dict[str, float]]],
    missing_cache: set[tuple[str, str]],
) -> FusionResult:
    decisions: list[CandidateDecision] = []
    by_dataset: dict[str, DatasetScore] = {}
    for dataset in datasets:
        rows: list[CandidateDecision] = []
        for case in cases_by_dataset[dataset.name]:
            key = (dataset.name, case.case_id)
            if key in missing_cache:
                rows.append(CandidateDecision(
                    dataset=dataset.name,
                    case_id=case.case_id,
                    gt_shabad_id=case.shabad_id,
                    predicted_shabad_id=None,
                    score=None,
                    runner_up_id=None,
                    runner_up_score=None,
                    gt_rank=None,
                    ok=False,
                    missing_cache=True,
                ))
                continue
            ranked = rank_candidates(feature_tables[key], policy)
            pred_id, pred_score = ranked[0]
            runner_id, runner_score = ranked[1] if len(ranked) > 1 else (None, 0.0)
            rank_lookup = {sid: index + 1 for index, (sid, _) in enumerate(ranked)}
            rows.append(CandidateDecision(
                dataset=dataset.name,
                case_id=case.case_id,
                gt_shabad_id=case.shabad_id,
                predicted_shabad_id=pred_id,
                score=pred_score,
                runner_up_id=runner_id,
                runner_up_score=runner_score,
                gt_rank=rank_lookup.get(case.shabad_id),
                ok=int(pred_id) == int(case.shabad_id),
            ))
        decisions.extend(rows)
        missing = sum(1 for row in rows if row.missing_cache)
        total = len(rows) - missing
        by_dataset[dataset.name] = DatasetScore(
            correct=sum(1 for row in rows if row.ok),
            total=total,
            missing_cache=missing,
        )
    return FusionResult(policy=policy, by_dataset=by_dataset, decisions=decisions)


def rank_results(results: list[FusionResult]) -> list[FusionResult]:
    return sorted(
        results,
        key=lambda result: (
            result.macro_accuracy,
            result.paired_accuracy,
            result.oos_accuracy,
            -len(result.policy.features),
            result.policy.label,
        ),
        reverse=True,
    )


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _fmt(score: DatasetScore) -> str:
    return f"{score.correct}/{score.total} ({100 * score.accuracy:.1f}%)"


def render_markdown(
    *,
    datasets: list[DatasetSpec],
    corpus_dir: pathlib.Path,
    asr_cache_dir: pathlib.Path,
    asr_tag: str,
    feature_specs: list[FeatureSpec],
    ranked: list[FusionResult],
    top_n: int,
) -> str:
    best = ranked[0] if ranked else None
    lines: list[str] = [
        "# Phase 2.13 lock-evidence fusion",
        "",
        "**Diagnostic only.** This report searches sparse evidence-fusion",
        "policies for shabad locking using paired benchmark labels and",
        "machine-assisted OOS silver labels. It is not a production accuracy",
        "claim.",
        "",
        "## Inputs",
        "",
        f"- ASR cache: `{_display_path(asr_cache_dir)}`",
        f"- ASR tag: `{asr_tag}`",
        f"- Corpus dir: `{_display_path(corpus_dir)}` ({len(list(corpus_dir.glob('*.json')))} cached shabads)",
        f"- Features: {', '.join(spec.name for spec in feature_specs)}",
    ]
    for dataset in datasets:
        lines.append(f"- {dataset.name}: `{_display_path(dataset.gt_dir)}` — {dataset.role}")

    lines.extend([
        "",
        "## Top fusion policies",
        "",
        "| Rank | Policy | Macro | Paired | Assisted OOS |",
        "|---:|---|---:|---:|---:|",
    ])
    for index, result in enumerate(ranked[:top_n], start=1):
        paired = result.by_dataset.get("paired", DatasetScore(0, 0, 0))
        oos = result.by_dataset.get("assisted_oos", DatasetScore(0, 0, 0))
        lines.append(
            f"| {index} | `{result.policy.label}` | {100 * result.macro_accuracy:.1f}% | "
            f"{_fmt(paired)} | {_fmt(oos)} |"
        )

    if ranked:
        best_oos = max(ranked, key=lambda r: (r.oos_accuracy, r.paired_accuracy, r.macro_accuracy))
        paired_safe = [r for r in ranked if r.paired_accuracy >= 0.75]
        best_oos_safe = max(paired_safe, key=lambda r: (r.oos_accuracy, r.macro_accuracy)) if paired_safe else None
        lines.extend([
            "",
            "## Guardrail views",
            "",
            "| View | Policy | Macro | Paired | Assisted OOS |",
            "|---|---|---:|---:|---:|",
        ])
        for name, result in [
            ("best macro", best),
            ("best assisted OOS", best_oos),
            ("best assisted OOS with paired >=75%", best_oos_safe),
        ]:
            if result is None:
                lines.append(f"| {name} | — | — | — | — |")
                continue
            paired = result.by_dataset.get("paired", DatasetScore(0, 0, 0))
            oos = result.by_dataset.get("assisted_oos", DatasetScore(0, 0, 0))
            lines.append(
                f"| {name} | `{result.policy.label}` | {100 * result.macro_accuracy:.1f}% | "
                f"{_fmt(paired)} | {_fmt(oos)} |"
            )

    if best:
        paired = best.by_dataset.get("paired", DatasetScore(0, 0, 0))
        oos = best.by_dataset.get("assisted_oos", DatasetScore(0, 0, 0))
        lines.extend([
            "",
            "## Current silver decision",
            "",
            f"Best macro fusion: `{best.policy.label}`.",
            "",
            f"- Paired lock accuracy: {_fmt(paired)}",
            f"- Assisted-OOS lock accuracy: {_fmt(oos)}",
            f"- Silver macro objective: {100 * best.macro_accuracy:.1f}%",
            "",
            "This is evidence for the next opt-in runtime experiment only if it",
            "improves over Phase 2.12 without paired regression. Gold OOS remains",
            "required before promotion.",
            "",
            "## Best-policy per-case decisions",
            "",
            "| Dataset | Case | GT | Pred | Score | Runner-up | GT rank | Result |",
            "|---|---|---:|---:|---:|---|---:|---|",
        ])
        for row in best.decisions:
            if row.missing_cache:
                lines.append(
                    f"| {row.dataset} | {row.case_id} | {row.gt_shabad_id} | — | — | — | — | MISSING_CACHE |"
                )
                continue
            runner = "—"
            if row.runner_up_id is not None and row.runner_up_score is not None:
                runner = f"{row.runner_up_id} ({row.runner_up_score:.3f})"
            status = "OK" if row.ok else "BAD"
            lines.append(
                f"| {row.dataset} | {row.case_id} | {row.gt_shabad_id} | {row.predicted_shabad_id} | "
                f"{row.score:.3f} | {runner} | {row.gt_rank or '—'} | {status} |"
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-gt-dir", type=pathlib.Path, default=DEFAULT_PAIRED_GT)
    parser.add_argument("--oos-gt-dir", type=pathlib.Path, default=DEFAULT_OOS_GT)
    parser.add_argument("--corpus-dir", type=pathlib.Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--asr-cache-dir", type=pathlib.Path, default=DEFAULT_ASR_CACHE)
    parser.add_argument("--asr-tag", default="medium_word")
    parser.add_argument("--lookbacks", default="30,45,60,90")
    parser.add_argument("--max-features", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    datasets = [
        DatasetSpec("paired", args.paired_gt_dir, "paired benchmark regression/dev labels"),
        DatasetSpec("assisted_oos", args.oos_gt_dir, "machine-assisted OOS silver labels"),
    ]
    cases_by_dataset = {dataset.name: load_cases(dataset.gt_dir) for dataset in datasets}
    for dataset in datasets:
        if not cases_by_dataset[dataset.name]:
            print(f"error: no GT files in {dataset.gt_dir}", file=sys.stderr)
            return 1
    corpora = load_corpora(args.corpus_dir)
    if not corpora:
        print(f"error: no corpus files in {args.corpus_dir}", file=sys.stderr)
        return 1

    feature_specs = make_feature_specs(parse_lookbacks(args.lookbacks))
    tfidf_scorer = ShabadDocTfidf(corpora)
    feature_tables: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    missing_cache: set[tuple[str, str]] = set()
    for dataset in datasets:
        for case in cases_by_dataset[dataset.name]:
            key = (dataset.name, case.case_id)
            chunks = load_cached_chunks(args.asr_cache_dir, case.video_id, args.asr_tag)
            if chunks is None:
                missing_cache.add(key)
                continue
            feature_tables[key] = build_feature_table(
                chunks,
                corpora,
                feature_specs,
                tfidf_scorer=tfidf_scorer,
            )

    policies = make_policy_grid(feature_specs, max_features=args.max_features)
    ranked = rank_results([
        evaluate_policy(
            policy=policy,
            datasets=datasets,
            cases_by_dataset=cases_by_dataset,
            feature_tables=feature_tables,
            missing_cache=missing_cache,
        )
        for policy in policies
    ])
    markdown = render_markdown(
        datasets=datasets,
        corpus_dir=args.corpus_dir,
        asr_cache_dir=args.asr_cache_dir,
        asr_tag=args.asr_tag,
        feature_specs=feature_specs,
        ranked=ranked,
        top_n=args.top_n,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown)
    print(f"wrote: {args.out}")
    if ranked:
        best = ranked[0]
        print(
            f"best: {best.policy.label} | macro={100 * best.macro_accuracy:.1f}% "
            f"paired={100 * best.paired_accuracy:.1f}% "
            f"assisted_oos={100 * best.oos_accuracy:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
