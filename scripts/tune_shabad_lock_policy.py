#!/usr/bin/env python3
"""Tune shabad-lock policy variants on paired + silver OOS labels.

This is a Phase 2.12 silver-learning tool. It deliberately does not run ASR and
does not score line timing. It reuses cached ASR transcripts plus the current
corpus cache to answer:

    Which generic lock policy best balances paired-benchmark locks and
    machine-assisted OOS silver locks?

The output is a development report, not a production validation result.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_shabad_lock import (  # noqa: E402
    Case,
    identify_hybrid,
    load_cached_chunks,
    load_cases,
    load_corpora,
    parse_csv,
)
from src.shabad_id import ShabadIdResult, identify_shabad  # noqa: E402


DEFAULT_PAIRED_GT = REPO_ROOT.parent / "live-gurbani-captioning-benchmark-v1" / "test"
DEFAULT_OOS_GT = REPO_ROOT / "eval_data" / "oos_v1" / "assisted_test"
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus_cache"
DEFAULT_ASR_CACHE = REPO_ROOT / "asr_cache"
DEFAULT_OUT = REPO_ROOT / "diagnostics" / "phase2_12_silver_lock_policy.md"
DEFAULT_WINDOW_GRID = "30;30,45;30,45,60;30,45,60,90;45;45,60;45,60,90;60;60,90;90"
DEFAULT_AGGREGATES = "chunk_vote,tfidf,topk:3,tfidf_then_topk3"
DEFAULT_MIN_SCORES = "0,1,10,25,50,85,120,170"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    gt_dir: pathlib.Path
    role: str


@dataclass(frozen=True)
class Policy:
    aggregate: str
    windows: tuple[float, ...]
    min_lock_score: float

    @property
    def label(self) -> str:
        windows = ",".join(f"{w:g}" for w in self.windows)
        return f"{self.aggregate}@{windows}s|min={self.min_lock_score:g}"


@dataclass(frozen=True)
class Decision:
    dataset: str
    case_id: str
    gt_shabad_id: int
    predicted_shabad_id: int | None
    score: float | None
    runner_up_id: int | None
    runner_up_score: float | None
    selected_window: float | None
    mode: str
    ok: bool
    missing_cache: bool = False


@dataclass(frozen=True)
class DatasetScore:
    correct: int
    total: int
    missing_cache: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass(frozen=True)
class PolicyResult:
    policy: Policy
    by_dataset: dict[str, DatasetScore]
    decisions: list[Decision]

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


def parse_window_grid(raw: str) -> list[tuple[float, ...]]:
    """Parse semicolon-delimited window sequences.

    Example: ``"30;30,45;30,45,60"``.
    """
    out: list[tuple[float, ...]] = []
    for group in raw.split(";"):
        group = group.strip()
        if not group:
            continue
        windows = tuple(float(part.strip()) for part in group.split(",") if part.strip())
        if windows:
            out.append(windows)
    return out


def parse_score_grid(raw: str) -> list[float]:
    return [float(part) for part in parse_csv(raw)]


def make_policy_grid(
    *,
    aggregates: list[str],
    window_sets: list[tuple[float, ...]],
    min_scores: list[float],
) -> list[Policy]:
    policies: list[Policy] = []
    for aggregate in aggregates:
        for windows in window_sets:
            for min_score in min_scores:
                policies.append(Policy(
                    aggregate=aggregate,
                    windows=windows,
                    min_lock_score=min_score,
                ))
    return policies


def _identify(
    *,
    chunks,
    corpora: dict[int, list[dict]],
    start_t: float,
    lookback_seconds: float,
    aggregate: str,
    tfidf_min_score: float,
    tfidf_min_margin: float,
) -> tuple[ShabadIdResult, str]:
    if aggregate == "tfidf_then_topk3":
        return identify_hybrid(
            chunks,
            corpora,
            start_t=start_t,
            lookback_seconds=lookback_seconds,
            tfidf_min_score=tfidf_min_score,
            tfidf_min_margin=tfidf_min_margin,
        )
    return (
        identify_shabad(
            chunks,
            corpora,
            start_t=start_t,
            lookback_seconds=lookback_seconds,
            aggregate=aggregate,
            ratio="WRatio",
        ),
        aggregate,
    )


def decide_case(
    *,
    dataset_name: str,
    case: Case,
    corpora: dict[int, list[dict]],
    asr_cache_dir: pathlib.Path,
    asr_tag: str,
    policy: Policy,
    tfidf_min_score: float,
    tfidf_min_margin: float,
) -> Decision:
    chunks = load_cached_chunks(asr_cache_dir, case.video_id, asr_tag)
    if chunks is None:
        return Decision(
            dataset=dataset_name,
            case_id=case.case_id,
            gt_shabad_id=case.shabad_id,
            predicted_shabad_id=None,
            score=None,
            runner_up_id=None,
            runner_up_score=None,
            selected_window=None,
            mode=policy.aggregate,
            ok=False,
            missing_cache=True,
        )

    final_result: ShabadIdResult | None = None
    final_mode = policy.aggregate
    final_window = policy.windows[-1]
    for index, window in enumerate(policy.windows):
        result, mode = _identify(
            chunks=chunks,
            corpora=corpora,
            start_t=case.uem_start,
            lookback_seconds=window,
            aggregate=policy.aggregate,
            tfidf_min_score=tfidf_min_score,
            tfidf_min_margin=tfidf_min_margin,
        )
        final_result = result
        final_mode = mode
        final_window = window
        if index == len(policy.windows) - 1 or result.score >= policy.min_lock_score:
            break

    assert final_result is not None
    return Decision(
        dataset=dataset_name,
        case_id=case.case_id,
        gt_shabad_id=case.shabad_id,
        predicted_shabad_id=int(final_result.shabad_id),
        score=float(final_result.score),
        runner_up_id=None if final_result.runner_up_id is None else int(final_result.runner_up_id),
        runner_up_score=float(final_result.runner_up_score or 0.0),
        selected_window=final_window,
        mode=final_mode,
        ok=int(final_result.shabad_id) == int(case.shabad_id),
    )


def evaluate_policy(
    *,
    policy: Policy,
    datasets: list[DatasetSpec],
    cases_by_dataset: dict[str, list[Case]],
    corpora: dict[int, list[dict]],
    asr_cache_dir: pathlib.Path,
    asr_tag: str,
    tfidf_min_score: float,
    tfidf_min_margin: float,
) -> PolicyResult:
    decisions: list[Decision] = []
    by_dataset: dict[str, DatasetScore] = {}
    for dataset in datasets:
        rows = [
            decide_case(
                dataset_name=dataset.name,
                case=case,
                corpora=corpora,
                asr_cache_dir=asr_cache_dir,
                asr_tag=asr_tag,
                policy=policy,
                tfidf_min_score=tfidf_min_score,
                tfidf_min_margin=tfidf_min_margin,
            )
            for case in cases_by_dataset[dataset.name]
        ]
        decisions.extend(rows)
        missing = sum(1 for row in rows if row.missing_cache)
        total = len(rows) - missing
        by_dataset[dataset.name] = DatasetScore(
            correct=sum(1 for row in rows if row.ok),
            total=total,
            missing_cache=missing,
        )
    return PolicyResult(policy=policy, by_dataset=by_dataset, decisions=decisions)


def rank_results(results: list[PolicyResult]) -> list[PolicyResult]:
    return sorted(
        results,
        key=lambda result: (
            result.macro_accuracy,
            result.paired_accuracy,
            result.oos_accuracy,
            -len(result.policy.windows),
            -result.policy.min_lock_score,
            result.policy.aggregate,
        ),
        reverse=True,
    )


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _fmt_accuracy(score: DatasetScore) -> str:
    return f"{score.correct}/{score.total} ({100 * score.accuracy:.1f}%)"


def render_markdown(
    *,
    datasets: list[DatasetSpec],
    corpus_dir: pathlib.Path,
    asr_cache_dir: pathlib.Path,
    asr_tag: str,
    ranked: list[PolicyResult],
    top_n: int,
) -> str:
    best = ranked[0] if ranked else None
    corpus_count = len(list(corpus_dir.glob("*.json")))
    lines: list[str] = [
        "# Phase 2.12 silver lock-policy tuning",
        "",
        "**Diagnostic only.** This report tunes shabad-lock policy variants using",
        "the paired benchmark plus machine-assisted OOS labels. It is a learning",
        "signal, not a production validation claim and not a replacement for gold",
        "OOS correction.",
        "",
        "## Inputs",
        "",
        f"- ASR cache: `{_display_path(asr_cache_dir)}`",
        f"- ASR tag: `{asr_tag}`",
        f"- Corpus dir: `{_display_path(corpus_dir)}` ({corpus_count} cached shabads)",
    ]
    for dataset in datasets:
        lines.append(f"- {dataset.name}: `{_display_path(dataset.gt_dir)}` — {dataset.role}")
    lines.extend([
        "",
        "## Top policies",
        "",
        "| Rank | Policy | Macro | Paired | Assisted OOS |",
        "|---:|---|---:|---:|---:|",
    ])
    for index, result in enumerate(ranked[:top_n], start=1):
        paired = result.by_dataset.get("paired", DatasetScore(0, 0, 0))
        oos = result.by_dataset.get("assisted_oos", DatasetScore(0, 0, 0))
        lines.append(
            f"| {index} | `{result.policy.label}` | {100 * result.macro_accuracy:.1f}% | "
            f"{_fmt_accuracy(paired)} | {_fmt_accuracy(oos)} |"
        )

    if ranked:
        def best_for(key_name: str, candidates: list[PolicyResult]) -> PolicyResult | None:
            if not candidates:
                return None
            if key_name == "paired":
                return max(candidates, key=lambda r: (r.paired_accuracy, r.oos_accuracy, r.macro_accuracy))
            if key_name == "oos":
                return max(candidates, key=lambda r: (r.oos_accuracy, r.paired_accuracy, r.macro_accuracy))
            return max(candidates, key=lambda r: r.macro_accuracy)

        guardrail_candidates = [result for result in ranked if result.paired_accuracy >= 0.75]
        views = [
            ("best macro", best_for("macro", ranked)),
            ("best paired", best_for("paired", ranked)),
            ("best assisted OOS", best_for("oos", ranked)),
            ("best assisted OOS with paired >=75%", best_for("oos", guardrail_candidates)),
        ]
        lines.extend([
            "",
            "## Guardrail views",
            "",
            "| View | Policy | Macro | Paired | Assisted OOS |",
            "|---|---|---:|---:|---:|",
        ])
        for name, result in views:
            if result is None:
                lines.append(f"| {name} | — | — | — | — |")
                continue
            paired = result.by_dataset.get("paired", DatasetScore(0, 0, 0))
            oos = result.by_dataset.get("assisted_oos", DatasetScore(0, 0, 0))
            lines.append(
                f"| {name} | `{result.policy.label}` | {100 * result.macro_accuracy:.1f}% | "
                f"{_fmt_accuracy(paired)} | {_fmt_accuracy(oos)} |"
            )

    if best is not None:
        paired = best.by_dataset.get("paired", DatasetScore(0, 0, 0))
        oos = best.by_dataset.get("assisted_oos", DatasetScore(0, 0, 0))
        lines.extend([
            "",
            "## Current silver decision",
            "",
            f"Best macro policy: `{best.policy.label}`.",
            "",
            f"- Paired lock accuracy: {_fmt_accuracy(paired)}",
            f"- Assisted-OOS lock accuracy: {_fmt_accuracy(oos)}",
            f"- Silver macro objective: {100 * best.macro_accuracy:.1f}%",
            "",
        ])
        if paired.accuracy < 1.0:
            lines.extend([
                "Do **not** promote this as a runtime default yet. The paired set is",
                "still a regression guardrail, and this search is intentionally tiny.",
                "The useful output is the policy shape to test next under full frame",
                "scoring, not a final architecture decision.",
                "",
            ])
        else:
            lines.extend([
                "This policy clears paired shabad-lock regression in this diagnostic.",
                "It is still silver-tuned and must be checked with full frame scoring",
                "plus gold OOS before promotion.",
                "",
            ])

        lines.extend([
            "## Best-policy per-case decisions",
            "",
            "| Dataset | Case | GT | Pred | Window | Mode | Score | Runner-up | Result |",
            "|---|---|---:|---:|---:|---|---:|---|---|",
        ])
        for row in best.decisions:
            if row.missing_cache:
                lines.append(
                    f"| {row.dataset} | {row.case_id} | {row.gt_shabad_id} | — | — | "
                    f"{row.mode} | — | — | MISSING_CACHE |"
                )
                continue
            runner = "—"
            if row.runner_up_id is not None and row.runner_up_score is not None:
                runner = f"{row.runner_up_id} ({row.runner_up_score:.1f})"
            status = "OK" if row.ok else "BAD"
            lines.append(
                f"| {row.dataset} | {row.case_id} | {row.gt_shabad_id} | "
                f"{row.predicted_shabad_id} | {row.selected_window:g}s | `{row.mode}` | "
                f"{row.score:.1f} | {runner} | {status} |"
            )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "This report allows us to keep learning without waiting on human OOS",
        "correction. It should be used to choose the next runtime experiment.",
        "It should not be used to report final model quality.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-gt-dir", type=pathlib.Path, default=DEFAULT_PAIRED_GT)
    parser.add_argument("--oos-gt-dir", type=pathlib.Path, default=DEFAULT_OOS_GT)
    parser.add_argument("--corpus-dir", type=pathlib.Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--asr-cache-dir", type=pathlib.Path, default=DEFAULT_ASR_CACHE)
    parser.add_argument("--asr-tag", default="medium_word")
    parser.add_argument("--window-grid", default=DEFAULT_WINDOW_GRID)
    parser.add_argument("--aggregates", default=DEFAULT_AGGREGATES)
    parser.add_argument("--min-lock-scores", default=DEFAULT_MIN_SCORES)
    parser.add_argument("--tfidf-min-score", type=float, default=10.0)
    parser.add_argument("--tfidf-min-margin", type=float, default=1.0)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    datasets = [
        DatasetSpec("paired", args.paired_gt_dir, "paired benchmark regression/dev labels"),
        DatasetSpec("assisted_oos", args.oos_gt_dir, "machine-assisted OOS silver labels"),
    ]
    cases_by_dataset: dict[str, list[Case]] = {}
    for dataset in datasets:
        cases = load_cases(dataset.gt_dir)
        if not cases:
            print(f"error: no GT files in {dataset.gt_dir}", file=sys.stderr)
            return 1
        cases_by_dataset[dataset.name] = cases

    corpora = load_corpora(args.corpus_dir)
    if not corpora:
        print(f"error: no corpus files in {args.corpus_dir}", file=sys.stderr)
        return 1

    policies = make_policy_grid(
        aggregates=parse_csv(args.aggregates),
        window_sets=parse_window_grid(args.window_grid),
        min_scores=parse_score_grid(args.min_lock_scores),
    )
    results = [
        evaluate_policy(
            policy=policy,
            datasets=datasets,
            cases_by_dataset=cases_by_dataset,
            corpora=corpora,
            asr_cache_dir=args.asr_cache_dir,
            asr_tag=args.asr_tag,
            tfidf_min_score=args.tfidf_min_score,
            tfidf_min_margin=args.tfidf_min_margin,
        )
        for policy in policies
    ]
    ranked = rank_results(results)
    markdown = render_markdown(
        datasets=datasets,
        corpus_dir=args.corpus_dir,
        asr_cache_dir=args.asr_cache_dir,
        asr_tag=args.asr_tag,
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
