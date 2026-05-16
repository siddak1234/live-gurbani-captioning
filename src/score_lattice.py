"""Score-lattice diagnostics for post-lock alignment.

This module is not a production aligner. It records the evidence a future
aligner will consume: ASR chunks, per-line matcher scores, simple stay-bias
choices, and the GT line at each chunk midpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.matcher import TfidfScorer, normalize, score_chunk


@dataclass(frozen=True)
class TopScore:
    line_idx: int
    score: float


@dataclass(frozen=True)
class ScoreLatticeRow:
    start: float
    end: float
    text: str
    gt_line_idx: int | None
    best_line_idx: int | None
    stay_line_idx: int | None
    top_scores: list[TopScore]


def line_at_midpoint(
    start: float,
    end: float,
    gt_segments: list[dict[str, Any]],
) -> int | None:
    """Return the GT line whose interval contains the chunk midpoint."""
    midpoint = (float(start) + float(end)) / 2.0
    for segment in gt_segments:
        if float(segment["start"]) <= midpoint < float(segment["end"]):
            return int(segment["line_idx"])
    return None


def choose_stay_bias_path(
    chunks_with_scores: list[tuple[float, float, list[float]]],
    *,
    stay_margin: float,
    score_threshold: float,
) -> list[int | None]:
    """Return the per-chunk choices made by ``smooth_with_stay_bias``.

    ``smooth_with_stay_bias`` emits collapsed segments. For diagnostics we need
    the chunk-level choices before collapse so we can compare them with GT.
    """
    out: list[int | None] = []
    prev_line: int | None = None
    for _, _, scores in chunks_with_scores:
        if not scores:
            out.append(None)
            continue
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_score = scores[best_idx]
        chosen: int | None = best_idx
        if (
            prev_line is not None
            and prev_line < len(scores)
            and best_score - scores[prev_line] < stay_margin
        ):
            chosen = prev_line
        if chosen is None or scores[chosen] < score_threshold:
            out.append(None)
            continue
        out.append(chosen)
        prev_line = chosen
    return out


def build_score_lattice(
    chunks: list[Any],
    lines: list[dict[str, Any]],
    gt_segments: list[dict[str, Any]],
    *,
    ratio: str = "WRatio",
    blend: dict[str, float] | None = None,
    stay_margin: float = 6.0,
    score_threshold: float = 0.0,
    top_k: int = 5,
) -> list[ScoreLatticeRow]:
    """Build chunk-level evidence rows for a locked shabad."""
    tfidf = None
    if blend and "tfidf" in blend:
        tfidf = TfidfScorer([normalize(line.get("transliteration_english", "")) for line in lines])

    scored: list[tuple[float, float, list[float]]] = []
    texts: list[str] = []
    for chunk in chunks:
        text = getattr(chunk, "text", None)
        start = getattr(chunk, "start", None)
        end = getattr(chunk, "end", None)
        if text is None and isinstance(chunk, dict):
            text = chunk.get("text", "")
            start = chunk.get("start", 0.0)
            end = chunk.get("end", 0.0)
        text = str(text or "")
        scores = score_chunk(text, lines, ratio=ratio, blend=blend, tfidf=tfidf)
        scored.append((float(start), float(end), scores))
        texts.append(text)

    stay_path = choose_stay_bias_path(
        scored,
        stay_margin=stay_margin,
        score_threshold=score_threshold,
    )

    rows: list[ScoreLatticeRow] = []
    for (start, end, scores), text, stay_line_idx in zip(scored, texts, stay_path):
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        top_scores = [TopScore(line_idx=i, score=float(scores[i])) for i in top_indices]
        best_line_idx = top_scores[0].line_idx if top_scores else None
        rows.append(ScoreLatticeRow(
            start=start,
            end=end,
            text=text,
            gt_line_idx=line_at_midpoint(start, end, gt_segments),
            best_line_idx=best_line_idx,
            stay_line_idx=stay_line_idx,
            top_scores=top_scores,
        ))
    return rows


def score_lattice_summary(rows: list[ScoreLatticeRow]) -> dict[str, int]:
    """Small aggregate useful in run summaries."""
    with_gt = [row for row in rows if row.gt_line_idx is not None]
    return {
        "chunks": len(rows),
        "chunks_with_gt": len(with_gt),
        "best_matches_gt": sum(1 for row in with_gt if row.best_line_idx == row.gt_line_idx),
        "stay_matches_gt": sum(1 for row in with_gt if row.stay_line_idx == row.gt_line_idx),
        "null_gt_chunks": len(rows) - len(with_gt),
    }
