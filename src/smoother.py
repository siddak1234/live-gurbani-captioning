"""Merge per-chunk matches into submission segments.

Initial implementation: contiguous same-`line_idx` matches collapse into one
segment; gaps and `None` matches break the run. Plenty of room to add
HMM-style smoothing later (Path B territory).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    start: float
    end: float
    line_idx: int


def smooth(matches: list[tuple[float, float, int | None]]) -> list[Segment]:
    """Collapse consecutive same-line matches into segments. Drop `None`s."""
    segments: list[Segment] = []
    cur: Segment | None = None
    for start, end, line_idx in matches:
        if line_idx is None:
            if cur is not None:
                segments.append(cur)
                cur = None
            continue
        if cur is not None and cur.line_idx == line_idx:
            cur.end = end
        else:
            if cur is not None:
                segments.append(cur)
            cur = Segment(start=start, end=end, line_idx=line_idx)
    if cur is not None:
        segments.append(cur)
    return segments


def smooth_with_stay_bias(
    chunks_with_scores: list[tuple[float, float, list[float]]],
    *,
    stay_margin: float = 5.0,
    score_threshold: float = 0.0,
) -> list[Segment]:
    """Like `smooth()`, but biases toward continuity with the previous line.

    For each chunk, picks `argmax(scores)` unless the previously-emitted line
    scores within `stay_margin` of the top — in which case it stays on the
    previous line. Targets oscillation between similar-scoring lines (e.g. a
    rahao tuk that's hovering with a nearby verse).

    `prev_line` is NOT reset when a chunk falls below `score_threshold`; quiet
    interludes shouldn't lose the trajectory.
    """
    segments: list[Segment] = []
    cur: Segment | None = None
    prev_line: int | None = None

    for start, end, scores in chunks_with_scores:
        if not scores:
            continue
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_score = scores[best_idx]

        chosen: int | None = best_idx
        if (prev_line is not None and prev_line < len(scores)
                and best_score - scores[prev_line] < stay_margin):
            chosen = prev_line

        if chosen is None or scores[chosen] < score_threshold:
            if cur is not None:
                segments.append(cur)
                cur = None
            continue

        if cur is not None and cur.line_idx == chosen:
            cur.end = end
        else:
            if cur is not None:
                segments.append(cur)
            cur = Segment(start=start, end=end, line_idx=chosen)
        prev_line = chosen

    if cur is not None:
        segments.append(cur)
    return segments
