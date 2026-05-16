"""Merge per-chunk matches into submission segments.

Initial implementation: contiguous same-`line_idx` matches collapse into one
segment; gaps and `None` matches break the run. Plenty of room to add
HMM-style smoothing later (Path B territory).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.matcher import normalize


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


_SIMRAN_PREFIXES = ("vaahigur", "wahegur", "vahigur")


def is_simran_dominant(text: str, *, min_tokens: int = 4, ratio: float = 0.65) -> bool:
    """Return True when a chunk is dominated by repeated Waheguru/simran.

    These chunks are devotional filler, not canonical pangti evidence. Forcing
    them into the nearest shabad line creates long false-positive spans in cold
    windows. The detector is intentionally generic: no shabad ID, case name, or
    benchmark timing enters the rule.
    """
    tokens = normalize(text).split()
    if len(tokens) < min_tokens:
        return False
    simran_tokens = sum(
        1
        for token in tokens
        if any(token.startswith(prefix) for prefix in _SIMRAN_PREFIXES)
    )
    return (simran_tokens / len(tokens)) >= ratio


def smooth_with_loop_align(
    chunks_with_scores: list[tuple[float, float, list[float], str]],
    *,
    stay_margin: float = 5.0,
    score_threshold: float = 0.0,
) -> list[Segment]:
    """Null-aware text-score aligner for Phase 2.9.

    First pass: preserve the known-good stay-bias line tracker, but add a null
    state for chunks dominated by repeated simran (e.g. "ਵਾਹਿਗੁਰੂ" loops).
    This handles sparse cold-window filler without adding benchmark-specific
    route tables or timing rules.
    """
    stripped: list[tuple[float, float, int | None]] = []
    prev_line: int | None = None

    for start, end, scores, text in chunks_with_scores:
        if is_simran_dominant(text):
            stripped.append((start, end, None))
            continue
        if not scores:
            stripped.append((start, end, None))
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
            stripped.append((start, end, None))
            continue

        stripped.append((start, end, chosen))
        prev_line = chosen

    return smooth(stripped)


def _transition_score(
    prev: int | None,
    cur: int | None,
    *,
    jump_penalty: float,
    backtrack_penalty: float,
    null_switch_penalty: float,
) -> float:
    """Return additive transition score for the Viterbi smoother.

    ``None`` is the null/no-line state. It is useful for absorbing low-confidence
    filler chunks without resetting the surrounding line trajectory.
    """
    if prev is None or cur is None:
        return 0.0 if prev is cur else -null_switch_penalty
    if prev == cur:
        return 0.0
    distance = abs(cur - prev)
    score = -jump_penalty * distance
    if cur < prev:
        score -= backtrack_penalty * distance
    return score


def smooth_with_viterbi(
    chunks_with_scores: list[tuple[float, float, list[float]]],
    *,
    jump_penalty: float = 4.0,
    backtrack_penalty: float = 8.0,
    score_threshold: float = 0.0,
    null_score: float | None = None,
    null_switch_penalty: float = 0.0,
) -> list[Segment]:
    """Sequence-level smoother over per-line chunk scores.

    ``smooth_with_stay_bias`` is greedy: a noisy chunk can jump to a wrong line
    if that line is locally best and the previous line is not close enough. This
    function solves the whole chunk sequence with Viterbi dynamic programming so
    local scores are balanced against line-continuity penalties.

    The optional null state (``null_score``) represents "no lyric line". It lets
    low-confidence chunks disappear from the emitted caption stream while still
    keeping the global path continuity for neighboring chunks. This is a generic
    alignment primitive, not a shabad-specific route table.
    """
    usable = [(start, end, scores) for start, end, scores in chunks_with_scores if scores]
    if not usable:
        return []

    n_lines = len(usable[0][2])
    if any(len(scores) != n_lines for _, _, scores in usable):
        raise ValueError("all score vectors must have the same length")

    states: list[int | None] = list(range(n_lines))
    if null_score is not None:
        states.append(None)

    def local_score(scores: list[float], state: int | None) -> float:
        if state is None:
            return float(null_score if null_score is not None else 0.0)
        if scores[state] < score_threshold:
            return -1_000_000.0
        return float(scores[state])

    # dp[t][j] = best total score through chunk t ending in states[j].
    dp: list[list[float]] = []
    backpointers: list[list[int]] = []

    first_scores = usable[0][2]
    dp.append([local_score(first_scores, state) for state in states])

    for _, _, scores in usable[1:]:
        prev_scores = dp[-1]
        cur_scores: list[float] = []
        cur_bp: list[int] = []
        for cur_i, cur_state in enumerate(states):
            local = local_score(scores, cur_state)
            best_total = -1_000_000_000.0
            best_prev_i = 0
            for prev_i, prev_state in enumerate(states):
                total = (
                    prev_scores[prev_i]
                    + _transition_score(
                        prev_state,
                        cur_state,
                        jump_penalty=jump_penalty,
                        backtrack_penalty=backtrack_penalty,
                        null_switch_penalty=null_switch_penalty,
                    )
                    + local
                )
                if total > best_total:
                    best_total = total
                    best_prev_i = prev_i
            cur_scores.append(best_total)
            cur_bp.append(best_prev_i)
        dp.append(cur_scores)
        backpointers.append(cur_bp)

    state_i = max(range(len(states)), key=lambda i: dp[-1][i])
    path_indices = [state_i]
    for t in range(len(backpointers) - 1, -1, -1):
        state_i = backpointers[t][state_i]
        path_indices.append(state_i)
    path_indices.reverse()
    chosen_states = [states[i] for i in path_indices]

    matches = [
        (start, end, state)
        for (start, end, _), state in zip(usable, chosen_states)
    ]
    return smooth(matches)
