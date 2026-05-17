"""Blind shabad identification.

Two strategies:
  - `aggregate="max" / "topk:N"`: rapidfuzz scoring, max or sum-of-top-N over
    each candidate shabad's lines. Vulnerable to length effects and shared
    common-word noise.
  - `aggregate="tfidf"`: TF-IDF cosine over shabads-as-documents. Each shabad
    is one document = concatenation of all its lines; IDF is computed across
    the candidate set, so distinctive tokens dominate and refrain/common
    tokens carry near-zero weight. The right tool when the candidate set
    contains shabads with overlapping superficial vocabulary.
  - `aggregate="fusion:<spec>"`: experimental Phase 2.13 candidate-evidence
    fusion. Example: `fusion:tfidf_60+0.5*chunk_vote_90`.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from src.matcher import normalize, score_chunk


@dataclass
class ShabadIdResult:
    shabad_id: int
    score: float
    runner_up_id: int | None
    runner_up_score: float


def buffer_text(asr_chunks, *, start_t: float, lookback_seconds: float) -> str:
    """Concatenate ASR chunks whose time range overlaps `[start_t, start_t+lookback_seconds]`."""
    end_t = start_t + lookback_seconds
    parts: list[str] = []
    for c in asr_chunks:
        if c.start >= end_t:
            break
        if c.end <= start_t:
            continue
        parts.append(c.text)
    return " ".join(parts)


def per_chunk_global_match(
    chunks,
    corpora: dict[int, list[dict]],
    *,
    ratio: str = "WRatio",
    blend: dict[str, float] | None = None,
) -> list[tuple[int | None, int | None, float]]:
    """For each chunk, find the global best `(shabad_id, line_idx, score)` across all corpora.

    Used during the live shabad-ID buffer: predictions in this window can't depend
    on a committed shabad yet, so each chunk independently picks the best
    (shabad, line) pair globally. After the buffer, the runner reverts to the
    constrained-shabad matcher.
    """
    out: list[tuple[int | None, int | None, float]] = []
    for c in chunks:
        best = (None, None, 0.0)
        for sid, lines in corpora.items():
            scores = score_chunk(c.text, lines, ratio=ratio, blend=blend)
            for lidx, s in enumerate(scores):
                if s > best[2]:
                    best = (sid, lidx, s)
        out.append(best)
    return out


class ShabadDocTfidf:
    """TF-IDF over shabads-as-documents. One vector per shabad."""

    def __init__(self, corpora: dict[int, list[dict]]):
        documents: dict[int, str] = {}
        for sid, lines in corpora.items():
            joined = " ".join(
                normalize(l.get("transliteration_english", "")) for l in lines
            )
            documents[sid] = joined

        n = max(1, len(documents))
        df: Counter[str] = Counter()
        for doc in documents.values():
            for tok in set(doc.split()):
                df[tok] += 1
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self.shabad_vecs: dict[int, dict[str, float]] = {
            sid: self._vec(doc) for sid, doc in documents.items()
        }

    def _vec(self, s: str) -> dict[str, float]:
        if not s:
            return {}
        tf = Counter(s.split())
        v = {t: c * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values()))
        return {t: x / norm for t, x in v.items()} if norm > 0 else {}

    def score(self, query: str) -> dict[int, float]:
        qv = self._vec(normalize(query))
        if not qv:
            return {sid: 0.0 for sid in self.shabad_vecs}
        return {
            sid: 100.0 * sum(qv[t] * sv.get(t, 0.0) for t in qv)
            for sid, sv in self.shabad_vecs.items()
        }


def _parse_fusion_term(raw: str) -> tuple[float, str, float, float]:
    """Parse ``[weight*]feature_window`` for experimental fusion aggregates."""
    term = raw.strip()
    if "*" in term:
        weight_s, feature = term.split("*", 1)
        weight = float(weight_s.strip())
    else:
        weight = 1.0
        feature = term
    feature = feature.strip()
    if feature.startswith("tail_chunk_vote_"):
        parts = feature.removeprefix("tail_chunk_vote_").split("_")
        if len(parts) != 2:
            raise ValueError(
                f"bad tail fusion term {raw!r}; expected tail_chunk_vote_<tail>_<window>"
            )
        tail = float(parts[0])
        window = float(parts[1])
        if tail <= 0 or window <= 0 or tail > window:
            raise ValueError("tail fusion windows must satisfy 0 < tail <= window")
        return weight, "chunk_vote", tail, window - tail
    if "_" not in feature:
        raise ValueError(f"bad fusion term {raw!r}; expected feature_window")
    name, window_s = feature.rsplit("_", 1)
    aliases = {
        "topk3": "topk:3",
        "chunkvote": "chunk_vote",
    }
    aggregate = aliases.get(name, name)
    if aggregate not in {"chunk_vote", "tfidf", "topk:3"}:
        raise ValueError(f"bad fusion feature {name!r}; expected chunk_vote, tfidf, or topk3")
    window = float(window_s)
    if window <= 0:
        raise ValueError("fusion windows must be positive")
    return weight, aggregate, window, 0.0


def parse_fusion_spec(spec: str) -> list[tuple[float, str, float, float]]:
    """Parse experimental fusion spec into ``(weight, aggregate, window, offset)`` terms."""
    terms = [_parse_fusion_term(part) for part in spec.split("+") if part.strip()]
    if not terms:
        raise ValueError("fusion spec must contain at least one term")
    return terms


def _chunk_vote_score_map(
    asr_chunks,
    corpora: dict[int, list[dict]],
    *,
    start_t: float,
    lookback_seconds: float,
    ratio: str,
    blend: dict[str, float] | None,
) -> dict[int, float]:
    end_t = start_t + lookback_seconds
    weights: dict[int, float] = {sid: 0.0 for sid in corpora}
    for c in asr_chunks:
        if c.start >= end_t:
            break
        if c.end <= start_t:
            continue
        best_sid: int | None = None
        best_s = 0.0
        for sid, lines in corpora.items():
            scores = score_chunk(c.text, lines, ratio=ratio, blend=blend)
            s = max(scores) if scores else 0.0
            if s > best_s:
                best_s = s
                best_sid = sid
        if best_sid is not None:
            weights[best_sid] += best_s
    return weights


def _score_map_for_aggregate(
    asr_chunks,
    corpora: dict[int, list[dict]],
    *,
    start_t: float,
    lookback_seconds: float,
    ratio: str,
    blend: dict[str, float] | None,
    aggregate: str,
    tfidf_scorer: ShabadDocTfidf | None = None,
) -> dict[int, float]:
    if aggregate == "chunk_vote":
        return _chunk_vote_score_map(
            asr_chunks,
            corpora,
            start_t=start_t,
            lookback_seconds=lookback_seconds,
            ratio=ratio,
            blend=blend,
        )
    buf = buffer_text(asr_chunks, start_t=start_t, lookback_seconds=lookback_seconds)
    if aggregate == "tfidf":
        scorer = tfidf_scorer or ShabadDocTfidf(corpora)
        return scorer.score(buf)
    if aggregate.startswith("topk:"):
        k = int(aggregate.split(":")[1])
        out: dict[int, float] = {}
        for sid, lines in corpora.items():
            scores = score_chunk(buf, lines, ratio=ratio, blend=blend)
            out[sid] = sum(sorted(scores, reverse=True)[:k]) if scores else 0.0
        return out
    out = {}
    for sid, lines in corpora.items():
        scores = score_chunk(buf, lines, ratio=ratio, blend=blend)
        out[sid] = max(scores) if scores else 0.0
    return out


def identify_shabad_fusion(
    asr_chunks,
    corpora: dict[int, list[dict]],
    *,
    spec: str,
    start_t: float = 0.0,
    ratio: str = "WRatio",
    blend: dict[str, float] | None = None,
) -> ShabadIdResult:
    """Pick a shabad with sparse normalized evidence fusion.

    This is an opt-in experimental lock policy for Phase 2.13. Each term is
    normalized by its per-case maximum before weighting so a high-magnitude
    feature family (for example chunk-vote sums) cannot dominate by scale alone.
    """
    terms = parse_fusion_spec(spec)
    tfidf_scorer = ShabadDocTfidf(corpora) if any(agg == "tfidf" for _, agg, _, _ in terms) else None
    total: dict[int, float] = {sid: 0.0 for sid in corpora}
    for weight, aggregate, window, offset in terms:
        raw = _score_map_for_aggregate(
            asr_chunks,
            corpora,
            start_t=start_t + offset,
            lookback_seconds=window,
            ratio=ratio,
            blend=blend,
            aggregate=aggregate,
            tfidf_scorer=tfidf_scorer,
        )
        max_score = max(raw.values()) if raw else 0.0
        for sid in corpora:
            normalized = raw.get(sid, 0.0) / max_score if max_score > 0 else 0.0
            total[sid] += weight * normalized
    ranked = sorted(total.items(), key=lambda x: (-x[1], x[0]))
    top = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else (None, 0.0)
    return ShabadIdResult(top[0], top[1], runner_up[0], runner_up[1])


def identify_shabad(
    asr_chunks,
    corpora: dict[int, list[dict]],
    *,
    start_t: float = 0.0,
    lookback_seconds: float = 30.0,
    ratio: str = "WRatio",
    blend: dict[str, float] | None = None,
    aggregate: str = "max",  # "max", "topk:N", or "tfidf"
    tfidf_scorer: ShabadDocTfidf | None = None,
) -> ShabadIdResult:
    """Pick the best-matching shabad_id from `corpora` for a window of ASR audio.

    `aggregate` controls how per-line scores collapse to a per-shabad score:
      - "max": single best line wins (sensitive, high variance)
      - "topk:N": sum of top-N line scores (rewards multi-line matches; kirtan
        typically sings the rahao + a verse + the rahao again within 30-60s, so
        a long lookback should produce N matches against the right shabad)
    """
    buf = buffer_text(asr_chunks, start_t=start_t, lookback_seconds=lookback_seconds)
    if not buf:
        first = sorted(corpora)[0]
        return ShabadIdResult(first, 0.0, None, 0.0)

    if aggregate.startswith("fusion:"):
        return identify_shabad_fusion(
            asr_chunks,
            corpora,
            spec=aggregate.split(":", 1)[1],
            start_t=start_t,
            ratio=ratio,
            blend=blend,
        )

    if aggregate == "chunk_vote":
        # Each ASR chunk in the window votes for its top-scoring shabad.
        # Vote weight = top1_score (so high-confidence chunks count more).
        # Repetition becomes additional votes for the same correct shabad,
        # which is robust to shared "kaa thaan"-style hooks across shabads.
        weights = _chunk_vote_score_map(
            asr_chunks,
            corpora,
            start_t=start_t,
            lookback_seconds=lookback_seconds,
            ratio=ratio,
            blend=blend,
        )
        ranked = sorted(weights.items(), key=lambda x: -x[1])
    elif aggregate == "tfidf":
        if tfidf_scorer is None:
            tfidf_scorer = ShabadDocTfidf(corpora)
        score_map = tfidf_scorer.score(buf)
        ranked = sorted(score_map.items(), key=lambda x: -x[1])
    else:
        if aggregate.startswith("topk:"):
            k = int(aggregate.split(":")[1])
            agg_fn = lambda scores: sum(sorted(scores, reverse=True)[:k])
        else:
            agg_fn = max
        ranked = []
        for sid, lines in corpora.items():
            scores = score_chunk(buf, lines, ratio=ratio, blend=blend)
            ranked.append((sid, agg_fn(scores) if scores else 0.0))
        ranked.sort(key=lambda x: -x[1])

    top = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else (None, 0.0)
    return ShabadIdResult(top[0], top[1], runner_up[0], runner_up[1])
