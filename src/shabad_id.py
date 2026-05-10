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

    if aggregate == "chunk_vote":
        # Each ASR chunk in the window votes for its top-scoring shabad.
        # Vote weight = top1_score (so high-confidence chunks count more).
        # Repetition becomes additional votes for the same correct shabad,
        # which is robust to shared "kaa thaan"-style hooks across shabads.
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
