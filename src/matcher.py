"""Fuzzy match ASR chunks to shabad lines.

Whisper outputs Devanagari (Hindi script) for Punjabi audio; the BaniDB
corpus carries `transliteration_english` (latin). We romanize the ASR side
via unidecode, normalize both, then rapidfuzz `WRatio` against each line.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from rapidfuzz import fuzz
from unidecode import unidecode

RATIO_FUNCS = {
    "WRatio": fuzz.WRatio,
    "ratio": fuzz.ratio,
    "partial_ratio": fuzz.partial_ratio,
    "token_sort_ratio": fuzz.token_sort_ratio,
    "token_set_ratio": fuzz.token_set_ratio,
    "partial_token_set_ratio": fuzz.partial_token_set_ratio,
}


@dataclass
class LineMatch:
    line_idx: int | None  # None when no candidate clears the threshold
    score: float


_PANGTI_MARK_RE = re.compile(r"\|\|\s*\d*\s*\|*")
_PARENS_RE = re.compile(r"\([^)]*\)")
_NONWORD_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_RAHAAU_RE = re.compile(r"\brahaau\b")


def normalize(text: str) -> str:
    """Romanize, strip pangti markers / parenthetical hints, lowercase, collapse ws."""
    text = unidecode(text).lower()
    text = _PARENS_RE.sub("", text)
    text = _PANGTI_MARK_RE.sub(" ", text)
    text = _RAHAAU_RE.sub(" ", text)
    text = _NONWORD_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


class TfidfScorer:
    """Per-shabad TF-IDF cosine similarity. Down-weights tokens that appear in many lines.

    Built once per shabad from the corpus. Refrain tokens (e.g. `man bauraa re` in
    shabad 1341) appear in nearly every line, so their IDF approaches the floor and
    they contribute almost nothing to the score; distinctive tokens dominate.
    """

    def __init__(self, normalized_lines: list[str]):
        self.lines = normalized_lines
        n = max(1, len(normalized_lines))
        df: Counter[str] = Counter()
        for line in normalized_lines:
            for tok in set(line.split()):
                df[tok] += 1
        # Smoothed IDF (sklearn-style). Token in all N lines: idf=1. Token in 1 line:
        # idf = log((N+1)/2) + 1, much higher. Avoids zero weights / div-by-zero.
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self.line_vecs = [self._vec(line) for line in normalized_lines]

    def _vec(self, s: str) -> dict[str, float]:
        if not s:
            return {}
        tf = Counter(s.split())
        vec = {t: c * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {t: v / norm for t, v in vec.items()} if norm > 0 else {}

    def score_all(self, chunk_norm: str) -> list[float]:
        cv = self._vec(chunk_norm)
        if not cv:
            return [0.0] * len(self.line_vecs)
        return [
            100.0 * sum(cv[t] * lv.get(t, 0.0) for t in cv)
            for lv in self.line_vecs
        ]


def _score(chunk_norm: str, cand: str, ratio: str, blend: dict[str, float] | None) -> float:
    if blend:
        return sum(w * RATIO_FUNCS[name](chunk_norm, cand) for name, w in blend.items())
    return RATIO_FUNCS[ratio](chunk_norm, cand)


def score_chunk(
    chunk_text: str,
    lines: list[dict],
    *,
    ratio: str = "WRatio",
    blend: dict[str, float] | None = None,
    tfidf: TfidfScorer | None = None,
) -> list[float]:
    """Return a per-line score in the same order as `lines`. Empty lines score 0."""
    chunk_norm = normalize(chunk_text)
    if not chunk_norm:
        return [0.0] * len(lines)

    use_tfidf = blend is not None and "tfidf" in blend
    tfidf_scores = tfidf.score_all(chunk_norm) if (use_tfidf and tfidf is not None) else None

    out: list[float] = []
    for i, line in enumerate(lines):
        cand = normalize(line.get("transliteration_english", ""))
        if not cand:
            out.append(0.0)
            continue
        if blend:
            s = 0.0
            for name, w in blend.items():
                if name == "tfidf":
                    s += w * (tfidf_scores[i] if tfidf_scores else 0.0)
                else:
                    s += w * RATIO_FUNCS[name](chunk_norm, cand)
        else:
            s = RATIO_FUNCS[ratio](chunk_norm, cand)
        out.append(s)
    return out


def match_chunk(
    chunk_text: str,
    lines: list[dict],
    *,
    score_threshold: float = 55.0,
    margin_threshold: float = 0.0,
    ratio: str = "WRatio",
    blend: dict[str, float] | None = None,
    tfidf: TfidfScorer | None = None,
) -> LineMatch:
    """Return the best-matching `line_idx` for `chunk_text`, or None below threshold.

    `ratio` selects which rapidfuzz scorer to use (see `RATIO_FUNCS`). If `blend`
    is given (mapping scorer-name → weight), uses a weighted sum. `"tfidf"` is a
    valid name in `blend` and requires `tfidf` to be a precomputed TfidfScorer.
    `margin_threshold` is the minimum top1-top2 gap.
    """
    chunk_norm = normalize(chunk_text)
    if not chunk_norm:
        return LineMatch(None, 0.0)

    use_tfidf = blend is not None and "tfidf" in blend
    tfidf_scores: list[float] | None = (
        tfidf.score_all(chunk_norm) if (use_tfidf and tfidf is not None) else None
    )

    best_idx: int | None = None
    best_score: float = 0.0
    second_score: float = 0.0
    for i, line in enumerate(lines):
        cand = normalize(line.get("transliteration_english", ""))
        if not cand:
            continue
        if blend:
            score = 0.0
            for name, w in blend.items():
                if name == "tfidf":
                    score += w * (tfidf_scores[i] if tfidf_scores else 0.0)
                else:
                    score += w * RATIO_FUNCS[name](chunk_norm, cand)
        else:
            score = RATIO_FUNCS[ratio](chunk_norm, cand)

        if score > best_score:
            second_score = best_score
            best_score = score
            best_idx = int(line["line_idx"])
        elif score > second_score:
            second_score = score

    if best_score < score_threshold:
        return LineMatch(None, best_score)
    if best_score - second_score < margin_threshold:
        return LineMatch(None, best_score)
    return LineMatch(best_idx, best_score)
