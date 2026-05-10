"""Character-level Gurmukhi tokenizer for the MMS-pan vocabulary.

MMS's Punjabi adapter outputs Gurmukhi characters directly (~95 token vocab),
so "G2P" collapses to a per-character vocab lookup. Characters not in the
vocab (e.g. unusual punctuation, ZWJ marks) are dropped.

Whitespace handling: MMS treats `|` as the inter-word separator in its output.
We mirror that on input — collapse runs of whitespace to single `|`.
"""

from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")
_WORD_SEP = "|"


def normalize_for_mms(text: str) -> str:
    """Light normalization: collapse whitespace to MMS's `|` separator."""
    return _WS_RE.sub(_WORD_SEP, text).strip(_WORD_SEP)


def tokenize_line(text: str, vocab: dict[str, int]) -> list[int]:
    """Map a Gurmukhi line to a list of MMS vocab token IDs.

    Characters not in `vocab` are silently dropped. The returned list is the
    raw target token sequence (no CTC blanks inserted — the forward algorithm
    handles those).
    """
    normalized = normalize_for_mms(text)
    out: list[int] = []
    for char in normalized:
        if char in vocab:
            out.append(vocab[char])
    return out


def diagnose_coverage(text: str, vocab: dict[str, int]) -> tuple[int, int, set[str]]:
    """Return (mapped, total, missing) for a piece of text against the vocab.

    Useful for spot-checking that our corpus lines tokenize well against the MMS vocab.
    """
    missing: set[str] = set()
    normalized = normalize_for_mms(text)
    mapped = 0
    for char in normalized:
        if char in vocab:
            mapped += 1
        else:
            missing.add(char)
    return mapped, len(normalized), missing
