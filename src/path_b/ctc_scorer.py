"""CTC forward algorithm: log P(audio_window | target_token_sequence).

This is the math centerpiece of Path B. Given per-frame log-probabilities from
a CTC acoustic model and a target token sequence (no blanks), compute the
total log-probability of all alignment paths that decode to that target.

Standard textbook CTC forward DP in log-space. O(T * S) where S = 2L+1 and L
is the target length. Augmented target is [blank, t_0, blank, t_1, ..., t_{L-1}, blank].

Used per-window-per-line to score which line of the shabad is most likely
acoustically given the audio.
"""

from __future__ import annotations

import math

import numpy as np

NEG_INF = -1e30


def _logsumexp(*xs: float) -> float:
    m = max(xs)
    if m == NEG_INF:
        return NEG_INF
    total = 0.0
    for x in xs:
        total += math.exp(x - m)
    return m + math.log(total)


def ctc_log_prob(
    log_probs: np.ndarray,
    target_tokens: list[int],
    blank_id: int,
) -> float:
    """Compute log P(audio | target) under CTC.

    `log_probs` is `(T, V)` per-frame log-probabilities (already log-softmaxed).
    `target_tokens` is the line's character-token IDs (no blanks).
    Returns log P; -inf if no valid alignment exists.

    Handles the CTC rule that two consecutive identical target tokens
    require a blank between them — the skip-blank transition is disallowed
    when ext[s] == ext[s-2].
    """
    T, _V = log_probs.shape
    L = len(target_tokens)
    if L == 0:
        # Empty target: only the all-blank path counts.
        return float(log_probs[:, blank_id].sum())

    # Augmented target with blanks: [_, t0, _, t1, _, ..., t_{L-1}, _]
    ext = [blank_id]
    for tok in target_tokens:
        ext.append(tok)
        ext.append(blank_id)
    S = 2 * L + 1

    # alpha[s] = log prob of all paths through frames so far ending at ext position s
    alpha = [NEG_INF] * S
    alpha[0] = float(log_probs[0, blank_id])
    if S >= 2:
        alpha[1] = float(log_probs[0, ext[1]])

    for t in range(1, T):
        new_alpha = [NEG_INF] * S
        for s in range(S):
            # Three possible incoming transitions:
            #   stay at s (allowed always)
            #   advance from s-1 (allowed if s>=1)
            #   skip blank from s-2 (allowed if s>=2 and ext[s] != blank and ext[s] != ext[s-2])
            cand = alpha[s]
            if s >= 1:
                cand = _logsumexp(cand, alpha[s - 1])
            if s >= 2 and ext[s] != blank_id and ext[s] != ext[s - 2]:
                cand = _logsumexp(cand, alpha[s - 2])
            new_alpha[s] = cand + float(log_probs[t, ext[s]])
        alpha = new_alpha

    return _logsumexp(alpha[S - 1], alpha[S - 2])


def ctc_log_prob_normalized(
    log_probs: np.ndarray,
    target_tokens: list[int],
    blank_id: int,
) -> float:
    """Length-normalized variant: log P divided by audio-window frame count.

    Useful for comparing scores across lines because the raw log P scales linearly
    with the number of frames in the window. Same-window line comparisons can use
    the raw `ctc_log_prob`; cross-window or per-second comparisons should use this.
    """
    T = log_probs.shape[0]
    if T == 0:
        return NEG_INF
    return ctc_log_prob(log_probs, target_tokens, blank_id) / T
