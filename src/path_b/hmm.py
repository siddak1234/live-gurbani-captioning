"""Loop-aware HMM line tracker for Path B.

States: one "macro state" per shabad line. Each macro state contains a CTC
alignment lattice for that line's token sequence — within-line transitions
follow standard CTC rules (stay / advance / skip-blank). Across-line
transitions cost `switch_log_prob` per frame and connect the end of one line
to the start of any other line (or itself, for loop-back).

The forward algorithm runs causally over all frames. At each frame, the
posterior over lines is `logsumexp` over positions within each line. The
argmax line per frame is the prediction.

This formulation fixes Phase B2's length bias: a line can't "cherry-pick"
frames anymore — it must traverse its lattice in order, accumulating
emission costs, and switching to another line costs explicit probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

NEG_INF = -1e30


@dataclass
class LineSpec:
    line_idx: int
    tokens: list[int]   # MMS token IDs (no blanks)


class ShabadHmm:
    """Forward-algorithm decoder over a small set of "line" macro states."""

    def __init__(
        self,
        lines: list[LineSpec],
        blank_id: int,
        switch_log_prob: float = math.log(1e-4),
    ):
        self.lines = lines
        self.blank_id = blank_id
        self.switch_log_prob = switch_log_prob

        # Per-line augmented target arrays: [blank, t0, blank, t1, ..., t_{L-1}, blank]
        self.ext_seqs: list[np.ndarray] = []
        self.skip_masks: list[np.ndarray] = []
        for line in lines:
            ext = [blank_id]
            for tok in line.tokens:
                ext.append(tok)
                ext.append(blank_id)
            ext_arr = np.array(ext, dtype=np.int64)
            self.ext_seqs.append(ext_arr)

            # Skip-blank mask: position s can receive a "skip" from s-2 iff
            # ext[s] is non-blank AND ext[s] != ext[s-2] (CTC rule).
            S = len(ext_arr)
            mask = np.zeros(S, dtype=bool)
            if S >= 3:
                mask[2:] = (ext_arr[2:] != blank_id) & (ext_arr[2:] != ext_arr[:-2])
            self.skip_masks.append(mask)

    def forward(
        self,
        log_probs: np.ndarray,
        *,
        viterbi: bool = False,
    ) -> np.ndarray:
        """Run the forward algorithm. Returns marginal/best-path log-probs per line per frame.

        `log_probs` shape: (T, V).
        Output shape: (T, N) where N = number of lines.
        If `viterbi=True`, uses max instead of logsumexp throughout — gives best-path
        scores per line per frame instead of marginalized scores. Often more stable
        when line lengths differ a lot, since longer lines don't accumulate sum mass.
        """
        T, _V = log_probs.shape
        N = len(self.lines)
        if N == 0 or T == 0:
            return np.zeros((T, N), dtype=np.float64)

        # Choose combinator based on mode: forward uses logsumexp, Viterbi uses max.
        combine = (lambda a, b: np.maximum(a, b)) if viterbi else np.logaddexp
        line_marg_fn = (lambda x: float(x.max())) if viterbi else _line_marginal
        scalar_combine = max if viterbi else _logaddexp_scalar

        alpha_per_line: list[np.ndarray] = []
        for k in range(N):
            ext = self.ext_seqs[k]
            S_k = len(ext)
            a = np.full(S_k, NEG_INF, dtype=np.float64)
            a[0] = -math.log(N) + float(log_probs[0, self.blank_id])
            if S_k >= 2:
                a[1] = -math.log(N) + float(log_probs[0, ext[1]])
            alpha_per_line.append(a)

        marginals = np.full((T, N), NEG_INF, dtype=np.float64)
        for k in range(N):
            marginals[0, k] = line_marg_fn(alpha_per_line[k])

        for t in range(1, T):
            log_p_t = log_probs[t]

            end_masses = np.full(N, NEG_INF, dtype=np.float64)
            for k in range(N):
                a = alpha_per_line[k]
                S_k = len(a)
                last = float(a[S_k - 1])
                second_last = float(a[S_k - 2]) if S_k >= 2 else NEG_INF
                end_masses[k] = scalar_combine(last, second_last)

            new_alpha_per_line: list[np.ndarray] = []
            for k_target in range(N):
                a = alpha_per_line[k_target]
                ext = self.ext_seqs[k_target]
                S_k = len(a)

                # Disallow "stay" at the end position. Without this, alpha at
                # the final blank grows monotonically by absorbing blank-emitting
                # frames, and the HMM gets stuck on whichever line finished
                # first. Forcing exit via cross-line transitions only.
                stay = a.copy()
                if S_k >= 1:
                    stay[S_k - 1] = NEG_INF
                advance = np.empty_like(a)
                advance[0] = NEG_INF
                advance[1:] = a[:-1]
                cand = combine(stay, advance)

                if S_k >= 3:
                    skip = np.empty_like(a)
                    skip[:2] = NEG_INF
                    skip[2:] = a[:-2]
                    cand_with_skip = combine(cand, skip)
                    cand = np.where(self.skip_masks[k_target], cand_with_skip, cand)

                new_a = cand + log_p_t[ext]

                # Cross-line: incoming mass from *other* lines' ends.
                # Pick the best other line's end mass under the chosen combinator.
                best_other = NEG_INF
                for k_src in range(N):
                    if k_src == k_target:
                        continue
                    best_other = scalar_combine(best_other, float(end_masses[k_src]))
                if best_other > NEG_INF / 2:
                    cross = best_other + self.switch_log_prob
                    new_a[0] = scalar_combine(float(new_a[0]),
                                              cross + float(log_p_t[self.blank_id]))
                    if S_k >= 2:
                        new_a[1] = scalar_combine(float(new_a[1]),
                                                  cross + float(log_p_t[ext[1]]))

                new_alpha_per_line.append(new_a)

            alpha_per_line = new_alpha_per_line
            for k in range(N):
                marginals[t, k] = line_marg_fn(alpha_per_line[k])

        return marginals

    def decode(self, log_probs: np.ndarray, *, viterbi: bool = False) -> np.ndarray:
        """Per-frame line index prediction (argmax of forward marginals / Viterbi)."""
        marginals = self.forward(log_probs, viterbi=viterbi)
        return np.array([self.lines[k].line_idx for k in marginals.argmax(axis=1)],
                        dtype=np.int64)


def _line_marginal(alpha_k: np.ndarray) -> float:
    """Sum (in log space) all positions of a line's alpha vector."""
    m = float(alpha_k.max())
    if m <= NEG_INF / 2:
        return NEG_INF
    return m + float(np.log(np.exp(alpha_k - m).sum()))


def _logsumexp_arr(arr: np.ndarray) -> float:
    m = float(arr.max())
    if m <= NEG_INF / 2:
        return NEG_INF
    return m + float(np.log(np.exp(arr - m).sum()))


def _logsubexp(a: float, b: float) -> float:
    """log(exp(a) - exp(b)). Assumes a > b (otherwise returns NEG_INF)."""
    if a <= b:
        return NEG_INF
    return a + math.log1p(-math.exp(b - a))


def _logaddexp_scalar(a: float, b: float) -> float:
    if a == NEG_INF and b == NEG_INF:
        return NEG_INF
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))
