"""Runtime ID-lock composition for Phase 2.7.

This module is Layer 2: it composes two ordinary ``engine.predict`` calls but
knows nothing about benchmark JSON files, output directories, or argparse.

Architecture:

1. A conservative pre-lock engine runs blind/live for the first lookback window
   and emits tentative captions.
2. Once that engine commits a shabad_id, a post-lock engine runs against only
   that shabad's line set.
3. Output merge policy decides how final captions handle the buffered pre-lock
   window: commit-cutover preserves tentative pre-lock captions before commit;
   retro-buffered lets the locked-shabad post engine revise from UEM start.

The default post-lock context is "buffered": the post-lock smoother may use the
pre-lock transcript as causal state/context, but emitted segments are still cut
at commit time. This mirrors a real streaming system that buffers transcript
state during the ID window and then switches the displayed engine after lock.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, replace
from typing import Literal

from src.engine import EngineConfig, PredictionResult, Segment, predict


PostContextMode = Literal["buffered", "strict-live"]
MergePolicy = Literal["commit-cutover", "retro-buffered"]


@dataclass
class IdLockPredictionResult:
    """Runtime ID-lock output plus the child engine diagnostics."""

    prediction: PredictionResult
    pre_lock: PredictionResult
    post_lock: PredictionResult
    commit_time: float
    post_context: PostContextMode
    merge_policy: MergePolicy


def _clip_segment(
    segment: Segment,
    *,
    start: float | None = None,
    end: float | None = None,
) -> Segment | None:
    """Return a clipped copy of ``segment`` or ``None`` if it becomes empty."""
    clipped = replace(segment)
    if start is not None:
        clipped.start = max(float(clipped.start), float(start))
    if end is not None:
        clipped.end = min(float(clipped.end), float(end))
    if float(clipped.end) <= float(clipped.start):
        return None
    return clipped


def merge_segments_at_commit(
    pre_segments: list[Segment],
    post_segments: list[Segment],
    *,
    commit_time: float,
) -> list[Segment]:
    """Keep pre-lock segments before commit and post-lock segments after it."""
    merged: list[Segment] = []
    for segment in pre_segments:
        clipped = _clip_segment(segment, end=commit_time)
        if clipped is not None:
            merged.append(clipped)
    for segment in post_segments:
        clipped = _clip_segment(segment, start=commit_time)
        if clipped is not None:
            merged.append(clipped)
    merged.sort(key=lambda s: (float(s.start), float(s.end)))
    return merged


def retro_buffer_segments(
    post_segments: list[Segment],
    *,
    start_time: float,
) -> list[Segment]:
    """Use locked-shabad post segments from ``start_time`` onward.

    This models a live UI where pre-lock captions are tentative. Once shabad ID
    is committed, the system can revise the buffered window with the final
    locked-shabad alignment. It is state/time based, not benchmark-shabad based.
    """
    out: list[Segment] = []
    for segment in post_segments:
        clipped = _clip_segment(segment, start=start_time)
        if clipped is not None:
            out.append(clipped)
    out.sort(key=lambda s: (float(s.start), float(s.end)))
    return out


def predict_idlocked(
    audio: pathlib.Path,
    corpora: dict[int, list[dict]],
    *,
    uem_start: float = 0.0,
    pre_config: EngineConfig | None = None,
    post_config: EngineConfig | None = None,
    post_context: PostContextMode = "buffered",
    merge_policy: MergePolicy = "commit-cutover",
) -> IdLockPredictionResult:
    """Run a state/time-based two-engine ID-lock prediction.

    Args:
        audio: path to 16 kHz mono WAV.
        corpora: dict shabad_id -> canonical line records.
        uem_start: start time for the active UEM region.
        pre_config: config for the blind ID/tentative-caption engine.
        post_config: config for the locked-shabad alignment engine.
        post_context: ``"buffered"`` runs post-lock alignment over the full
            buffered transcript and emits only after commit. ``"strict-live"``
            makes the post engine itself ignore pre-commit chunks.
        merge_policy: ``"commit-cutover"`` preserves the pre-lock tentative
            segments before commit and post-lock segments after commit.
            ``"retro-buffered"`` uses the locked-shabad post result from
            ``uem_start`` onward, treating pre-lock output as revisable.

    Returns:
        ``IdLockPredictionResult`` with merged prediction and child diagnostics.
    """
    if post_context not in ("buffered", "strict-live"):
        raise ValueError(f"unknown post_context: {post_context}")
    if merge_policy not in ("commit-cutover", "retro-buffered"):
        raise ValueError(f"unknown merge_policy: {merge_policy}")

    pre_cfg = replace(pre_config or EngineConfig(), live=True, tentative_emit=True)
    pre_result = predict(
        audio,
        corpora,
        shabad_id=None,
        uem_start=uem_start,
        config=pre_cfg,
    )

    commit_time = float(uem_start) + float(pre_cfg.blind_lookback)

    post_base = post_config or EngineConfig()
    post_cfg = replace(
        post_base,
        live=(post_context == "strict-live"),
        tentative_emit=False,
        blind_lookback=pre_cfg.blind_lookback,
    )
    post_result = predict(
        audio,
        corpora,
        shabad_id=pre_result.shabad_id,
        uem_start=uem_start,
        config=post_cfg,
    )

    if merge_policy == "commit-cutover":
        merged_segments = merge_segments_at_commit(
            pre_result.segments,
            post_result.segments,
            commit_time=commit_time,
        )
    else:
        merged_segments = retro_buffer_segments(
            post_result.segments,
            start_time=float(uem_start),
        )
    merged = PredictionResult(
        segments=merged_segments,
        shabad_id=pre_result.shabad_id,
        n_chunks=pre_result.n_chunks + post_result.n_chunks,
        blind_id_score=pre_result.blind_id_score,
        blind_runner_up_score=pre_result.blind_runner_up_score,
    )
    return IdLockPredictionResult(
        prediction=merged,
        pre_lock=pre_result,
        post_lock=post_result,
        commit_time=commit_time,
        post_context=post_context,
        merge_policy=merge_policy,
    )
