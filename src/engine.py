"""Path A inference engine — pure callable library.

Wraps the ASR + matcher + shabad-ID + smoother pipeline behind a single
``predict()`` function. Knows nothing about benchmark GT JSON, submission JSON
files, argparse, or output directories — pure inputs to typed outputs.

The runner ``scripts/run_path_a.py`` is the thin I/O shim around this.

Decoupling rationale: the iOS/on-device deployment path needs the engine as a
callable library (Core ML / ONNX export later), separate from the benchmark
runner.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

from src.asr import transcribe
from src.matcher import match_chunk, score_chunk, normalize, TfidfScorer
from src.shabad_id import identify_shabad, per_chunk_global_match
from src.smoother import (
    smooth,
    smooth_with_loop_align,
    smooth_with_loop_align_confirmed,
    smooth_with_stay_bias,
    smooth_with_viterbi,
)


@dataclass
class EngineConfig:
    """Configuration knobs for ``predict()``. Defaults reproduce Path A v3.2."""

    # ASR
    backend: str = "faster_whisper"
    model_size: str = "medium"
    adapter_dir: str | None = None
    asr_cache_dir: pathlib.Path | None = None
    word_timestamps: bool = False
    no_speech_threshold: float | None = None
    vad_filter: bool = False

    # Matching
    ratio: str = "WRatio"
    blend: dict[str, float] | None = None
    score_threshold: float = 55.0
    margin_threshold: float = 0.0
    stay_bias: float = 0.0
    smoother: str = "auto"
    viterbi_jump_penalty: float = 4.0
    viterbi_backtrack_penalty: float = 8.0
    viterbi_null_score: float | None = None
    viterbi_null_switch_penalty: float = 0.0

    # Blind shabad ID
    blind_lookback: float = 30.0
    blind_aggregate: str = "max"
    blind_ratio: str = "WRatio"
    blind_blend: dict[str, float] | None = None

    # Live mode
    live: bool = False
    tentative_emit: bool = False


@dataclass
class Segment:
    """Output segment — a contiguous time range mapped to one canonical line."""
    start: float
    end: float
    line_idx: int
    shabad_id: int
    verse_id: str
    banidb_gurmukhi: str


@dataclass
class PredictionResult:
    """Engine output: segments plus metadata about how the prediction was made."""
    segments: list[Segment]
    shabad_id: int                              # decided shabad (==input if oracle)
    n_chunks: int = 0                           # total ASR chunks transcribed
    blind_id_score: float | None = None         # top-1 score from blind ID (None if oracle)
    blind_runner_up_score: float | None = None  # top-2 score from blind ID


def predict(
    audio: pathlib.Path,
    corpora: dict[int, list[dict]],
    *,
    shabad_id: int | None = None,
    uem_start: float = 0.0,
    config: EngineConfig | None = None,
) -> PredictionResult:
    """Run the Path A pipeline against a single audio file.

    Args:
        audio: path to 16 kHz mono WAV.
        corpora: dict shabad_id → list of line dicts (must include line_idx,
            verse_id, banidb_gurmukhi; transliteration_english is required only
            if ``config.blend`` contains "tfidf").
        shabad_id: oracle shabad_id if known; None to run blind shabad ID.
        uem_start: benchmark UEM start time (s). Defaults to 0 for arbitrary
            real-time audio.
        config: pipeline knobs; defaults reproduce v3.2 Path A canonical.

    Returns:
        ``PredictionResult`` with segments + metadata. Caller is responsible
        for serializing to whatever output format they need (e.g. submission
        JSON, iOS CoreData, gRPC payload).
    """
    cfg = config or EngineConfig()

    chunks = transcribe(
        audio,
        backend=cfg.backend,
        model_size=cfg.model_size,
        cache_dir=cfg.asr_cache_dir,
        adapter_dir=cfg.adapter_dir,
        word_timestamps=cfg.word_timestamps,
        no_speech_threshold=cfg.no_speech_threshold,
        vad_filter=cfg.vad_filter,
    )
    n_chunks_total = len(chunks)

    commit_time = uem_start + cfg.blind_lookback

    blind = shabad_id is None
    blind_id_score = None
    blind_runner_up_score = None
    if blind:
        sid_result = identify_shabad(
            chunks, corpora,
            start_t=uem_start, lookback_seconds=cfg.blind_lookback,
            ratio=cfg.blind_ratio, blend=cfg.blind_blend, aggregate=cfg.blind_aggregate,
        )
        shabad_id = sid_result.shabad_id
        blind_id_score = sid_result.score
        blind_runner_up_score = sid_result.runner_up_score

    if shabad_id not in corpora:
        raise ValueError(f"predicted shabad {shabad_id} not in corpora")
    lines = corpora[shabad_id]

    tfidf = None
    if cfg.blend and "tfidf" in cfg.blend:
        tfidf = TfidfScorer([normalize(l.get("transliteration_english", "")) for l in lines])

    pre_commit_segments: list[Segment] = []
    if cfg.live:
        # Causal: matcher only sees chunks ≥ commit time. Pre-commit chunks
        # emit per-chunk global-best (shabad, line) only when tentative_emit.
        if cfg.tentative_emit and blind:
            pre_chunks = [c for c in chunks
                          if c.start >= uem_start and c.start < commit_time]
            matches = per_chunk_global_match(pre_chunks, corpora, ratio=cfg.ratio, blend=cfg.blend)
            cur_sid: int | None = None
            cur_lidx: int | None = None
            cur_start: float = 0.0
            cur_end: float = 0.0
            for c, (sid, lidx, _) in zip(pre_chunks, matches):
                if sid is None or lidx is None:
                    continue
                if (sid, lidx) == (cur_sid, cur_lidx):
                    cur_end = c.end
                else:
                    if cur_sid is not None and cur_lidx is not None:
                        ln = next(l for l in corpora[cur_sid] if l["line_idx"] == cur_lidx)
                        pre_commit_segments.append(Segment(
                            start=cur_start, end=cur_end,
                            line_idx=cur_lidx, shabad_id=cur_sid,
                            verse_id=ln["verse_id"],
                            banidb_gurmukhi=ln["banidb_gurmukhi"],
                        ))
                    cur_sid, cur_lidx = sid, lidx
                    cur_start, cur_end = c.start, c.end
            if cur_sid is not None and cur_lidx is not None:
                ln = next(l for l in corpora[cur_sid] if l["line_idx"] == cur_lidx)
                pre_commit_segments.append(Segment(
                    start=cur_start, end=cur_end,
                    line_idx=cur_lidx, shabad_id=cur_sid,
                    verse_id=ln["verse_id"],
                    banidb_gurmukhi=ln["banidb_gurmukhi"],
                ))
        chunks = [c for c in chunks if c.start >= commit_time]

    smoother_name = cfg.smoother
    if smoother_name == "auto":
        smoother_name = "stay_bias" if cfg.stay_bias > 0.0 else "basic"

    if smoother_name not in ("basic", "stay_bias", "viterbi", "loop_align", "loop_align_confirmed"):
        raise ValueError(f"unknown smoother: {cfg.smoother}")

    if smoother_name in ("stay_bias", "viterbi", "loop_align", "loop_align_confirmed"):
        scored = [
            (c.start, c.end,
             score_chunk(c.text, lines, ratio=cfg.ratio, blend=cfg.blend, tfidf=tfidf))
            for c in chunks
        ]
        if smoother_name == "stay_bias":
            smooth_segments = smooth_with_stay_bias(
                scored, stay_margin=cfg.stay_bias, score_threshold=cfg.score_threshold,
            )
        elif smoother_name == "viterbi":
            smooth_segments = smooth_with_viterbi(
                scored,
                jump_penalty=cfg.viterbi_jump_penalty,
                backtrack_penalty=cfg.viterbi_backtrack_penalty,
                score_threshold=cfg.score_threshold,
                null_score=cfg.viterbi_null_score,
                null_switch_penalty=cfg.viterbi_null_switch_penalty,
            )
        else:
            scored_with_text = [
                (start, end, scores, c.text)
                for (start, end, scores), c in zip(scored, chunks)
            ]
            if smoother_name == "loop_align":
                smooth_segments = smooth_with_loop_align(
                    scored_with_text,
                    stay_margin=cfg.stay_bias,
                    score_threshold=cfg.score_threshold,
                )
            else:
                smooth_segments = smooth_with_loop_align_confirmed(
                    scored_with_text,
                    stay_margin=cfg.stay_bias,
                    score_threshold=cfg.score_threshold,
                )
    else:
        raw = [
            (c.start, c.end,
             match_chunk(c.text, lines,
                         score_threshold=cfg.score_threshold,
                         margin_threshold=cfg.margin_threshold,
                         ratio=cfg.ratio,
                         blend=cfg.blend,
                         tfidf=tfidf).line_idx)
            for c in chunks
        ]
        smooth_segments = smooth(raw)

    by_idx = {l["line_idx"]: l for l in lines}
    main_segments = [
        Segment(
            start=s.start, end=s.end, line_idx=s.line_idx, shabad_id=shabad_id,
            verse_id=by_idx[s.line_idx]["verse_id"],
            banidb_gurmukhi=by_idx[s.line_idx]["banidb_gurmukhi"],
        )
        for s in smooth_segments
    ]

    return PredictionResult(
        segments=pre_commit_segments + main_segments,
        shabad_id=shabad_id,
        n_chunks=n_chunks_total,
        blind_id_score=blind_id_score,
        blind_runner_up_score=blind_runner_up_score,
    )
