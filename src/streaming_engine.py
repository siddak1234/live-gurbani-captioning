"""Streaming variant of ``src.engine.predict()`` — iOS-shaped contract.

The contract
------------

::

    eng = StreamingEngine(corpora, config)
    eng.reset(shabad_id=None)                       # None = blind mode
    for frame in audio_stream():                    # iOS: AVAudioEngine tap
        new_segments = eng.process_pcm(frame, sr)   # incremental output
    final = eng.get_segments()

Why this exists
---------------

The benchmark and ``scripts/run_path_a.py`` run in *batch* mode — full audio
in, full segments out. iOS hands us audio in small frames over time and
expects captions to appear incrementally. This module exposes the same
inference logic behind that streaming contract, so the iOS Swift app can
mirror this API directly when it lands in M5.

Current implementation strategy
-------------------------------

Naive: every ``process_pcm`` snapshot-writes the ring buffer to a temp WAV
and re-runs ``engine.predict()``. This is *correct* but inefficient — Whisper
re-transcribes audio it has already seen. The naive version validates the
streaming-contract architecture; M5 (iOS Core ML) replaces the temp-file +
re-transcribe with a stateful Core ML encoder that consumes PCM in-memory.

Tuning
------

``process_interval_s`` (default 5 s) throttles how often the underlying
``predict()`` actually fires. Smaller = lower-latency captions but more
redundant work. iOS will adapt this based on word-boundary detection
from WhisperKit's streaming decoder.
"""

from __future__ import annotations

import pathlib
import tempfile
from dataclasses import dataclass, field

import numpy as np

from src.audio_buffer import AudioBuffer
from src.engine import EngineConfig, Segment, predict


@dataclass
class StreamingState:
    """Snapshot of the engine's internal state — useful for debugging / iOS."""
    committed_shabad_id: int | None = None
    audio_seconds: float = 0.0
    n_segments: int = 0
    last_processed_at_s: float = 0.0


class StreamingEngine:
    """iOS-shaped streaming wrapper around ``engine.predict()``."""

    def __init__(
        self,
        corpora: dict[int, list[dict]],
        config: EngineConfig | None = None,
        *,
        buffer_capacity_s: float = 120.0,
        process_interval_s: float = 5.0,
    ):
        self.corpora = corpora
        self.config = config or EngineConfig()
        self.buffer = AudioBuffer(capacity_s=buffer_capacity_s)
        self.process_interval_s = float(process_interval_s)

        self._committed_shabad_id: int | None = None
        self._segments: list[Segment] = []
        self._last_processed_at_s: float = 0.0
        # Optional override for blind mode — if user pre-commits a shabad
        # via reset(shabad_id=X), we stay in oracle mode permanently.
        self._oracle_locked: bool = False

    # -- session control ------------------------------------------------------

    def reset(self, shabad_id: int | None = None) -> None:
        """Start a new session.

        ``shabad_id`` known up-front → oracle mode (engine never runs blind ID).
        ``shabad_id`` None → blind mode (engine identifies shabad from audio).
        """
        self.buffer.reset()
        self._segments = []
        self._last_processed_at_s = 0.0
        self._committed_shabad_id = shabad_id
        self._oracle_locked = shabad_id is not None

    # -- streaming ingestion --------------------------------------------------

    def process_pcm(self, audio: np.ndarray, sr: int = 16000) -> list[Segment]:
        """Append new audio. Return segments emitted since the last call.

        The full segment list is always available via :meth:`get_segments`;
        this method returns only the *new* segments so callers can append
        them to a UI without recomputing diffs.
        """
        if audio is None or len(audio) == 0:
            return []
        self.buffer.append_pcm(audio, sr=sr)

        if self.buffer.current_time - self._last_processed_at_s < self.process_interval_s:
            return []

        prev_count = len(self._segments)
        self._segments = self._run_predict_on_buffer()
        self._last_processed_at_s = self.buffer.current_time

        # Diff: anything beyond ``prev_count`` is "new" in append-only terms.
        # In practice segment lists can rewrite (e.g. tentative pre-commit
        # segments collapsed into a confirmed one), so this is a best-effort
        # newest-first append, not a strict diff.
        return self._segments[prev_count:]

    def flush(self) -> list[Segment]:
        """Force a final predict pass regardless of the interval timer.

        Use at end-of-stream to make sure the last buffered audio is scored.
        Returns the full final segment list.
        """
        if self.buffer.fill_samples == 0:
            return list(self._segments)
        self._segments = self._run_predict_on_buffer()
        self._last_processed_at_s = self.buffer.current_time
        return list(self._segments)

    # -- introspection --------------------------------------------------------

    def get_segments(self) -> list[Segment]:
        return list(self._segments)

    def get_state(self) -> StreamingState:
        return StreamingState(
            committed_shabad_id=self._committed_shabad_id,
            audio_seconds=self.buffer.current_time,
            n_segments=len(self._segments),
            last_processed_at_s=self._last_processed_at_s,
        )

    # -- inner: temp-file naive predict ---------------------------------------

    def _run_predict_on_buffer(self) -> list[Segment]:
        """Snapshot the buffer to a temp WAV and run ``engine.predict()``.

        Replaced by an in-memory Core ML path in iOS (M5). The output contract
        — a list of ``Segment`` — stays identical.
        """
        import soundfile as sf

        audio = self.buffer.read_all()
        if len(audio) == 0:
            return []

        # uem_start measures the absolute offset of the *earliest* sample in
        # the buffer. If the ring has aged out, predict() will treat that
        # offset as t=0; for the streaming-on-benchmark audit we keep the full
        # session in scope by sizing the ring large enough that nothing ages.
        uem_start = self.buffer.earliest_time

        # Determine shabad mode: oracle if reset() committed one, else blind.
        shabad_for_predict = self._committed_shabad_id if self._oracle_locked else None

        tmp_path: pathlib.Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = pathlib.Path(tmp.name)
            sf.write(str(tmp_path), audio, self.buffer.sample_rate)

            result = predict(
                tmp_path, self.corpora,
                shabad_id=shabad_for_predict,
                uem_start=uem_start,
                config=self.config,
            )
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        # If blind ID just confirmed a shabad, remember it for future calls.
        if not self._oracle_locked:
            self._committed_shabad_id = result.shabad_id

        return result.segments
