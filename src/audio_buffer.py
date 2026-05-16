"""Ring buffer for streaming 16 kHz mono PCM.

Used by ``src.streaming_engine.StreamingEngine`` to hold a sliding window of
recent audio (typically 60-120 s) while new PCM frames arrive. iOS will use
the same contract via ``AVAudioEngine`` callbacks.

Design notes
------------

* Pre-allocated ``np.float32`` array of fixed capacity (samples).
* Internally tracks two cursors:
    - ``_n_total_samples``: monotonic count of every sample ever appended.
    - ``_fill``: how many samples currently live in the buffer.
* ``read_window(start_s, end_s)`` interprets timestamps as *absolute* —
  measured from the start of the session — and returns whatever overlaps the
  ring's current span. Audio that has aged out returns shorter slices.

The buffer is intentionally simple. No thread safety, no resampling beyond
basic mono mixdown, no exotic interpolation. The iOS port replaces this with
``AVAudioPCMBuffer`` + a ``CircularBuffer`` from Apple's swift-collections;
the Python version exists to validate the streaming engine in isolation.
"""

from __future__ import annotations

import numpy as np


class AudioBuffer:
    """Fixed-capacity ring buffer for 16 kHz mono float32 PCM."""

    def __init__(self, capacity_s: float = 60.0, sample_rate: int = 16000):
        if capacity_s <= 0 or sample_rate <= 0:
            raise ValueError("capacity_s and sample_rate must be positive")
        self.sample_rate = sample_rate
        self.capacity = int(capacity_s * sample_rate)
        self._buffer = np.zeros(self.capacity, dtype=np.float32)
        self._n_total_samples = 0  # monotonic count since last reset()
        self._fill = 0             # samples currently in buffer

    # -- introspection --------------------------------------------------------

    @property
    def current_time(self) -> float:
        """Total seconds of audio appended since the last ``reset()``."""
        return self._n_total_samples / self.sample_rate

    @property
    def earliest_time(self) -> float:
        """Earliest absolute timestamp still in the buffer (oldest data)."""
        oldest_sample = max(0, self._n_total_samples - self._fill)
        return oldest_sample / self.sample_rate

    @property
    def fill_samples(self) -> int:
        return self._fill

    @property
    def fill_seconds(self) -> float:
        return self._fill / self.sample_rate

    # -- mutation -------------------------------------------------------------

    def reset(self) -> None:
        """Drop all buffered audio and reset the absolute clock."""
        self._n_total_samples = 0
        self._fill = 0

    def append_pcm(self, audio: np.ndarray, sr: int = 16000) -> None:
        """Append a chunk of PCM. Resamples to ``self.sample_rate`` if needed.

        Multi-channel input is downmixed to mono by averaging across channels.
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32, copy=False)

        if sr != self.sample_rate:
            import scipy.signal
            new_len = int(len(audio) * self.sample_rate / sr)
            audio = scipy.signal.resample(audio, new_len).astype(np.float32)

        n = len(audio)
        if n == 0:
            return

        if n >= self.capacity:
            # New chunk exceeds full capacity — keep only its tail.
            self._buffer[:] = audio[-self.capacity:]
            self._fill = self.capacity
        else:
            overflow = self._fill + n - self.capacity
            if overflow > 0:
                # Shift existing data left to make room.
                self._buffer[: self._fill - overflow] = self._buffer[overflow: self._fill]
                self._fill -= overflow
            self._buffer[self._fill : self._fill + n] = audio
            self._fill += n

        self._n_total_samples += n

    # -- read -----------------------------------------------------------------

    def read_window(self, start_s: float, end_s: float) -> np.ndarray:
        """Return audio in ``[start_s, end_s)`` (absolute seconds).

        If part or all of the window has aged out of the ring, returns only the
        portion that is still buffered. Returns a 0-length array if the window
        lies entirely outside the buffered span.
        """
        if end_s <= start_s:
            return np.zeros(0, dtype=np.float32)

        earliest = self.earliest_time
        latest = self.current_time
        clip_start = max(start_s, earliest)
        clip_end = min(end_s, latest)
        if clip_start >= clip_end:
            return np.zeros(0, dtype=np.float32)

        offset = int(round((clip_start - earliest) * self.sample_rate))
        n_samples = int(round((clip_end - clip_start) * self.sample_rate))
        return self._buffer[offset : offset + n_samples].copy()

    def read_all(self) -> np.ndarray:
        """Return a copy of every sample currently in the ring."""
        return self._buffer[: self._fill].copy()
