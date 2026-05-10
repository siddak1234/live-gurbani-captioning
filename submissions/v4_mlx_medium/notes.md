# v4_mlx_medium

Exploratory: same Path A pipeline as v3.2, but transcribed with `mlx-whisper`
medium model on Apple Silicon GPU instead of `faster-whisper` medium on CPU.

## Score: 70.6% (regression of -15.9 from v3.2's 86.5%)

## What this run is for

Documenting the regression so future sessions don't re-discover it.

We swapped backends to get GPU acceleration. The neural net weights are the
same; the *scaffolding* around it is not. Specifically:

- mlx-whisper does not bundle Silero VAD pre-filtering; faster-whisper does.
- mlx-whisper uses fixed-time-window chunking; faster-whisper uses speech-aware boundaries.
- Defaults for decoding/temperature/threshold differ.

Result: mlx produced **fewer, longer chunks** (e.g. 13 chunks for an audio
where fw produced 27). Our matcher, smoother, and chunk-vote shabad ID were
all tuned for fw's granularity. Two of 12 blind IDs failed; tentative emission
yielded fewer, less-precise pre-commit segments.

## Run command

```bash
python scripts/run_path_a.py --backend mlx_whisper --model medium \
  --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 --stay-bias 6 \
  --blind --blind-aggregate chunk_vote --blind-lookback 30 \
  --live --tentative-emit \
  --out-dir submissions/v4_mlx_medium
```

## Lesson

Library swaps under a tuned downstream pipeline are not free even when the
neural net is identical. Validate per-chunk granularity before assuming parity.
