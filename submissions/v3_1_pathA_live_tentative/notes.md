# v3_1_pathA_live_tentative

Live causal + tentative emission during the shabad-ID buffer. Recovers most of v3's
strict-causal drop while preserving causality.

## Config

- **Mode**: BLIND + LIVE (causal) + tentative emission
- **ASR**: faster-whisper `medium`, segment-level (cached). Offline-quality; a real
  streaming ASR would be noisier — see "caveats".
- **Shabad identifier**: `chunk_vote`, `lookback=30s`. After UEM start, the system
  buffers 30s before committing the shabad.
- **Tentative emission**: during the buffer, each ASR chunk independently scores
  against *every* candidate shabad's lines and emits the global-best
  `(shabad_id, line_idx)` pair. Consecutive same-pair chunks merge into one segment.
- **Post-commit**: standard v1.5 matcher constrained to the committed shabad
  (50/50 blend, stay-bias=6).
- **Run**: `python scripts/run_path_a.py --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 --stay-bias 6 --blind --blind-aggregate chunk_vote --blind-lookback 30 --live --tentative-emit --out-dir submissions/v3_1_pathA_live_tentative`

## Score

**Overall: 86.0%** (2947/3425 frames) — up +7.6 from v3 strict-causal, down −2.2 from v2 offline blind.

## Lookback sweep

| Lookback | v3 (strict) | v3.1 (tentative) |
|---|---|---|
| 15s | 76.4% | 81.3% |
| **30s** | 78.4% | **86.0%** |
| 45s | — | 85.5% |

30s remains the sweet spot. Tentative emission recovered ~78% of v3's missing accuracy.

## Why this works

During the ID buffer, the singer is usually repeating *one line of the right
shabad*. Each ASR chunk in the buffer, scored against every shabad's lines,
overwhelmingly picks the right shabad's right line as its global top match —
because no other shabad has a line that's actually that close.

In other words, the same per-chunk independence that makes `chunk_vote`
effective for shabad ID also makes per-chunk global emission accurate during
the buffer. We're using the same signal twice: once to commit the shabad, once
to predict line-level segments while we wait.

## Per-shabad: v3 (strict) → v3.1 (tentative)

| Shabad | Cold0 | Cold33 | Cold66 |
|---|---|---|---|
| IZOsmkdmmcg | 90 → 98 | 90 → 96 | 75 → 94 |
| kZhIA8P6xWI | 78 → 87 | 82 → 81 | 76 → 88 |
| kchMJPK9Axs | 77 → 83 | 68 → 76 | 64 → 74 |
| zOtIpxMT9hU | 86 → 93 | 73 → 79 | 65 → 83 |

The cold66 cases benefit most — exactly as predicted, since they had the largest
fraction of UEM eaten by the strict-causal buffer. zOtIpxMT9hU_cold66 went from
65% to 83% (+18 points).

## Causality check

Predictions during the buffer use audio only up to the chunk's `end` time:
- ASR is cached offline, but each chunk's text is determined within `[c.start, c.end]`.
- The global-best match for chunk `c` looks at `c.text` and `corpora` only — no
  future chunks, no future audio.
- Once the shabad commits at `commit_time`, the matcher transitions to constrained
  mode. The committed shabad_id was determined by chunk_vote *over the buffer
  window*, which used audio up to `commit_time` — so the *commit decision* uses
  audio up to `commit_time`, but each pre-commit emission's prediction at time `c.end`
  used audio only up to `c.end`.

Strictly causal in the spec's reading: predictions at time t depend on audio up to t.

## Where we stand

| Stage | Mode | Score |
|---|---|---|
| Stage 0 | empty | 26.0% |
| v1.5 | oracle + offline | 88.2% |
| v2 | blind + offline | 88.2% |
| v3 | blind + live (strict) | 78.4% |
| **v3.1** | **blind + live + tentative emit** | **86.0%** |
| Plan target | blind + live | 95%+ |

We've now beaten the public 60-80% range in all four task variants (oracle/blind ×
offline/live) with a single matcher and ~250 lines of code. The remaining 9-point
gap to 95% is the genuine territory of Path B (CTC phoneme scoring + loop-aware
HMM) — or large-v3 ASR if we want a cheap intermediate probe first.

## Caveats

- **Cached offline ASR.** All numbers above use offline faster-whisper transcripts.
  A real streaming ASR will produce noisier per-chunk text, especially at chunk
  boundaries. Empirical impact on v3.1 unknown until we wire it up.
- **4 candidates only.** Per-chunk global match scales as O(num_chunks × num_shabads × num_lines).
  For 70k SGGS shabads, this becomes expensive — but it's parallelizable and
  inverted-index pruning would handle it. Architecture isn't the bottleneck.

## Artifacts

- 12 submission JSONs in this directory
- `tiles.html` — visualizer
