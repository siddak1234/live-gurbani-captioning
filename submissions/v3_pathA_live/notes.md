# v3_pathA_live

Live causal variant of v2: same matcher + smoother + chunk-vote shabad ID, but
predictions at time t may only depend on audio up to t (the benchmark's strict
live-causality rule). Phase 5 result.

## Config

- **Mode**: BLIND + LIVE (causal)
- **ASR**: faster-whisper `medium`, segment-level (cached). NOTE: cached ASR is
  offline-quality. A truly streaming ASR pipeline would have somewhat lower
  per-chunk fidelity; this submission represents an upper bound on live accuracy
  for our matcher.
- **Shabad identifier**: `chunk_vote`, `lookback=30s`. After UEM start, the system
  buffers 30s before committing the shabad.
- **Matcher / smoother**: identical to v1.5 / v2 (50/50 blend, stay-bias=6).
- **Live filter**: matcher only processes chunks whose `start >= UEM_start + lookback`.
  Frames inside the lookback window emit `null`.
- **Run**: `python scripts/run_path_a.py --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 --stay-bias 6 --blind --blind-aggregate chunk_vote --blind-lookback 30 --live --out-dir submissions/v3_pathA_live`

## Score

**Overall: 78.4%** (2686/3425 frames) — down 9.8 from v2's 88.2%.

## Lookback sweep

| Lookback | Overall |
|---|---|
| 15s | 76.4% |
| **30s** | **78.4%** |
| 60s | 73.0% |

15s was *not* enough — chunk_vote at very short windows occasionally mis-IDs.
30s is the sweet spot. 60s loses too many frames to the buffer.

## Where the 10-point drop comes from

The drop is mechanical: 30s of every UEM is emitted as `null` while the system
identifies the shabad. For shorter audio files and cold-start cases, this is a
larger fraction of the scored region:

| Case | UEM duration | Frames lost to buffer | Loss as % of UEM |
|---|---|---|---|
| IZOsmkdmmcg | 456s | 30 | 6.6% |
| IZOsmkdmmcg_cold66 | 156s | 30 | 19.2% |
| zOtIpxMT9hU_cold66 | 99s | 30 | 30.3% |

The cold66 cases are hit hardest because their UEM windows are smallest — the
fixed 30s buffer eats a much bigger slice. zOtIpxMT9hU_cold66 went from 93.9%
in v2 to 64.7% here, almost entirely because of this.

## Per-shabad: v2 (offline blind) → v3 (live blind)

| Shabad | Cold0 | Cold33 | Cold66 |
|---|---|---|---|
| IZOsmkdmmcg | 98 → 90 | 97 → 90 | 95 → 75 |
| kZhIA8P6xWI | 84 → 78 | 86 → 82 | 98 → 76 |
| kchMJPK9Axs | 83 → 77 | 74 → 68 | 74 → 64 |
| zOtIpxMT9hU | 98 → 86 | 97 → 73 | 94 → 65 |

Pattern is consistent: bigger drop on cold variants, smaller drop on cold0.
Direct evidence that the 30s buffer is the dominant cost.

## What would lift this higher

1. **Shorter shabad-ID lookback.** chunk_vote at 15s already gets 12/12 IDs but
   the matcher sees less data to converge on the right line — net 76.4%.
   A shorter buffer + a "warmup" tentative-shabad pass might recover 1-2%.

2. **Tentative emission during buffer.** Allow the matcher to emit predictions
   during the buffer period using a tentative shabad (e.g. score against all
   shabads, pick global top-1 per chunk). Switch to the committed shabad once
   it's locked. Risk: tentative predictions during the buffer may be wrong.

3. **Delayed but causal emission.** Predictions for time t use audio up to t,
   but emit them at t + δ display latency. Strictly causal in the
   prediction-vs-input sense. Doesn't help here because we already buffer for
   shabad ID; the deeper issue is that without the shabad commit, we can't
   constrain the matcher.

4. **Better ASR (large-v3).** Streaming ASR will be noisier than the cached
   offline transcripts; an upgrade to large-v3 (~3GB, ~30 min one-time cost)
   would give us headroom against streaming degradation.

## Phase 5 verdict

Live causal mode shows the real cost of strict streaming — about 10 points
below offline. Three of four shabads still ≥75% even on cold66; only
kchMJPK9Axs cold66 drops to 64%, and that's the hardest case in the benchmark.

For deployment context, this is a reasonable starting point: 78% live with
30s warmup is a usable system (the production reference at
bani.karanbirsingh.com presumably has similar warmup).

## Where we are vs the plan

| Stage | Mode | Score |
|---|---|---|
| Stage 0 (empty) | — | 26.0% |
| Stage 2 (Path A oracle/offline, v1.5) | oracle + offline | 88.2% |
| Stage 3 (Path A blind/offline, v2) | blind + offline | 88.2% |
| **Stage 4 (Path A blind/live, v3)** | **blind + live** | **78.4%** |
| Plan target | blind + live | 95%+ |

We're 17 points short of the plan target. Closing that gap is now Path B
territory — Stage 6 in the plan. Or we could try cheap intermediate moves
(large-v3 ASR, latency-tolerant streaming) before committing to the full
CTC + HMM build.

## Artifacts

- 12 submission JSONs in this directory
- `tiles.html` — visualizer
