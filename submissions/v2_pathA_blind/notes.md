# v2_pathA_blind

Blind variant of v1.5: identifies the shabad from audio, then runs the v1.5 matcher.
Phase 4 winner.

## Config

- **Mode**: BLIND (no GT shabad_id used) + offline
- **ASR**: faster-whisper `medium`, `language="pa"`, segment-level (cached)
- **Shabad identifier**: `aggregate=chunk_vote`, `lookback=60s`, scorer = same v1.5 blend as the matcher (`0.5 * token_sort_ratio + 0.5 * WRatio`)
- **Matcher**: v1.5 blend
- **Smoother**: stay-bias = 6
- **Run**: `python scripts/run_path_a.py --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 --stay-bias 6 --blind --blind-aggregate chunk_vote --blind-lookback 60 --out-dir submissions/v2_pathA_blind`

## Score

**Overall: 88.2%** (3020/3425 frames) — *identical to v1.5 oracle.*

## Shabad ID accuracy: 12/12

Every case identified correctly at every lookback (30s through 360s). No score drop vs oracle.

## Aggregation sweep

| Method | Lookback | IDs correct | Overall |
|---|---|---|---|
| max | 30s | 9/12 | 66.2% (initial v2) |
| max | 90s | 8/12 | 58.8% |
| topk:5 | 90s | 10/12 | 71.7% (best non-vote) |
| tfidf | 90s | 8/12 | 62.4% |
| **chunk_vote** | **30s+** | **12/12** | **88.2%** ✨ |

## Why chunk_vote works (and the others don't)

The 2-3 ID failures in `max` and `topk:5` were all flavors of the same trap:
the buffer is dominated by the singer repeating one line of the right shabad,
but a *wrong* shabad happens to have a single line that shares a phonetic hook
with that line. Aggregating over the whole shabad (max or top-K sum) lets the
wrong shabad's other generic phonetic content build up enough score to win.

Concrete example: IZOsmkdmmcg cold0 buffer is `"parmeshar ka thaan ..."`
repeated. Shabad 3712 line 3 is `"... asathir jaa kaa thaan he"` — the
`"kaa thaan"` hook is shared. With topk:5 sum, 3712 wins because *its other
five lines also collect moderate noise scores*. With chunk_vote, every chunk
of `"parmeshar ka thaan"` independently votes for shabad 4377 (whose line 1
matches the entire chunk strongly), and 3712 collects no votes.

In short:
- Aggregating over **lines** (max, topk) inflates wrong shabads with phonetic noise.
- Aggregating over **chunks** (vote) lets repetition reinforce the right answer.

The right granularity is per-chunk because that's where the signal is independent.

## Per-shabad scores (matches v1.5 exactly)

| Shabad | Cold0 | Cold33 | Cold66 |
|---|---|---|---|
| IZOsmkdmmcg | 98.2 | 97.4 | 94.9 |
| kZhIA8P6xWI | 84.2 | 85.5 | 98.1 |
| kchMJPK9Axs | 82.6 | 74.4 | 73.9 |
| zOtIpxMT9hU | 97.9 | 96.9 | 93.9 |

## Caveats

- **Only 4 candidate shabads.** Real-world deployment would test against 1000s+ of common kirtan shabads (or the full SGGS, ~70k lines × shabad mapping). With more candidates, chunk_vote should degrade gracefully (per-chunk top-1 should still pick the right shabad if it's distinctively better) but I haven't verified it. Worth doing eventually.
- **Offline mode.** The shabad ID buffers `lookback_seconds` of future audio; in live mode we'd need to wait that long before committing. 30s is achievable but introduces latency. Phase 5 problem.

## What's next per the plan

Phase 4 hit the stopping criterion (no real drop from oracle). Move to **Phase 5: drop offline → live causal streaming**. Re-architect ASR + matcher + smoother as a streaming pipeline. Latency budget: a few seconds. Expected: 2-5% drop from offline numbers.

## Artifacts

- 12 submission JSONs in this directory
- `tiles.html` — visualizer
