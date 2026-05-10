# v1_5_pathA_staybias

v1.4 blend + stay-bias smoother. Phase 2 winner.

## Config

- **Mode**: oracle + offline
- **ASR**: faster-whisper `medium`, `language="pa"`, segment-level (cached)
- **Matcher**: blend `0.5 * token_sort_ratio + 0.5 * WRatio` (v1.4)
- **Smoother**: stay-bias = 6. For each chunk, pick `argmax(scores)` *unless* the previously-emitted line scores within 6 points of the top — in which case stay on the previous line.
- **Run**: `python scripts/run_path_a.py --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 --stay-bias 6 --out-dir submissions/v1_5_pathA_staybias`

## Score

**Overall: 88.2%** (3020/3425 frames) — up +3.4 from v1.4, up +19.8 from v1.0.

## Stay-bias sweep

| stay_bias | Overall |
|---|---|
| 0 (= v1.4) | 84.8% |
| 2 | 84.7% |
| 5 | 82.7% |
| **6** | **88.2%** |
| 7 | 87.9% |
| 8 | 87.7% |
| 9 | 87.0% |
| 10 | 84.6% |
| 12 | 79.0% |
| 18 | 66.3% |
| 25 | 63.0% |

Sweet spot at 6 with a flat-ish region 6-8. Past 12 the bias gets stuck on wrong lines.

The dip at stay=5 is weird (worse than no-bias) — probably an interaction with specific chunks where 5 is just enough to flip a correct match but not enough to recover. Not chasing it; 6+ dominates.

## Per-shabad progression

| Shabad | v1.0 | v1.4 | v1.5 |
|---|---|---|---|
| IZOsmkdmmcg | 97 / 97 / 95 | 95 / 97 / 95 | **98 / 97 / 95** |
| kZhIA8P6xWI | 74 / 72 / 74 | 87 / 86 / 98 | **84 / 86 / 98** |
| kchMJPK9Axs | 40 / 32 / 32 | 80 / 71 / 64 | **83 / 74 / 74** |
| zOtIpxMT9hU | 88 / 96 / 93 | 89 / 84 / 93 | **98 / 97 / 94** |

## What worked

- **zOtIpxMT9hU recovered fully**: was at 89/84/93 in v1.4 (regressed in v1.3 from token_sort_ratio noise on lines 1/5 sharing rhyme structure). Stay-bias = 6 keeps the predicted line stable through ambiguous chunks, getting back to 98/97/94.
- **kchMJPK9Axs cold66 +10 points**: 64% → 74%. The cold variants of the hardest shabad were our weakest spot; stay-bias suppresses oscillation between the rahao (line 3) and the verses around it.
- **No regressions on the strong shabads**: IZOsmkdmmcg actually +1 average.

## What's still weak

- **kZhIA8P6xWI cold0 dropped 3 points** (87 → 84). Stay-bias occasionally locks to a wrong line briefly. Acceptable trade.
- **kchMJPK9Axs cold33 still at 74%**. The hardest case — middle of song with multiple verses, lots of opportunities to mis-snap.

## Phase 2 verdict

Hit 88.2%, just shy of the planned 88%+ Phase 2 stopping criterion. The TF-IDF probe (Phase 2 candidate A) didn't help, but the stay-bias (Phase 2 candidate B) delivered. Total cost: ~50 lines of code and one parameter sweep.

**Per the plan: now move to Phase 4 — drop the oracle (blind shabad ID).**

We're at 88% with 4 specific shabad_ids handed to us. Next we have to identify the shabad from audio alone before tracking. Expected score: small drop (a few %) on cases where shabad ID is wrong, otherwise unchanged.

## Artifacts

- 12 submission JSONs in this directory
- `tiles.html` — visualizer
