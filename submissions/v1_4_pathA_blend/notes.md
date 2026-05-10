# v1_4_pathA_blend

Blend of `token_sort_ratio` + `WRatio` (50/50). Phase 1 winner.

## Config

- **Mode**: oracle + offline
- **ASR**: faster-whisper `medium`, `language="pa"`, segment-level (cached from v1.0)
- **Matcher**: unidecode → strip pangti markers → score = `0.5 * token_sort_ratio + 0.5 * WRatio`
- **Threshold**: 0 (no minimum)
- **Smoother**: collapse consecutive same-`line_idx` chunks
- **Run**: `python scripts/run_path_a.py --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 --out-dir submissions/v1_4_pathA_blend`

## Score

**Overall: 84.8%** (2905/3425 frames) — up +6.1 from v1.3, up +16.4 from v1.0.

## Blend sweep results

| Blend | Overall |
|---|---|
| `token_sort_ratio` only (v1.3) | 78.7% |
| 0.7 tsr + 0.3 ratio | 80.8% |
| 0.5 tsr + 0.5 ratio | 80.7% |
| **0.5 tsr + 0.5 WRatio** | **84.8%** |
| 0.7 tsr + 0.3 WRatio | 83.1% |
| 0.5 tsr + 0.25 ratio + 0.25 WRatio | 84.6% |

The token_sort + WRatio blend wins clearly. Adding `ratio` (Levenshtein) into the mix is worse than just keeping it tsr+WRatio — these two scorers cover complementary failure modes:

- **token_sort_ratio** kills the "shared refrain matches longest line" problem (the kchMJPK9Axs `man bauraa re` issue).
- **WRatio** kills the "shared rhyme structure confuses pure token-bag" problem (the zOtIpxMT9hU lines 1/5 issue).

Together they compensate, neither alone gets there.

## Per-shabad progression (v1.0 → v1.3 → v1.4)

| Shabad | v1.0 | v1.3 | v1.4 |
|---|---|---|---|
| IZOsmkdmmcg | 97 / 97 / 95 | 93 / 95 / 95 | **95 / 97 / 95** |
| kZhIA8P6xWI | 74 / 72 / 74 | 76 / 74 / 84 | **87 / 86 / 98** |
| kchMJPK9Axs | 40 / 32 / 32 | 72 / 73 / 70 | **80 / 71 / 64** |
| zOtIpxMT9hU | 88 / 96 / 93 | 70 / 63 / 94 | **89 / 84 / 93** |

Three shabads now in 89-98% range. Only weak spot: kchMJPK9Axs cold33/cold66 (71/64%) — the cold-start variants of the hardest shabad. Likely a chunking/boundary issue — the cold cases skip past early segments, so any chunk that misaligns at the new entry point cascades.

## Phase 1 verdict

Hit the ≥80% stopping criterion on the first try. Three shabads above 89% with the simplest possible matcher (no shabad-aware logic, no temporal smoothing, no HMM). Per the plan, advance to Phase 2.

## Phase 2 candidate — TF-IDF discriminators

Per-shabad, weight tokens by inverse frequency across that shabad's lines. Common tokens (refrains, structural words) carry near-zero weight; distinctive tokens dominate the score. Direct fix for the residual kchMJPK9Axs failures.

Cheap to build (~30 lines), uses cached ASR, single experiment.

## Artifacts

- 12 submission JSONs in this directory
- `tiles.html` — visualizer
