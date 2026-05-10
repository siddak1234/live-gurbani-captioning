# v1_3_pathA_tsr

Sweep of rapidfuzz scorers; winner is `token_sort_ratio` with `threshold=0`.

## Config

- **Mode**: oracle + offline
- **ASR**: same as v1.0 — faster-whisper `medium`, `language="pa"`, segment-level
- **Matcher**: unidecode → strip pangti markers → `fuzz.token_sort_ratio` against `transliteration_english`. **No score threshold** (every chunk gets matched to its top candidate).
- **Smoother**: collapse consecutive same-`line_idx` chunks
- **Run**: `python scripts/run_path_a.py --ratio token_sort_ratio --threshold 0 --out-dir submissions/v1_3_pathA_tsr`

## Score

**Overall: 78.7%** (2694/3425 frames) — up 10.3 points from v1.0's 68.4%

## Sweep results (all 8 cells)

| Scorer | th=0 | th=55 |
|---|---|---|
| WRatio | 72.9% | 68.4% (v1.0) |
| ratio | 73.3% | 60.1% |
| **token_sort_ratio** | **78.7%** | 59.2% |
| partial_token_set_ratio | 59.7% | 57.2% |

Two takeaways:
1. **Threshold 55 was hurting us across the board.** With `threshold=0` (no minimum), we accept some bad matches but gain way more correct ones. Worth keeping at 0 — `score_threshold` is the wrong signal.
2. **`token_sort_ratio` clearly wins.** It sorts tokens alphabetically before scoring, so word-order differences (which Whisper produces against the canonical line order) don't hurt.

## Per-shabad (v1.0 → v1.3)

| Shabad | v1.0 | v1.3 | Δ |
|---|---|---|---|
| IZOsmkdmmcg | 97 / 97 / 95 | 93 / 95 / 95 | -2 avg |
| kZhIA8P6xWI | 74 / 72 / 74 | 76 / 74 / 84 | +5 avg |
| kchMJPK9Axs | 40 / 32 / 32 | **72 / 73 / 70** | **+37 avg** ✨ |
| zOtIpxMT9hU | 88 / 96 / 93 | 70 / 63 / 94 | -17 avg |

## What worked

- The `man bauraa re` collapse on shabad 1341 is fixed. token_sort_ratio doesn't reward "found a substring inside a longer line" the way `WRatio`'s partial-ratio component did.
- Keeping `threshold=0` lets short, high-quality chunks with low absolute scores still register.

## What regressed

- **zOtIpxMT9hU lost ~17 points** on the first two cases. Lines 1 and 5 share end-rhyme structure (`raam he` / `haaraa`). After token-sort, the shared tokens dominate and confuse the matcher.
- **IZOsmkdmmcg lost ~2 points** — was already near-perfect, marginal change.

## Diagnosis

There's no single ratio that wins on every shabad. Different shabads have different overlap structures:
- Shared **refrain**: WRatio/partial_ratio over-credits long lines containing the refrain (kchMJPK9Axs)
- Shared **rhyme/structural words**: token_sort_ratio over-credits any line with the same common words (zOtIpxMT9hU)

Same root cause: **shared tokens between lines aren't penalized**. They should carry less weight than tokens unique to a single line.

## Hypothesis for v2 (Phase 2)

**TF-IDF style line discriminators per shabad.**
- Compute IDF for each token across the shabad's lines.
- Rare tokens (high IDF) carry more weight; refrain-like tokens carry near-zero weight.
- Score chunks against each line using IDF-weighted similarity.

Direct fix for the shared-token issue. Per-shabad pre-computation, ~free at runtime.

## Artifacts

- 12 submission JSONs in this directory
- `tiles.html` — visualizer
- Reproduce: see `## Run` above (single command, ASR cached)
