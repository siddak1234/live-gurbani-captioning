# v1_pathA_oracle

First Path A run: faster-whisper (medium) + rapidfuzz (WRatio) + naive smoothing.

## Config

- **Mode**: oracle (GT shabad_id given) + offline (full audio available, non-causal smoothing OK)
- **ASR**: faster-whisper `medium`, `language="pa"`, `beam_size=5`, segment-level timestamps, `compute_type="int8"` (CPU)
- **Matcher**: unidecode → strip pangti markers / `(n)` hints / `rahaau` token → rapidfuzz `WRatio` against each line's `transliteration_english`
- **Smoother**: collapse consecutive same-`line_idx` chunks; `score < 55` → null
- **Run command**: `python scripts/run_path_a.py`

## Score

**Overall: 68.4% frame accuracy** (2344/3425 frames, collar=1s)

| Video | Shabad | Cold0 | Cold33 | Cold66 |
|---|---|---|---|---|
| IZOsmkdmmcg | 4377 | 98.2% | 97.4% | 94.9% |
| kZhIA8P6xWI | 1821 | 73.9% | 71.5% | 74.3% |
| kchMJPK9Axs | 1341 | **39.5%** | **31.7%** | **31.5%** |
| zOtIpxMT9hU | 3712 | 88.2% | 96.4% | 92.9% |

## What worked

- Whisper outputs Devanagari for Punjabi audio (despite `language="pa"`); BaniDB's `transliteration_english` is in latin. unidecode bridges them well — phonetics survive both transformations.
- WRatio is forgiving enough to handle Whisper's vowel drops (`prmeshr` ↔ `paramesar`) on shabads without internal repetition.
- Three of four shabads scored 88%+ with this naive pipeline — already at the upper end of public claims.

## Why kchMJPK9Axs (1341) crashes

Every line of this shabad ends with the refrain **`man bauraa re`** ("oh foolish mind"). When the ASR emits a short chunk like `"samajh man bauraa"`, WRatio matches it nearly equally against multiple candidate lines, and the longest line (line 1: `kaalaboot kee hasatanee man bauraa re chalat rachio jagadhees`) wins on token-richness alone. Result: large stretches of audio that should be lines 3, 4, 5, 6 get mis-attributed to line 1.

Concrete example: GT says line 4 from 49.8s-100.7s; we predicted line 1. The ASR there is `"nirpe hoe na har paje man bauraa re"` — phonetically a clear match for line 4 (`nirabhai hoi na har bhaje man bauraa re`), but WRatio assigns it to line 1 because line 1 also contains `man bauraa re` and is longer.

Other shabads don't share an internal refrain like this, which is why they scored cleanly.

## Hypotheses for the next run (v1.1 or v2)

1. **Margin-based confidence gate** — emit null when `top1 - top2 < margin_threshold`. Cheap fix; converts ambiguous chunks (which currently score wrong) into nulls (which the scorer accepts in gaps).
2. **Down-weight shared n-grams** — for each shabad, identify n-grams appearing in multiple lines and either remove them before scoring or reduce their weight. Custom per-shabad, but the cache already lets us precompute this.
3. **Switch to `token_sort_ratio` or `ratio`** — less generous than `WRatio`'s partial_ratio component for short overlaps. Would test in isolation first.
4. **Use a stay-bias / loop-aware decoder** — once we predict line 4 with high confidence, transitions should prefer staying on line 4 over jumping to line 1. Path B territory but a simple "stay unless margin > X" heuristic could capture most of the win cheaply.
5. **Larger ASR chunks** — `word_timestamps=True` would give finer-grained chunks; `--vad-filter` might give cleaner segment boundaries. Worth a probe.

## Next concrete step

Try (1) — margin gate — first. Single-parameter change, no model swap, easy to A/B. Expect kchMJPK9Axs to climb from 30s into 60s+, modest losses on the cleaner shabads (some confident chunks downgraded to null).

## Artifacts

- 12 submission JSONs in this directory
- `tiles.html` — visualizer (open in any browser; hover to see GT vs pred Gurmukhi)
