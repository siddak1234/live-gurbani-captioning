# Silver 300h segment eval — Phase 2.10

Generated 2026-05-16/17 as the automated bridge while gold OOS v1 remains
pending.

## Data

- Source: `surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical`
- Slice: shards `10-19` (the `v5b_mac_diverse` adapter trained on shards `0-9`)
- Pull target: `make data-silver-300h`
- Local manifest: `training_data/silver_300h_holdout/manifest.json`
- Pull result: 8,306 clips, 16.701 h, 19 source videos, 308 shabad tokens
- Training-slice overlap: 0 video overlap with `training_data/v5b_mac_diverse`
- Benchmark holdouts: video/content holdouts active; 0 benchmark video/content rejections in this slice

This is a **silver** eval. The labels are high-confidence canonical matches from
the online dataset, not hand-corrected full-shabad GT. It is useful for ASR /
canonical-text diagnostics; it does not replace gold OOS promotion.

## Command

```bash
make eval-silver-300h \
  SILVER_LIMIT=100 \
  SILVER_ADAPTER_DIR= \
  SILVER_OUT=submissions/silver_300h_surt_base_100.json

make eval-silver-300h \
  SILVER_LIMIT=100 \
  SILVER_ADAPTER_DIR=lora_adapters/v5b_mac_diverse \
  SILVER_OUT=submissions/silver_300h_v5b_100.json
```

Rows are selected with deterministic round-robin-by-video sampling, so the
100-row run spans 19 videos and 28 shabad tokens.

## Result

| Run | n | videos | shabads | mean WRatio | median WRatio | exact normalized |
|---|---:|---:|---:|---:|---:|---:|
| `surt-small-v3` base | 100 | 19 | 28 | 96.29 | 100.00 | 75.0% |
| `surt-small-v3 + v5b_mac_diverse` | 100 | 19 | 28 | 96.33 | 100.00 | 73.0% |

## Interpretation

The adapter is effectively neutral on broad silver ASR segments: +0.05 mean
WRatio, -2.0 pp exact normalized match. This matches the earlier Phase 2.6
diagnosis: `v5b_mac_diverse` is not the source of a large raw-ASR gain. The
best paired-benchmark lift came from runtime architecture (`phase2_9_loop_align`
at 91.2%), especially ID-lock + retro-buffered finalization + loop/null-aware
alignment.

Recommended next experiment: use the silver set to inspect weak videos
(`iQAbsSM5FO8`, `PYUPZn6wiR8`, `2d_Wy2Vb6n4`) and decide whether failures are
ASR omissions, label noise, or alignment/repetition artifacts. Do not scale
adapter training solely from this result.

## Weak-slice audit

Follow-up reports:

- `diagnostics/phase2_10_silver_weak_slices.md`
- `diagnostics/phase2_10_silver_source_audit.md`

Result: 11 / 100 rows were weak under `best(base, v5b) WRatio < 90`, and all 11
look like silver-label risks after checking original parquet metadata.

Breakdown:

- 10 / 11: model prediction matches raw caption better than canonical final.
- 1 / 11: heavy canonical fixes in the source row.

Examples:

- `PYUPZn6wiR8`: raw/model text `ਸਦਾ ਜਾਗੇ ਰਾਮ ਪਿਆਰੇ`; canonical final
  `ਸੰਤ ਜਨਾ ਰਾਮ ਪਿਆਰੇ`.
- `iQAbsSM5FO8`: raw/model text `ਮਨਿ ਪਰਚਾਇਆ`; canonical final `ਧਨਿ ਰੁਚ ਇਆ`
  with low retrieval margin.

Decision: silver failures do not justify broad adapter scaling. They mostly
confirm that silver labels are useful for diagnostics but unsafe as a promotion
gate. Keep gold OOS as the production gate and continue architecture-oriented
work around ID-lock / buffering / loop-aware alignment.
