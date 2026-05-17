# Phase 2.10 — silver weak-slice audit

## Why this checkpoint exists

The automated silver eval answered the first question: broad segment ASR is
already strong, and `v5b_mac_diverse` is essentially neutral versus the
`surt-small-v3` base model.

That means the next step should not be another broad training run. The useful
question is narrower:

> What distinguishes the rows and videos that still fail, and are those failures
> acoustic, label quality, repeated-line ambiguity, or model-specific regressions?

This checkpoint keeps us on the path to the 95% goal by avoiding aimless scaling.
If silver failures are mostly label noise, they should not drive model changes.
If failures are concentrated in specific acoustic conditions, the next training
run should target those conditions. If failures are repeated-line / partial-line
issues, the architecture work remains the right path.

## Inputs

- `submissions/silver_300h_surt_base_100.json`
- `submissions/silver_300h_v5b_100.json`
- `training_data/silver_300h_holdout/manifest.json`

The silver set itself is not promotion-grade gold OOS. It is an automated,
high-confidence canonical-label diagnostic from never-trained shards `10-19` of
the 300h dataset.

## First weak slices

From the 100-row round-robin run:

| Video | Base mean WRatio | v5b mean WRatio | Initial read |
|---|---:|---:|---|
| `iQAbsSM5FO8` | 66.3 | 64.6 | Strongest failure; inspect first. |
| `PYUPZn6wiR8` | 77.8 | 79.2 | Moderate failure; v5b helps slightly. |
| `2d_Wy2Vb6n4` | 93.1 | 95.1 | Mostly good, but has one low row worth inspecting. |

## Audit questions

For each weak row:

1. Is the prediction missing most target tokens, or only a short phrase?
2. Does the model repeat a nearby refrain / common line?
3. Is the target label itself suspicious when compared with the audio duration
   and neighboring rows from the same video?
4. Does v5b improve, regress, or leave the row unchanged?
5. Are weak rows concentrated by duration bucket, shabad token, or video?

## Decision rules

- **Mostly label noise:** keep silver for diagnostics, but do not optimize on
  those rows. Prefer gold OOS and official v2 timestamped labels when HF API
  access is resolved.
- **Mostly acoustic misses:** create a targeted acoustic data pull, not a broad
  blind Phase 3 scale-up.
- **Mostly repeated-line / partial-line ambiguity:** continue runtime alignment
  work; do not expect raw ASR fine-tuning alone to reach 95%.
- **v5b consistently worse on weak rows:** pause adapter promotion and inspect
  adapter training labels / overfitting.

## Next action

Add a deterministic weak-slice report that:

- joins base/v5b silver result rows by manifest index;
- aggregates by video, shabad token, and duration bucket;
- emits worst rows with target/prediction text and model deltas;
- writes a Markdown report under `diagnostics/phase2_10_silver_weak_slices.md`.

Then use the report to choose the next experiment. No new model training should
start until this audit is read.

## Result

Implemented:

- `make report-silver-weak-slices`
- `make audit-silver-source-rows`
- [`diagnostics/phase2_10_silver_weak_slices.md`](../diagnostics/phase2_10_silver_weak_slices.md)
- [`diagnostics/phase2_10_silver_source_audit.md`](../diagnostics/phase2_10_silver_source_audit.md)

Findings:

- Only 11 / 100 sampled rows are weak under `best(base, v5b) WRatio < 90`.
- Weak rows are concentrated in three videos:
  - `iQAbsSM5FO8`: 5 / 5 weak rows in the sample
  - `PYUPZn6wiR8`: 5 / 5 weak rows in the sample
  - `2d_Wy2Vb6n4`: 1 / 6 weak rows in the sample
- Source-row audit says all 11 weak rows are **silver-label risks**, not clean
  ASR failures:
  - 10 / 11: prediction matches the raw caption better than the canonical final
    text.
  - 1 / 11: heavy canonical fixes in the source metadata.
- Examples:
  - `PYUPZn6wiR8`: raw text / model prediction is `ਸਦਾ ਜਾਗੇ ਰਾਮ ਪਿਆਰੇ`; canonical
    final is `ਸੰਤ ਜਨਾ ਰਾਮ ਪਿਆਰੇ`.
  - `iQAbsSM5FO8`: raw text / model prediction is `ਮਨਿ ਪਰਚਾਇਆ`; canonical final
    is `ਧਨਿ ਰੁਚ ਇਆ`, with very low retrieval margin.

## Decision

This silver failure audit does **not** justify broad adapter scaling. The broad
ASR signal is already high, and the weakest silver rows are mostly dataset
canonicalization risk. The current path remains correct:

1. Keep `phase2_9_loop_align` as the best runtime architecture candidate
   (91.2% paired benchmark, no catastrophic case).
2. Treat silver as a diagnostic filter, not a promotion gate.
3. Continue toward gold OOS validation for promotion.
4. For model-training work, only run targeted acoustic/data experiments if a
   future audit finds clean ASR misses rather than label-risk rows.
