# Phase 2.5 — diagnostic bridge execution plan

Phase 2.5 exists because `v5_mac_baseline` proved the pipeline works but did not prove the model improved. The next run must answer a narrower scientific question:

> Does a materially more diverse kirtan slice make the `surt-small-v3` LoRA adapter change transcripts and generalize, or is Whisper-small LoRA the wrong next scaling bet?

## Checkpoint analysis

| Area | Current evidence | Expert read |
|---|---|---|
| Training loop | Smoke passed, 3-epoch real run completed, loss moved `~0.77 -> ~0.65`. | Loop is sound enough to scale one diagnostic step. |
| Reproducibility | Same-seed smoke delta `0.000%`; `run_card.json` hashes stable. | Instrumentation is strong; failures are interpretable. |
| Data hygiene | Shabad-ID, video-ID, and content holdouts active. | Contamination risk is controlled for benchmark lines. |
| Data diversity | `v5_mac_baseline`: 200 clips, 0.418h, 28 shabad tokens, **2 videos**. `v5b_mac_diverse` pull now has 2,544 clips, 4.936h, 195 shabad tokens, 20 videos. | The narrow-data flaw is fixed enough for one diagnostic training run. |
| Benchmark result | `74.0%`, equal to `x4_pathA_surt`; 9/12 JSONs byte-identical. | Adapter did not materially affect most inference outputs. |
| OOS status | OOS v1 scoped; audio helper exists; cases not curated. | No model-improvement claim should promote without OOS. |
| Architecture | Inference stays layered through `engine.predict()`; data/training are separate. | Clean enough. Next work belongs in data tooling, not inference primitives. |

## Current checkpoint — 2026-05-16

The diversity-gated pull has passed and is ready for training:

| Field | `v5b_mac_diverse` |
|---|---:|
| Clips | 2,544 |
| Hours | 4.936 |
| Unique source videos | 20 |
| Unique shabad tokens | 195 |
| Source shards used | 0, 1, 2, 3, 4 |
| `min_score` | 0.85 |
| Benchmark video leaks | 0 |
| Benchmark content leaks | 0 |
| Diversity gate | PASS |

The pull intentionally exceeded the 1,000-clip minimum because diversity floors were active. This is the correct behavior: `DATA_SAMPLES` is a floor, not a cap, when `min_unique_*` constraints are set.

**Recommended next step:** train `v5b_mac_diverse` with the same known-good LoRA config. Do not change LoRA rank, LR, windowing, matcher weights, or architecture until this diagnostic run is scored.

## Architecture stance

Keep the architecture boring:

- Data-selection policy lives in `scripts/pull_dataset.py` and `data_card.md`.
- Training configuration lives in `configs/training/surt_lora_mac.yaml`.
- Experiment results live under `submissions/<run>/`.
- Inference primitives (`src/asr.py`, `src/matcher.py`, `src/smoother.py`, `src/engine.py`) stay untouched until a scored diagnostic proves the model path is worth changing.

This avoids mixing three variables at once. Phase 2.5 changes **data diversity first**, not model capacity, not LoRA rank, not matcher tuning.

## New data-pull guardrails

The puller now supports:

```bash
python scripts/pull_dataset.py kirtan \
  --out-dir training_data/v5b_mac_diverse \
  --num-samples 1000 \
  --min-score 0.85 \
  --shards 0-9 \
  --max-scan 20000 \
  --min-unique-videos 20 \
  --min-unique-shabads 100
```

or the Makefile shortcut:

```bash
make data-v5b
```

When diversity floors are active, `--num-samples` is a minimum, not a hard cap: the pull continues until both the sample target and diversity floors pass, or until `--max-scan` is exhausted. The manifest is written even when a floor fails, and the command exits non-zero. That is intentional: failed pulls are diagnostic artifacts. Inspect `training_data/v5b_mac_diverse/data_card.md`, then widen shards / scan budget or lower a floor only with an explicit note.

## Execution order

1. **Curate OOS v1.** *(still owed before promotion)*
   - Pick 5 recordings: 3 representative + 2 stress.
   - Fetch audio with `make fetch-oos-audio OOS_URL='case_001=https://...'`.
   - Hand-correct GT JSONs under `eval_data/oos_v1/test/`.
   - Baseline Path A v3.2, `x4_pathA_surt`, and `v5_mac_baseline`.
2. **Pull `v5b_mac_diverse`.** *(done; passed diversity and holdout audit)*
   - Run `make data-v5b`.
   - Required floors: `>=20` source videos and `>=100` shabad tokens.
   - If floors fail, do not train; widen the pull first.
3. **Train with the same config.** *(current step)*
   ```bash
   make train \
     DATA_DIR=training_data/v5b_mac_diverse \
     TRAIN_OUT=lora_adapters/v5b_mac_diverse
   ```
4. **Evaluate benchmark with the known-good window.**
   ```bash
   HF_WINDOW_SECONDS=10 make eval \
     TRAIN_OUT=lora_adapters/v5b_mac_diverse \
     EVAL_OUT=submissions/v5b_mac_diverse
   ```
5. **Evaluate OOS v1.**
   ```bash
   make eval-oos TRAIN_OUT=lora_adapters/v5b_mac_diverse
   ```
6. **Compare transcript deltas.**
   - If most benchmark prediction files remain byte-identical to `x4_pathA_surt`, the adapter is not influencing inference enough.
   - If transcripts change but scores do not, inspect matcher/blind-ID behavior before scaling.

## Promotion gate

`v5b_mac_diverse` promotes only if:

- OOS v1 has baseline scores.
- Benchmark is `>=75.0%` **or** OOS shows positive movement vs x4/v5 without a catastrophic per-case regression.
- Submission notes include data diversity counts, run card, score table, and transcript-delta summary.

If this gate fails, pause Phase 3. The next bet becomes alignment/integration: blind-ID robustness, chunking/windowing, word timestamps, full-shabad forced alignment, or IndicConformer.
