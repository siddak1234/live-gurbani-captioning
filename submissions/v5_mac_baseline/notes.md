# v5_mac_baseline - Phase 2 first real Mac fine-tune

**Date:** 2026-05-16

## Result

**Decision:** Phase 2 benchmark gate did not pass.

- Paired benchmark: **74.0%** frame accuracy, 2535/3425 frames, collar=1s
- Gate: **>= 75.0%** (x4_pathA_surt 74.0% + 1 pt)
- Delta vs x4_pathA_surt: **+0.0 pt**
- OOS v1: owed. The OOS pack is scoped in `docs/oos_v1_curation.md` but not curated yet.

The pipeline is validated end to end, but this 200-sample LoRA does not move the benchmark.

## Config

- Base model: `surindersinghssj/surt-small-v3`
- Adapter: `lora_adapters/v5_mac_baseline`
- Data: `training_data/v5_mac_baseline`
- Samples: 200 clips, 0.418 h, 28 shabad tokens, 2 source videos
- Holdout: shabad-ID, video-ID, and content-based benchmark-line filters active
- Training: 3 epochs, lr 1e-5, batch 4, grad_accum 2, LoRA r=16
- Precision/device: fp32 on MPS
- Train wall-clock: 232.4 s
- Peak MPS memory: 22.34 GB
- Final logged train loss: 0.6465
- Run card: `run_card.json`

## Eval command

The fair comparison to `x4_pathA_surt` requires 10-second HF windows. A 30-second window reproduces the known-bad X4 regime.

```bash
HF_WINDOW_SECONDS=10 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 make eval \
  TRAIN_OUT=lora_adapters/v5_mac_baseline \
  EVAL_OUT=submissions/v5_mac_baseline
```

## Scores

| Case | Accuracy | Frames | Pred segs |
|---|---:|---:|---:|
| IZOsmkdmmcg | 90.3% | 411/455 | 19 |
| IZOsmkdmmcg_cold33 | 10.1% | 31/308 | 15 |
| IZOsmkdmmcg_cold66 | 16.7% | 26/156 | 8 |
| kZhIA8P6xWI | 78.5% | 238/303 | 19 |
| kZhIA8P6xWI_cold33 | 73.4% | 152/207 | 14 |
| kZhIA8P6xWI_cold66 | 64.8% | 68/105 | 8 |
| kchMJPK9Axs | 92.0% | 597/649 | 19 |
| kchMJPK9Axs_cold33 | 92.5% | 405/438 | 15 |
| kchMJPK9Axs_cold66 | 92.8% | 206/222 | 7 |
| zOtIpxMT9hU | 79.8% | 229/287 | 12 |
| zOtIpxMT9hU_cold33 | 68.4% | 134/196 | 10 |
| zOtIpxMT9hU_cold66 | 38.4% | 38/99 | 6 |

Overall: **74.0%** (2535/3425).

## Diagnostics

- The adapter path is active: PEFT loads the saved LoRA adapter and uses a distinct ASR cache key with `_lora-v5_mac_baseline`.
- Predictions are score-equivalent to `x4_pathA_surt`; 9/12 JSON files are byte-for-byte identical to the baseline submission. The three differing files are the `IZOsmkdmmcg*` variants and do not change the overall frame count.
- The failed 30-second-window run scored 46.2%, matching the known X4 observation that 30s windows are too coarse for this backend. The retained score is the 10s-window run.
- Blind ID remains the main visible failure mode: `IZOsmkdmmcg_cold33` and `IZOsmkdmmcg_cold66` still predict shabad 3712 instead of 4377.

## Interpretation

The training loop is sound: smoke, same-seed reproducibility, real 3-epoch training, adapter save/load, and benchmark scoring all work. The specific 200-clip adapter is neutral on the benchmark.

Most likely explanations:

- 200 clips / 0.42 h is too small to move a 245M-parameter Whisper-small model that was already trained on Gurbani audio.
- Only 2 source videos means the fine-tune mostly adapts to recording quirks, not broad kirtan generalization.
- The paired benchmark failures are dominated by blind-ID/cold-window behavior, not raw ASR loss.

## Next action

Do not promote this run as a model improvement. Use it as an end-to-end pipeline proof and move to one of:

1. Pull a larger, more diverse slice (1000-5000 clips) and rerun the same config.
2. Curate OOS v1 first so the next adapter gets an independent generalization check.
3. Investigate blind-ID robustness for cold variants before scaling data.
