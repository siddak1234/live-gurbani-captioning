# v5b_mac_diverse - Phase 2.5 diversity diagnostic

**Date:** 2026-05-16

## Result

**Decision:** Phase 2.5 benchmark gate failed. Do not promote. Do not start Phase 3 scale-up from this adapter.

- Paired benchmark: **65.6%** frame accuracy, 2247/3425 frames, collar=1s
- Gate: **>= 75.0%** benchmark OR positive OOS movement with no catastrophic per-case regression
- Delta vs `x4_pathA_surt`: **-8.4 pt**
- Delta vs `v5_mac_baseline`: **-8.4 pt**
- OOS v1: owed, but this adapter already fails the paired benchmark gate.

The adapter learned the training slice well, but generalization to the paired benchmark regressed. This is a useful negative result: broader data changed ASR behavior, but the current Whisper-small LoRA plus Path A blind-ID/live integration is not yet a safe scaling path.

## Config

- Base model: `surindersinghssj/surt-small-v3`
- Adapter: `lora_adapters/v5b_mac_diverse`
- Data: `training_data/v5b_mac_diverse`
- Samples: 2,544 clips, 4.936 h, 195 shabad tokens, 20 source videos
- Source shards: 0, 1, 2, 3, 4
- Pull filters: `min_score >= 0.85`, duration 1-30s
- Holdout: shabad-ID, video-ID, and content-based benchmark-line filters active
- Holdout audit: 0 benchmark video leaks, 0 benchmark content leaks
- Training: 3 epochs, lr 1e-5, batch 4, grad_accum 2, LoRA r=16
- Precision/device: fp32 on MPS
- Train wall-clock: 2869.9 s
- Peak MPS memory: 27.05 GB
- Final logged train loss: 0.2486
- Run card: `run_card.json`

## Eval command

```bash
HF_WINDOW_SECONDS=10 make eval \
  TRAIN_OUT=lora_adapters/v5b_mac_diverse \
  EVAL_OUT=submissions/v5b_mac_diverse
```

## Scores

| Case | Accuracy | Frames | Pred segs | Blind ID |
|---|---:|---:|---:|---|
| IZOsmkdmmcg | 90.3% | 411/455 | 19 | 4377 correct |
| IZOsmkdmmcg_cold33 | 10.1% | 31/308 | 17 | 3712 wrong |
| IZOsmkdmmcg_cold66 | 16.7% | 26/156 | 9 | 3712 wrong |
| kZhIA8P6xWI | 4.6% | 14/303 | 19 | 3712 wrong |
| kZhIA8P6xWI_cold33 | 80.7% | 167/207 | 14 | 1821 correct |
| kZhIA8P6xWI_cold66 | 59.0% | 62/105 | 8 | 1821 correct |
| kchMJPK9Axs | 93.2% | 605/649 | 18 | 1341 correct |
| kchMJPK9Axs_cold33 | 94.3% | 413/438 | 14 | 1341 correct |
| kchMJPK9Axs_cold66 | 96.4% | 214/222 | 6 | 1341 correct |
| zOtIpxMT9hU | 78.4% | 225/287 | 12 | 3712 correct |
| zOtIpxMT9hU_cold33 | 12.2% | 24/196 | 5 | 1341 wrong |
| zOtIpxMT9hU_cold66 | 55.6% | 55/99 | 7 | 3712 correct |

Overall: **65.6%** (2247/3425).

## Diagnostics

- The adapter path is active: only 2/12 benchmark prediction JSONs are byte-identical to `x4_pathA_surt`, and 0/12 are byte-identical to `v5_mac_baseline`.
- Training loss moved strongly: early logs around `1.0`, final logged loss `0.2486`, aggregate train loss `0.3503`.
- Regression is concentrated in blind-ID-sensitive cases:
  - `kZhIA8P6xWI` full: `78.5% -> 4.6%` vs `v5_mac_baseline`, because blind ID flipped from 1821 to 3712.
  - `zOtIpxMT9hU_cold33`: `68.4% -> 12.2%`, because blind ID flipped from 3712 to 1341.
  - `IZOsmkdmmcg_cold*` stayed bad; same wrong 3712 blind ID as the v5 baseline.
- Some in-shabad alignment improved:
  - `kchMJPK9Axs*` improved from roughly 92-93% to 93-96%.
  - `kZhIA8P6xWI_cold33` improved from 73.4% to 80.7%.
  These gains are real but overwhelmed by wrong-shabad failures.

## Interpretation

The training loop and data hygiene are not the blocker. The model can learn from the larger slice, and the eval path applies the adapter. The blocker is integration robustness: adapter-induced transcript changes perturb blind shabad ID and cold-window behavior enough to erase any ASR gains.

This argues against Phase 3 as originally imagined (50h, LoRA r=32, 3 seeds) as the immediate next step. Scaling the same setup would spend much more compute before the failure mode is understood.

## Recommended next action

Do not tune matcher weights on this benchmark as a quick fix. That would overfit the 12-case paired benchmark.

Run a narrow Phase 2.6 alignment diagnostic instead:

1. Compare blind-ID evidence between `x4_pathA_surt`, `v5_mac_baseline`, and `v5b_mac_diverse` for the four wrong-ID cases.
2. Separate acoustic transcript quality from shabad-ID routing:
   - score with oracle shabad IDs, or
   - run the matcher against the ground-truth shabad only, or
   - log top-k shabad evidence over time for cold windows.
3. Test timestamp/window integration without retraining:
   - shorter HF windows,
   - word-level timestamps if available,
   - surt text with faster-whisper timestamps,
   - or a two-pass ID scheme that uses a conservative canonical engine for blind ID and the adapter only after ID lock.
4. Curate OOS v1 before any model-improvement claim.

If Phase 2.6 shows the adapter improves oracle-shabad alignment but hurts blind ID, keep Whisper-small LoRA and fix integration. If oracle-shabad alignment is also neutral or worse, pause Whisper-small LoRA and evaluate `surt-medium` or IndicConformer.
