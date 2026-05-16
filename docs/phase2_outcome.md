# Phase 2 outcome and pivot

Phase 2 is complete as a pipeline proof, but it did **not** promote as a model improvement.

## What ran

| Step | Result |
|---|---|
| Phase 2.A smoke | Passed. 20-step MPS smoke fine-tune completed, adapter saved, `run_card.json` emitted. |
| Phase 2.B reproducibility | Passed. Two same-seed smoke runs had identical config/data hashes and `0.000%` train-loss delta. |
| Phase 2.C data pull | Passed after Phase 1.E content-holdout fix. `200` clips, `0.418 h`, `28` shabad tokens, `2` source videos. |
| Phase 2.D training | Passed. 3 epochs on MPS, `232.4 s`, final logged train loss `0.6465`, peak MPS memory `22.34 GB`. |
| Phase 2.E benchmark eval | Completed with `HF_WINDOW_SECONDS=10`. Score: `74.0%` frame accuracy (`2535/3425`). |
| Phase 2.F archive | Completed under [`submissions/v5_mac_baseline/`](../submissions/v5_mac_baseline/). |

The gate was `>= 75.0%` benchmark accuracy (`x4_pathA_surt` at `74.0%` + 1 pt). `v5_mac_baseline` scored `74.0%`, so it missed the gate by 1 point and is neutral relative to the baseline.

## Interpretation

The training loop is sound. The evidence:

- Same-seed reproducibility is bit-identical for the smoke gate.
- Real training loss moved in the right direction (`~0.77 -> ~0.65`).
- The LoRA adapter saved and loaded through the PEFT inference path.
- Evaluation used a distinct ASR cache key for the adapter.
- The run is archived with a complete `run_card.json`.

The model result is neutral. The most likely reasons are not "the loop is broken"; they are:

- The training slice is very small: `0.418 h` of audio for a 245M-parameter model that was already trained on Gurbani.
- Diversity is too low: `200` clips came from only `2` source videos.
- The paired benchmark failures are dominated by blind-ID and cold-window behavior, especially `IZOsmkdmmcg_cold33` / `IZOsmkdmmcg_cold66`.
- Nine of twelve prediction JSONs are byte-identical to `x4_pathA_surt`, so this adapter did not materially change the transcripts on most cases.

## Decision

Do **not** start Phase 3 exactly as originally written. A 50h, 3-seed training run is too expensive while the next unknown is still diagnostic: does more diverse data move the ASR, and does that movement survive an OOS check?

Insert **Phase 2.5** before Phase 3.

## Phase 2.5 — diagnostic bridge

**Role:** ML Scientist + Speech Data Engineer.

**Hypothesis:** `v5_mac_baseline` was neutral because the data slice was too small and too narrow, and because the paired benchmark is dominated by blind-ID/cold-window failure modes. A more diverse diagnostic run plus OOS v1 should tell us whether scaling Whisper-small LoRA is still the right bet.

**Approach:**

1. Curate `oos_v1` before claiming another training win.
   - Pick 5 cases using [`docs/oos_v1_curation.md`](oos_v1_curation.md): 3 representative + 2 stress.
   - Establish baseline scores for Path A v3.2, `x4_pathA_surt`, and `v5_mac_baseline`.
2. Add/verify data-pull diversity controls before the next adapter.
   - Pull beyond the first parquet shard or otherwise sample across shards.
   - Target `1k-5k` clips for `v5b_mac_diverse`.
   - Require at least `20` source videos and at least `100` shabad tokens, or document why the dataset cannot provide that diversity at the requested size.
   - Keep `min_score >= 0.85` for the diagnostic pull.
   - Keep shabad-ID, video-ID, and content-based benchmark holdouts active.
3. Train `v5b_mac_diverse` with the same proven config first.
   - Do not change LoRA rank, LR, augmentation, or model size in the first retry.
   - The only intended variable is data scale/diversity.
4. Evaluate benchmark + OOS + transcript deltas.
   - Benchmark: same `HF_WINDOW_SECONDS=10` path used for `v5_mac_baseline`.
   - OOS: report mean and per-case scores.
   - Transcript delta: compare ASR/prediction outputs to `x4_pathA_surt`. If most files are still identical, training is not affecting the inference path enough.

**Success criteria:**

- `oos_v1` exists and has baseline scores.
- `v5b_mac_diverse` either clears the benchmark gate (`>= 75.0%`) or shows positive OOS movement without a catastrophic per-case regression.
- The submission archive includes data diversity counts, run card, score table, and transcript-delta summary.

**Failure response:**

If `v5b_mac_diverse` is still neutral and mostly transcript-identical to `x4_pathA_surt`, pause Whisper-small LoRA scaling. Shift effort to integration and alignment: blind-ID robustness, chunking/windowing, word timestamps, full-shabad forced alignment, or the IndicConformer path.

## Current next step

Start Phase 2.5 with OOS v1 curation and a diversity-aware pull. The execution plan is [`docs/phase2_5_plan.md`](phase2_5_plan.md).

Implemented helpers:

1. `make fetch-oos-audio OOS_URL='case_001=https://...'` for OOS audio.
2. `make data-v5b` for a diversity-gated diagnostic pull.
