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
2. Add/verify data-pull diversity controls before the next adapter. ✅ done
   - Pull beyond the first parquet shard or otherwise sample across shards.
   - Target `1k-5k` clips for `v5b_mac_diverse`.
   - Actual pull: `2,544` clips, `4.936 h`, `20` source videos, `195` shabad tokens, shards `0-4`.
   - Keep `min_score >= 0.85` for the diagnostic pull.
   - Keep shabad-ID, video-ID, and content-based benchmark holdouts active. Actual audit: `0` benchmark video leaks, `0` benchmark content leaks.
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

The diversity-aware pull passed, `v5b_mac_diverse` trained, and benchmark eval scored **65.6%**. This is below `x4_pathA_surt` / `v5_mac_baseline` at **74.0%**, so Phase 2.5 is a negative diagnostic. Phase 2.6 then showed the adapter is not globally worse: oracle-shabad/live0 alignment scores **87.4%** for `v5b`, compared with **85.2%** for x4/v5, and a historical v3.2-ID-lock proxy scores **87.1%**.

Phase 2.7 implemented that real runtime ID-lock integration as `src/idlock_engine.py` + `scripts/run_idlock_path.py`, but it scored only **75.6%**. The failure is localized to runtime blind-ID: two `kZhIA8P6xWI` starts commit to the wrong shabad. A second checkpoint also found that the archived v3.2 result is not currently reproducible: the documented command now scores **73.5%**, not **86.5%**.

Do not start Phase 3 scale-up. The next step is Phase 2.8: recover/pin ASR reproducibility and prototype timestamp/alignment fixes before any larger LoRA run. The execution plan is [`docs/phase2_8_plan.md`](phase2_8_plan.md).

Phase 2.8 sharpened the path: word timestamps fix shabad ID (12/12 locks) but
hurt full-run timing (`phase2_8_fw_word`: **72.0%**); VAD-on is catastrophic
(`phase2_8_fw_vad`: **25.4%**); using word timestamps only for the pre-lock ID
window and v5b after lock gives the best current runtime
(`phase2_8_idlock_preword`: **86.6%**). Two post-lock smoother probes then
regressed (`phase2_8_idlock_preword_viterbi`: **77.2%**,
`phase2_8_idlock_preword_viterbi_null45`: **77.1%**), so the remaining lift is
not another per-chunk smoother.

The next step is Phase 2.9: full-shabad alignment with explicit loop/refrain
handling, using the current best ID-lock path only to decide the shabad.

Phase 2.9's first policy probe is positive: `phase2_9_retro_buffered` keeps the
same models but lets the locked-shabad post engine revise the buffered pre-lock
window after ID commit. It scores **88.7%**, up from **86.6%**, without a route
table. This is the best current honest runtime architecture, but not promoted:
`zOtIpxMT9hU_cold66` remains below the catastrophic-case guardrail at **57.6%**
and OOS v1 is still owed.

Phase 2.9's first loop/null-aware aligner then validates the diagnosis:
`phase2_9_loop_align` adds one generic null rule for chunks dominated by
repeated simran (`ਵਾਹਿਗੁਰੂ` / `waheguru`) and keeps the retro-buffered ID-lock
stack unchanged. It scores **91.2%** with 12/12 locks and no catastrophic case;
`zOtIpxMT9hU_cold66` rises from **57.6%** to **86.9%**. This is the best current
honest runtime architecture and the first non-route-table runtime candidate
above 90%.

Protocol decision: still do **not** start Phase 3. The paired benchmark now
says the architecture is promising, but the production-goal gate is OOS v1.
The next step is to curate the 5-case OOS pack and score `phase2_9_loop_align`
outside the paired benchmark before making any promotion or training-scale
claim.

Implemented helpers:

1. `make fetch-oos-audio OOS_URL='case_001=https://...'` for OOS audio.
2. `make data-v5b` for a diversity-gated diagnostic pull.

Checkpoint 2026-05-16:

- OOS v1 now has a concrete 5-case candidate slate in
  [`docs/oos_v1_curation.md`](oos_v1_curation.md): 3 representative SGGS
  shabads + 2 stress cases.
- `scripts/build_corpus.py --shabad-id ...` and `make corpus-oos` support
  explicit OOS shabad caching; the selected OOS corpora have been fetched
  locally.
- `scripts/fetch_audio.py --clip case=START-END` and `make fetch-oos-audio
  OOS_CLIP=...` support bounded OOS clips; the five source clips have been
  downloaded locally under `eval_data/oos_v1/audio/` (gitignored).
- `configs/datasets.yaml` holdout video IDs now include the five OOS source
  YouTube IDs so future training pulls cannot sample these eval recordings.

Current next step: hand-curate `eval_data/oos_v1/test/case_001.json` through
`case_005.json` against the fetched audio, set `curation_status:
HUMAN_CORRECTED_V1`, and run `make validate-oos-gt`. Do **not** claim an OOS
score until those GT files are corrected, validated, and committed. Once they
exist, run:

```bash
make eval-oos-loop-align
```

Expert checkpoint on "more shabad variance": yes, the current paired benchmark
is too small to judge production readiness, but the immediate fix is **held-out
OOS validation**, not training on these five. The training path already has
large-scale shabad variance through the HuggingFace pulls and Phase 3/4 plans.
At this checkpoint, these five shabads must stay outside training so they can
tell us whether the 91.2% paired-benchmark architecture generalizes. If OOS v1
passes, then scale training/evaluation; if it fails, diagnose by slice before
spending the M4 Pro budget on Phase 3.
