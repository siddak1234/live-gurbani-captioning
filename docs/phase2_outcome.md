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

Phase 2.10 adds an automated silver bridge so progress does not depend solely
on manual OOS labeling. The preferred online timestamped dataset
(`surindersinghssj/gurbani-kirtan-dataset-v2`) is visible in the browser but
currently returns 401 from HF APIs, so the executable fallback uses the already
accessible 300h canonical dataset on never-trained shards `10-19`.

The fallback silver pull produced 8,306 clips / 16.70 h / 19 videos / 308
shabad tokens, with **0 video overlap** versus `v5b_mac_diverse`. On a
100-segment round-robin sample, `surt-small-v3` base scored mean WRatio
**96.29** / exact normalized **75.0%**, and `surt-small-v3 + v5b_mac_diverse`
scored mean WRatio **96.33** / exact normalized **73.0%**. This is effectively
neutral for raw segment ASR, reinforcing the Phase 2.6-2.9 diagnosis: the lift
is coming from runtime ID-lock / buffering / loop-aware alignment, not from a
large ASR adapter gain. Do not scale adapter training as the next move without
a more targeted acoustic failure diagnosis.

Implemented helpers:

1. `make fetch-oos-audio OOS_URL='case_001=https://...'` for OOS audio.
2. `make data-v5b` for a diversity-gated diagnostic pull.
3. `make data-silver-300h` and `make eval-silver-300h` for automated silver
   ASR/canonical-text diagnostics on never-trained 300h shards.

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

Current next step: run `make oos-review-pack`, `make prepare-oos-review`, and
`make audit-oos-assist`. The second target seeds `eval_data/oos_v1/test/case_001.json`
through `case_005.json` from the machine drafts but marks them
`NEEDS_HUMAN_CORRECTION`; the assisted audit then cross-checks those working
files against cached BaniDB corpus text, local Whisper ASR chunks, and any
available YouTube caption tracks. Validation will still fail until the files are
corrected against the fetched audio and promoted to
`curation_status: HUMAN_CORRECTED_V1`. Do **not** claim an OOS score until those
GT files are corrected, validated, and committed. The step-by-step labeling
checkpoint is [`docs/oos_v1_labeling_checkpoint.md`](oos_v1_labeling_checkpoint.md),
the first machine triage is [`docs/oos_v1_machine_audit.md`](oos_v1_machine_audit.md),
and the repeatable assisted cross-check is `make audit-oos-assist`. Once the GT
files pass validation, run:

```bash
make eval-oos-loop-align
```

To keep engineering moving without pretending machine labels are gold, there is
also a silver diagnostic path:

```bash
make prepare-oos-assisted
make eval-oos-loop-align-assisted
```

That path writes generated labels to `eval_data/oos_v1/assisted_test/` with
`curation_status=MACHINE_ASSISTED_V1_NOT_GOLD` and scores the current
`phase2_9_loop_align` runtime against them. Use the result to decide whether
gold review is likely worth finishing and whether Phase 3 scale-up is plausible;
do not report it as the final OOS score.

Assisted-OOS diagnostic result, 2026-05-17:

- `make eval-oos-loop-align-assisted` completed.
- Overall: **29.5%** frame accuracy (`260/880`) against machine-assisted labels.
- Shabad locks: **2/5** correct.
- Correct locks: case_002 (`906`) and case_003 (`2600`).
- Wrong locks: case_001 (`2333 -> 1821`), case_004 (`4892 -> 906`), case_005
  (`3297 -> 906`).

This is a diagnostic failure, not a model-training failure. The correct OOS
shabad corpora are present in `corpus_cache/`, so the failure is the blind
shabad-lock policy under a broader candidate set. cases 004/005 had zero lock
evidence at the fixed 30s commit point (`top=0.0`, `runner_up=0.0`) and still
committed to an arbitrary candidate. case_001 shows shared-hook ambiguity: the
current `chunk_vote` lock ties or confuses similar high-scoring lines across
candidate shabads.

Decision: do **not** start Phase 3 / all-300h training yet. The next phase is
[`docs/phase2_11_shabad_lock_plan.md`](phase2_11_shabad_lock_plan.md): build a
repeatable shabad-lock audit, add a no-zero-evidence commit rule, and prototype
a confidence-gated delayed lock policy. This keeps the architecture honest:
high paired-benchmark alignment is not useful if the runtime commits the wrong
shabad outside the benchmark.

Phase 2.11 first prototype result:

- Delayed zero-evidence lock guard (`--lock-lookbacks 30,45,60,90
  --min-lock-score 1`) improves assisted OOS from **29.5%** to **40.5%** and
  locks from **2/5** to **3/5**.
- This fixes the all-zero early commit for case_005, but case_004 still locks
  wrong at 45s and case_001 still has shared-hook ambiguity.
- Oracle-shabad scoring against the same machine-assisted labels reaches only
  **51.0%**, proving the assisted labels are diagnostic, not promotion-grade.

Conclusion: the next true blocker is robust shabad-lock/candidate retrieval plus
gold-quality OOS labels. More broad 300h training is still not the next expert
move.

Phase 2.12 answers the "can we continue learning without hand validation?"
question. Yes: continue with a silver learning loop, but do not call it final
accuracy. The plan is [`docs/phase2_12_silver_learning_plan.md`](phase2_12_silver_learning_plan.md):
use the paired benchmark, assisted OOS labels, and cached ASR/corpus evidence to
tune or learn a shabad-lock policy. Treat the resulting score as a development
signal only.

Executed Phase 2.12.A:

- `make tune-shabad-lock-policy`
- Report: `diagnostics/phase2_12_silver_lock_policy.md`
- Best macro policy: `chunk_vote@45s|min=0`
- Result: **67.5%** silver macro lock accuracy (**9/12** paired, **3/5**
  assisted OOS)
- Best OOS-only policy (`tfidf_then_topk3@45s|min=0`) reaches **5/5** assisted
  OOS locks but collapses paired to **3/12**

Decision: silver labels let us continue learning, but this tuning run rejects a
simple scorer/window switch as the next architecture. The next targeted step is
candidate retrieval / lock-evidence modeling. Broad 300h training remains
deferred because the current blocker is wrong shabad commitment, not raw ASR
capacity.

Phase 2.13 implemented that candidate-evidence step:

- Plan: [`docs/phase2_13_candidate_retrieval_plan.md`](phase2_13_candidate_retrieval_plan.md)
- Report: `diagnostics/phase2_13_lock_evidence_fusion.md`
- Best fusion: `fusion:tfidf_60+0.5*chunk_vote_90`
- Silver lock diagnostic: **9/12** paired, **5/5** assisted OOS, **87.5%**
  macro lock objective
- Full assisted-OOS frame diagnostic: **59.9%** with **5/5** locks
- Full paired diagnostic: **79.7%**, with failures concentrated in
  `zOtIpxMT9hU` / `zOtIpxMT9hU_cold33` locking to shabad `4892`

Decision: Phase 2.13 confirms evidence fusion is the right learning direction
and materially improves OOS lock behavior, but it is not a promotion candidate.
The remaining architecture problem is high-confidence false-candidate
disambiguation, especially `3712` vs `4892`, not M4 Pro utilization or lack of
300h training data.

Expert checkpoint on "more shabad variance": yes, the current paired benchmark
is too small to judge production readiness, but the immediate fix is **held-out
OOS validation**, not training on these five. The training path already has
large-scale shabad variance through the HuggingFace pulls and Phase 3/4 plans.
At this checkpoint, these five shabads must stay outside training so they can
tell us whether the 91.2% paired-benchmark architecture generalizes. If OOS v1
passes, then scale training/evaluation; if it fails, diagnose by slice before
spending the M4 Pro budget on Phase 3.

## Phase 2.10 — automated silver OOS bridge

The user correctly challenged the manual labeling bottleneck: public labeled
data should be used before asking for more human work. The updated plan is
[`docs/phase2_10_silver_oos_plan.md`](phase2_10_silver_oos_plan.md).

Decision:

- Keep `oos_v1` as the small **gold** OOS promotion gate.
- Add `silver_kirtan_v2` as an automated **silver** OOS diagnostic from online
  timestamped line-level data.
- Use silver to compare ASR/canonical-text behavior across `surt-small-v3`,
  `v5b_mac_diverse`, and baselines.
- Do not treat silver as production promotion unless it is reconstructed into
  full-shabad benchmark-shaped cases with verified canonical IDs.

Current access note: the Hugging Face page for
`surindersinghssj/gurbani-kirtan-dataset-v2` is visible, but the local HF API
client returned 401 on 2026-05-16. Resolve access first; if blocked, construct a
held-out silver split from the existing 300h canonical dataset.
