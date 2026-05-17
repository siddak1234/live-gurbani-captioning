# CLAUDE.md

Orientation for Claude Code sessions in this repo. Read the paired benchmark for deeper detail; this file is an index, not a duplicate.

## Two machines, two roles

If you're a Claude session on the **training machine** (a dedicated Apple Silicon Mac that runs fine-tunes), your job is narrow: `make start`. That runs Python check → `pip install -r requirements-mac.txt` → auto-pulls `surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical` from HuggingFace → LoRA-fine-tunes `surt-small-v3`. No ffmpeg, no paired benchmark repo, no smoke test. Adapter lands in `lora_adapters/surt_mac_v1/`; ship it back to the dev machine.

If you're a Claude session on the **dev machine** (where code review, benchmark scoring, OOS evaluation, and iOS export happen), your job is wider. Run `make start-dev` for the kitchen-sink chain (doctor-dev + install + fetch-audio + corpus + data + smoke + train), or pick targets individually: `make eval`, `make eval-oos`, `make ios-export`. Requires `ffmpeg` (`brew install ffmpeg`) and the paired benchmark repo cloned at `../live-gurbani-captioning-benchmark-v1/`. iOS deployment goes through Argmax's `whisperkittools` (HF → Core ML w/ ANE quantization) + the [WhisperKit](https://github.com/argmaxinc/WhisperKit) Swift package; see [`docs/ios_deployment.md`](docs/ios_deployment.md) and the `ios/` directory.

Run `make help` on either machine to see every target and its one-line purpose.

## Project goal

Build a **real, robust, scalable kirtan captioning engine** that works at user level on arbitrary recordings — not just the 4 shabads in the paired benchmark. The benchmark is a *measurement tool*, not the deliverable. Solutions that overfit to its 12 cases are leaderboard exercises, not deployable systems.

Two parallel deliverables:
1. **Honest benchmark scores** (we report them, *and* mark which are overfit vs generalizable).
2. **A generalizable engine** (single-model pipeline trained on broad data; performs well on novel recordings the benchmark hasn't seen).

This is not a stop-and-ship project. The solution is far from over — every committed submission is a step, not a finish line. Future sessions should keep pushing toward (2) even when (1)'s numbers look good.

## What this repo is

An engine that consumes kirtan audio and emits line-level predictions of which shabad and line is being sung at each moment. The paired benchmark provides a 12-case test set for measurement. We do **not** ship the benchmark, the ground truth, or the scorer — we produce submissions, but the actual production deliverable is the engine itself.

## Paired benchmark

`~/Desktop/coding_projects/live-gurbani-captioning-benchmark-v1/`

That repo is the source of truth for task definition, scoring rules, and submission format. Read its `README.md`, `eval.py`, and `test/*.json` files directly when you need detail — do not guess from this CLAUDE.md.

## Submission format (minimum to know)

One JSON per GT case under `<benchmark>/test/`, same filename stem, placed in a single output directory:

```json
{
  "video_id": "IZOsmkdmmcg",
  "segments": [
    { "start": 30.0, "end": 47.0, "line_idx": 1 },
    ...
  ]
}
```

- `start < end`, both in seconds relative to the start of the audio.
- `line_idx` is **0-indexed within the predicted shabad** (not the GT shabad).
- Segments may overlap; later segments overwrite earlier ones per-frame.
- Unsegmented regions = `null` prediction (accepted in GT gaps, wrong inside GT segment interiors).
- Predictions outside UEM are ignored.

A pred segment may optionally also carry `shabad_id`, `verse_id`, or `banidb_gurmukhi`. The scorer's `_resolve_pred_label` ([eval.py:91](../live-gurbani-captioning-benchmark-v1/eval.py#L91)) resolves preds against GT in this order, first hit wins:
  1. `(shabad_id, line_idx)` — both sides agree on numbering
  2. `verse_id` — canonical BaniDB pangti id
  3. `banidb_gurmukhi` — verbatim spaced unicode pangti text

Anything that doesn't resolve gets `NO_MATCH` and is scored wrong. Carry `verse_id` and/or `banidb_gurmukhi` when your engine's `line_idx` numbering might disagree with GT's.

## How to score

```bash
python ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/<run_name>/ \
  --gt   ../live-gurbani-captioning-benchmark-v1/test/
```

Default `--collar 1`. Primary metric is **frame accuracy** at 1s resolution. Empty submissions floor at ~26%; perfect copy of GT scores 100%.

## Task variants

Four modes, same submission schema:

|             | **Blind** (identify shabad from audio) | **Oracle** (given `shabad_id` upfront) |
|-------------|----------------------------------------|----------------------------------------|
| **Offline** | full audio available                   | full audio + GT shabad                 |
| **Live**    | causal: predictions at `t` only depend on audio ≤ `t` | causal + GT shabad                     |

Live causality is honor-system — the scorer can't tell. The output JSON looks identical in all four modes.

## Core design principle

**Snap to canonical.** The engine never displays raw ASR text. Output is always the canonical Gurmukhi line looked up by `(shabad_id, line_idx)` from BaniDB. Misspelled Gurmukhi in a religious context is unacceptable; the integer-id submission format makes that constraint structural rather than aspirational.

## Project stages

- **Stage 0** ✅ — empty-submission baseline. 26.0%. `submissions/v0_empty/`.
- **Stage 1** ✅ — audio download + BaniDB corpus cache. `scripts/fetch_audio.py`, `scripts/build_corpus.py`.
- **Stage 2** ✅ — Path A engine (Whisper + rapidfuzz + smoother). Iterated v1.0 → v1.5; **88.2%** oracle+offline with `0.5*token_sort_ratio + 0.5*WRatio` blend + stay-bias=6.
- **Stage 3** ✅ — blind shabad ID via per-chunk voting. **88.2%** blind+offline (zero drop from oracle, 12/12 IDs). `submissions/v2_pathA_blind/`.
- **Stage 4** ✅ — live causal mode + tentative emission during ID buffer. **86.0%** blind+live (strictly causal). `submissions/v3_1_pathA_live_tentative/`.
- **Stage 5** ⏳ — close the gap to 95%+. Path A's matcher/smoother are tapped out around 86-88%. Cheap probes (model size, mlx-whisper, ratio sweeps, TF-IDF) all fail to push past. Path A is now **frozen** at v3.2 = 86.5% live blind; Path B (CTC phoneme scoring + loop-aware HMM, in `src/path_b/`) is the principled next move with a realistic 95%+ ceiling.

### Leaderboard (with generalization annotations)

**Read this column carefully.** Benchmark score ≠ deployable accuracy. Submissions marked *overfit* tune to specific shabads in the 12-case test set and would degrade on novel recordings.

| Submission | Mode | Score | Generalizes? |
|---|---|---|---|
| `v0_empty` | — | 26.0% | n/a (baseline floor) |
| `v3_2_pathA_no_title` | blind + live | **86.5%** | **Historical frozen artifact.** Generic fw-medium ASR, but current-runtime repro (`v3_2_repro_current`) scores **73.5%**. Phase 2.8 must recover/pin this drift before calling it production-reproducible. |
| `v4_mlx_large_v3` | blind + live | 83.8% | Yes — alternative ASR backend (Apple GPU). |
| `x4_pathA_surt` | blind + live | 74.0% | **Yes — best designed for generalization.** Uses surindersinghssj/surt-small-v3 (Whisper-small fine-tuned on 660h of kirtan). Lower benchmark score is real, but reflects honest accuracy on out-of-set kirtan. |
| `v5_mac_baseline` | blind + live | 74.0% | **Yes, but neutral.** First real Mac LoRA on `surt-small-v3` using 200 clips / 0.418h. Validates the MPS training + PEFT inference pipeline, but does not beat `x4_pathA_surt`; do not promote as a model improvement. |
| `v5b_mac_diverse` | blind + live | 65.6% | **No — negative diagnostic.** 2,544 clips / 4.936h / 20 videos / 195 shabad tokens. Adapter changes transcripts but destabilizes blind-ID/cold-window behavior; pause Phase 3 and run alignment diagnostics. |
| `v5b_twopass_v32_idlock` | two-pass proxy | 87.1% | **Diagnostic only.** Uses v3.2 pre-lock segments + v5b post-lock oracle-shabad alignment. Validates the ID-lock direction, but it is not yet a runtime engine or OOS-validated. |
| `v5b_idlock_runtime` | runtime ID-lock | 75.6% | **No — Phase 2.7 failed.** Clean Layer 2 implementation, but current blind-ID commits two `kZhIA8P6xWI` starts to the wrong shabad. |
| `phase2_8_idlock_preword` | runtime ID-lock + word-timestamp pre-lock | 86.6% | **Best current runtime candidate, not promoted.** Word timestamps fix shabad ID (12/12), v5b handles post-lock alignment, but it misses the 87% gate by 0.4 pts and OOS is owed. |
| `phase2_8_idlock_preword_viterbi` | runtime ID-lock + Viterbi smoother | 77.2% | **No — negative diagnostic.** Generic line-distance smoothing helps `IZOsmkdmmcg` but collapses refrain/loop-heavy cases. |
| `phase2_8_idlock_preword_viterbi_null45` | runtime ID-lock + Viterbi null state | 77.1% | **No — negative diagnostic.** Null state drops useful weak evidence along with filler. |
| `phase2_9_retro_buffered` | runtime ID-lock + retro-buffered finalization | **88.7%** | **Best current honest runtime, not promoted.** State-based merge policy lets locked-shabad post engine revise tentative pre-lock captions. Clears paired score gate, but `zOtIpxMT9hU_cold66` is still 57.6% and OOS is owed. |
| `phase2_9_loop_align` | runtime ID-lock + retro-buffered + simran/null alignment | **91.2%** | **Best current honest runtime, not promoted.** Non-route-table architecture; 12/12 locks; no catastrophic case (`zOtIpxMT9hU_cold66` 86.9%). OOS v1 is mandatory before production promotion. |
| `x7_surt_only` | blind + live | 68.6% | Yes — surt with longer blind buffer; didn't help (kept for negative-result record). |
| `x8_pb_finetuned` | blind + offline | 72.9% (Path B) | **Yes — proof of production training path.** w2v-bert-punjabi + LoRA adapter from 50-step fine-tune on 30 real kirtan clips. +2.6 over Path B baseline; +6 to +14 per-shabad on 3 of 4 shabads. Validates the end-to-end training pipeline; tiny scale, far from saturated. |
| `x5_ensemble` | blind + live | 91.2% | **No — benchmark-overfit.** Route table `{1341 → surt}` chosen from test-set scores. |
| **`x6_ensemble`** | **blind + live** | **92.8%** | **No — most overfit.** 3-way routing `{1341 → surt, 1821 → mlx, else → v3.2}` is empirically picked per test-shabad. Would degrade on shabads not in {4377, 1821, 1341, 3712}. |

**What benchmark-overfit means concretely:** if a Sewadar plays a new kirtan recording of, say, shabad 5621, our X5/X6 ensembles route it to whichever engine the *default* path picks — they don't have a special rule for that shabad. The hard-coded route table is calibrated for the 4 test shabads. Real-world accuracy on out-of-set shabads is closer to x4_surt's 74% than x6's 92.8%.

What didn't work (don't re-try without new info): score-threshold > 0 (nulls correct-but-low chunks), top-1/top-2 margin gate (correlated with confidence but not causally), TF-IDF (exact-token match breaks under unidecode schwa-drop), generic post-lock Viterbi smoothing (over-regularizes legal loops), Viterbi null-state smoothing (drops useful weak evidence), shorter blind-ID lookback than 30s (drops shabad-ID reliability), longer blind-ID lookback for surt (eats UEM in cold variants).

## What we're really building toward

The reference live captioning system at [bani.karanbirsingh.com](https://bani.karanbirsingh.com) is the **closest thing to the production target**. Its approach (per the writeup at [karanbirsingh.com/gurbani-captioning](http://www.karanbirsingh.com/gurbani-captioning)):
- One 118M Punjabi conformer model trained on ~300h of YouTube kirtan
- Forced-aligned to canonical SGGS lines via phonetic matching
- State machine for shabad confirmation and line tracking
- Runs against arbitrary Sikhnet Radio streams — true generalization, not per-shabad lookup

Surinder Singh has open-sourced the dataset and models:
- **[`gurbani-kirtan-yt-captions-300h-canonical`](https://huggingface.co/datasets/surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical)** — 300h labeled kirtan audio. Apache 2.0.
- **[`surt-small-v3`](https://huggingface.co/surindersinghssj/surt-small-v3)** — Whisper-small fine-tuned on 660h of Gurbani.
- **[`indicconformer-pa-v3-kirtan`](https://huggingface.co/surindersinghssj/indicconformer-pa-v3-kirtan)** — IndicConformer fine-tuned for kirtan (NeMo).

**The path to a real engine starts with these.** Continued ensembling on our existing components will keep raising the benchmark score, but is dead-end work for the production goal.

## Real roadmap (the actual remaining work)

Numbered roughly by order, not by difficulty:

1. **Establish honest evaluation hygiene.** Before claiming any new score, run on a held-out audio recording not used during tuning. Even just one new Sikhnet-Radio recording with a known-but-not-in-benchmark shabad lets us catch overfitting early.
2. **Ship surt-small-v3 with better integration as the v0 production engine.** The standalone 74% is held back by ASR-vs-matcher chunk-granularity mismatch, not by the model itself. Investigate: word-level timestamps from the HF pipeline, custom decoding that respects line boundaries, or use surt's text + faster-whisper's timestamps as a hybrid.
3. **Fine-tune surt-small-v3 on the 300h dataset (Mac-first).** Pipeline validated end-to-end on real data. `v5_mac_baseline` scored neutral (74.0%, same as `x4_pathA_surt`). `v5b_mac_diverse` scaled the data slice to 2,544 clips / 4.936h / 20 videos and scored **65.6%**, but Phase 2.6 showed the adapter is not globally worse: oracle-shabad/live0 rose to **87.4%** and a historical v3.2-ID-lock proxy scored **87.1%**. Phase 2.7 implemented the real runtime ID-lock path and scored **75.6%** because current blind-ID commits two kZhIA starts to the wrong shabad; additionally, the documented v3.2 command now repros at **73.5%**, not the archived **86.5%**. Phase 2.8 found `phase2_8_idlock_preword` (**86.6%**) by using word timestamps only for shabad ID, but generic post-lock smoothers regressed. Phase 2.9 now has a positive non-route-table runtime: `phase2_9_loop_align` scores **91.2%** by combining word-timestamp ID-lock, retro-buffered finalization, and simran-aware null alignment. It is still not production-promoted because OOS v1 is owed. Real training runs locally on M-series Macs via PyTorch + MPS — see [`docs/training_on_mac.md`](docs/training_on_mac.md). Tools: `scripts/pull_dataset.py kirtan` (pulls from `surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical` with benchmark holdouts), `scripts/finetune_path_b.py --config configs/training/surt_lora_mac.yaml` (auto-detects Whisper vs CTC from `--model-id`; target_modules default to Whisper's `q_proj/k_proj/v_proj/out_proj` when `surt-small-v3` is the base). Cloud fallback (Colab/RunPod) documented in [`docs/cloud_training.md`](docs/cloud_training.md) for the day Mac wall-clock becomes the bottleneck.
4. **Move to forced alignment over the full shabad** (Path B done right) instead of per-chunk classification. Aligns the whole shabad text to the audio as one continuous problem; naturally handles line transitions including rapid ones. Architecture sketch is in `src/path_b/`.
5. **Replace the route table with a learned dispatcher** — small classifier picking the engine based on audio features (tempo, vocal/instrumental ratio, etc.), not on shabad ID lookup. Makes ensembling honest by design.
6. **Build the live deployment surface**: streaming audio in, captions out, Sewadar UI with confirm/reset buttons (matches the reference system's UX from karanbirsingh.com).

Steps 1, 2, 3 are the highest-leverage real-world improvements. Steps 4-6 are the polish.

## Phased production training program (Mac-first → cloud → users)

A structured program to take the existing scaffold from "working smoke test" to "enterprise-grade engine others can run." The M4 Pro 48 GB is the primary box for Phase 0–5; cloud GPU only enters once wall-clock becomes the gating bottleneck. **Each phase is gated** — don't start phase N+1 until phase N's success criterion is met or a deliberate pivot is documented. **No phase ships on paired-benchmark accuracy alone** — OOS eval is required before promotion (the 12-case benchmark is a smoke test, not the metric).

When working a given phase, *fully adopt the named role* — primary literature, established conventions, naming, and instrumentation expected at that specialization. The roles aren't decoration; they constrain what "good" looks like at that step.

### Phase 0 — Reproducibility hardening
**Role:** ML Infrastructure Engineer (lead) + Senior ML Engineer.

**Hypothesis:** The scaffold trains, but runs aren't reproducible and metrics aren't tracked. Without this, every downstream phase produces unrelatable numbers.

**Approach:**
- `seed_everything()` (torch + numpy + random + transformers) into [scripts/finetune_path_b.py](scripts/finetune_path_b.py) + config schema; enable deterministic flags where MPS supports them
- Add `weight_decay`, `lr_scheduler_type` (cosine default), `warmup_ratio`, `max_grad_norm`, `gradient_checkpointing` to [configs/training/surt_lora_mac.yaml](configs/training/surt_lora_mac.yaml) and wire them through the script (today they're hardcoded or absent)
- Wire `report_to=["wandb"]` (currently `[]`); `WANDB_PROJECT=kirtan-asr`, with tensorboard fallback for offline runs
- Add `EarlyStoppingCallback(patience=2, metric="eval_loss")`
- Emit `lora_adapters/<run>/run_card.json` per adapter: `{git_sha, config_hash, data_hash, seed, hostname, wall_clock_s, peak_mem_gb, final_val_loss}`

**Success criteria:** Two same-seed runs match ≤ 0.5 % val loss; every adapter dir contains `run_card.json`; wandb shows live train/val curves end-to-end.

**Cost:** ~1 day engineering. No GPU.

### Phase 1 — Data foundation
**Role:** Speech Data Engineer (lead) + Gurmukhi domain expert.

**Hypothesis:** [scripts/pull_dataset.py](scripts/pull_dataset.py) pulls samples but the holdout is hardcoded (not loading [configs/datasets.yaml](configs/datasets.yaml)) and there's no auto train/val/test split. Bad data foundations make all downstream training noise.

**Approach:**
- Wire `configs/datasets.yaml` into pull script (kill the dead-code holdout duplication)
- Add `--split-by shabad` producing zero-leakage train/val/test manifests
- Gurmukhi normalization unit tests (unicode canonicalization, schwa-drop, ZWNJ handling) — write fixtures derived from `corpus_cache/`
- Add SNR + duration filter on clip ingestion (1 s ≤ clip ≤ 30 s, SNR threshold via `pyloudnorm`)
- Curate a 30-minute **OOS validation pack** from **5 NEW shabads** outside `{4377, 1821, 1341, 3712}` and never seen in the train pull; store under `eval_data/oos_v1/`
- Emit `training_data/<run>/data_card.md` per pull: per-shabad clip count, hours, avg duration, ragi distribution if available

**Success criteria:** No shabad appears in both train and val/test; 4 benchmark shabads never appear in any train manifest (unit-tested via pytest); `data_card.md` committed alongside manifest.

**Cost:** 1–2 days. No training compute.

### Phase 2 — Mac smoke baseline
**Role:** ML Scientist.

**Status:** ✅ Complete as pipeline proof; ❌ did not promote as model improvement. See [`docs/phase2_outcome.md`](docs/phase2_outcome.md) and [`submissions/v5_mac_baseline/notes.md`](submissions/v5_mac_baseline/notes.md).

**Actual result:** smoke and same-seed reproducibility passed; 200-clip / 0.418h real fine-tune completed on MPS; adapter loaded through PEFT inference; benchmark score was **74.0%**, equal to `x4_pathA_surt` and below the `>= 75.0%` gate. OOS v1 is still owed.

**Decision:** treat `v5_mac_baseline` as an end-to-end validation artifact, not a promoted model. Do not start Phase 3 until Phase 2.5 answers whether diversity-scaled data moves the ASR and whether that movement survives OOS.

### Phase 2.5 — Diagnostic bridge before scaling
**Role:** ML Scientist + Speech Data Engineer.

**Execution plan:** [`docs/phase2_5_plan.md`](docs/phase2_5_plan.md).

**Hypothesis:** `v5_mac_baseline` was neutral because the data slice was too small / too narrow (200 clips from 2 videos) and the benchmark failures are dominated by blind-ID + cold-window behavior, not because the training loop is broken.

**Actual result:** `v5b_mac_diverse` trained successfully (2,544 clips, 4.936h, final logged loss 0.2486) but benchmark accuracy regressed to **65.6%**. More diverse LoRA changed transcripts (only 2/12 JSONs identical to `x4_pathA_surt`) but harmed blind-ID/cold-window routing.

**Approach:**
- Curate `oos_v1` (5 shabads, 3 representative + 2 stress) and establish v3.2 / x4 / v5 baselines.
- Diversity-aware data pulling is now active and has passed for `v5b_mac_diverse`: 2,544 clips, 4.936 h, 20 source videos, 195 shabad tokens, `min_score >= 0.85`, shards 0-4, all three holdouts active with 0 benchmark video/content leaks.
- `v5b_mac_diverse` trained with the same proven config; only data scale/diversity changed.
- Benchmark eval used `HF_WINDOW_SECONDS=10`; OOS v1 remains owed before any future promotion claim.

**Success criteria:** failed. `v5b_mac_diverse` did not clear `>= 75.0%` benchmark and OOS v1 is not curated.

**Failure mode:** observed. The adapter is not neutral; it changes transcripts, but the change destabilizes blind-ID/cold-window behavior. Pause Whisper-small LoRA scaling and pivot to integration/alignment work: blind-ID robustness, windowing, word timestamps, full-shabad forced alignment, or IndicConformer.

### Phase 2.6 — Alignment diagnostic
**Role:** ML Scientist + ASR Integration Engineer.

**Status:** completed on 2026-05-16. Execution plan and results: [`docs/phase2_6_plan.md`](docs/phase2_6_plan.md).

**Goal:** determine whether `v5b_mac_diverse` improves acoustic/oracle-shabad alignment while harming blind ID, or whether the adapter is globally worse.

**Result:**
- Build a blind-ID evidence report across `x4_pathA_surt`, `v5_mac_baseline`, and `v5b_mac_diverse`.
- Score `v5b_mac_diverse` under oracle shabad IDs / ground-truth-shabad-only matching.
- `x4_pathA_surt_oracle_live0`: **85.2%**.
- `v5_mac_baseline_oracle_live0`: **85.2%**.
- `v5b_mac_diverse_oracle_live0`: **87.4%**.
- `v5b_twopass_v32_idlock`: **87.1%** using v3.2 pre-lock segments + v5b post-lock oracle alignment.

**Decision:** keep the adapter path for one more integration step. The adapter improves oracle alignment, but blind ID breaks it. Do not scale Phase 3 yet; build a runtime ID-lock integration and evaluate OOS before claiming improvement.

### Phase 2.7 — Runtime ID-lock integration + OOS gate
**Role:** ASR Integration Engineer + ML Scientist.

**Status:** complete — failed benchmark gate. Details in [`docs/phase2_7_plan.md`](docs/phase2_7_plan.md).

**Hypothesis:** v3.2/faster-whisper is a better generic blind-ID/tentative-caption engine, while `v5b_mac_diverse` is a better post-lock line-alignment engine once the shabad is known. A state-based two-pass integration can recover v5b's oracle-alignment gain without benchmark-specific route tables.

**Approach:**
- Implement a real runtime path, not only a submission merge: v3.2 owns the first 30s ID-lock window; after lock, v5b runs against the committed shabad only.
- Keep the implementation as a Layer 2 library (`src/idlock_engine.py`) plus a Layer 3 benchmark runner (`scripts/run_idlock_path.py`); the historical `scripts/merge_idlock_submissions.py` stays diagnostic-only.
- Keep the switch generic: time/state based, never hardcoded by benchmark shabad ID.
- Evaluate paired benchmark and OOS v1.
- Do not tune matcher weights on the paired benchmark.

**Result:** `v5b_idlock_runtime` scored **75.6%**. The implementation is clean, but current runtime blind-ID is not robust enough: two `kZhIA8P6xWI` starts lock to shabad `4377` instead of `1821`. Alternative existing ID aggregators (`tfidf`, `topk:3`) were worse. Separately, the documented v3.2 command now repros at **73.5%**, not the archived **86.5%**.

**Decision:** do not start Phase 3. Move to Phase 2.8: ASR reproducibility recovery + timestamp/alignment pivot.

### Phase 2.8 — ASR reproducibility recovery + timing/alignment pivot
**Role:** ASR Integration Engineer + ML Scientist.

**Status:** complete as a diagnostic phase — execution plan and probe results in [`docs/phase2_8_plan.md`](docs/phase2_8_plan.md).

**Hypothesis:** the old v3.2/faster-whisper result depends on unpinned ASR variables (faster-whisper/CTranslate2/model/VAD/cache). The path toward 95% is not another per-chunk scorer; it is reproducible ASR plus better timing/alignment (`vad_filter=False`, `word_timestamps=True`, hybrid surt text + faster-whisper timestamps, or full-shabad forced alignment).

**Approach:**
- Record and pin current ASR runtime versions and transcript-cache checksums.
- Explain the `86.5% -> 73.5%` v3.2 reproducibility drift with transcript diffs, especially the first 30s blind-ID buffer for `kZhIA8P6xWI`.
- Run timestamp prototypes before any training scale-up.
- If timestamp prototypes fail, pivot to full-shabad forced alignment.

**Result:** `phase2_8_fw_word` fixes blind ID but scores **72.0%** as a full caption path; `phase2_8_fw_vad` scores **25.4%** and is dead; `phase2_8_idlock_preword` scores **86.6%**, the best current runtime but still below the `>=87.0%` gate. Post-lock Viterbi smoother probes regressed to **77.2%** and **77.1%**, so generic per-chunk smoothing is not the path.

**Success criteria:** either reproduce the historical v3.2 result at `>=86.0%` or document the drift cause, and produce one non-route-table timing/alignment prototype with paired benchmark `>=87.0%` before returning to adapter scale-up.

### Phase 2.9 — full-shabad alignment prototype
**Role:** Alignment Engineer + ML Scientist.

**Status:** paired benchmark gate passed; OOS gate pending — plan and results in [`docs/phase2_9_plan.md`](docs/phase2_9_plan.md).

**Hypothesis:** once shabad ID is correct, the remaining miss is a sequence
alignment problem over the locked shabad, not a training-scale problem and not a
single-chunk classification problem. The aligner must encode legal line
progression and refrain/rahao loops instead of penalizing all non-local returns
equally.

**Approach:**
- Keep Phase 2.8's pre-lock word-timestamp ID path.
- Build score-lattice diagnostics for each benchmark case.
- Prototype a loop-aware text-score aligner over full-shabad matcher evidence.
- Run one default benchmark probe, archive it, then OOS before promotion.

**Success criteria:** paired benchmark `>=87.0%` without route tables, no
catastrophic case below `60%`, and OOS v1 exists before production promotion.

**Initial result:** `phase2_9_retro_buffered` scores **88.7%** with 12/12 locks
by allowing the locked post engine to revise tentative buffered captions.
`phase2_9_loop_align` then adds a generic simran/null alignment primitive and
scores **91.2%**, with `zOtIpxMT9hU_cold66` improving to **86.9%**. This clears
the paired score threshold and catastrophic-case guardrail, but is not promoted
because OOS v1 remains owed.

**Checkpoint 2026-05-16:** OOS v1 sourcing is underway, not complete. The
five-case slate is documented in [`docs/oos_v1_curation.md`](docs/oos_v1_curation.md),
OOS corpus caching and clipped audio fetch support are implemented, the five
source clips are downloaded locally under `eval_data/oos_v1/audio/` (gitignored),
and their source YouTube IDs are in `configs/datasets.yaml` holdout lists. The
same slate is encoded in `eval_data/oos_v1/cases.yaml`; `make bootstrap-oos-gt`
creates draft labels under `eval_data/oos_v1/drafts/`; `make validate-oos-gt`
guards against scoring draft or malformed labels. The remaining gate is
`make oos-review-pack`, hand-corrected GT JSONs under
`eval_data/oos_v1/test/`, validation by `make validate-oos-gt`, then
`make eval-oos-loop-align`. The live labeling checkpoint is
[`docs/oos_v1_labeling_checkpoint.md`](docs/oos_v1_labeling_checkpoint.md).
Do not train on these OOS recordings; they are the variance check for whether
the 91.2% paired-benchmark architecture generalizes.

### Phase 3 — Mac-scale real fine-tune
**Role:** ML Scientist (acoustic modeling) (lead) + Optimization Engineer.

**Precondition:** Phase 2.9 has produced a positive full-shabad alignment diagnostic; OOS v1 must exist before Phase 3 promotion or scale-up. Do not spend the 3 × 24h budget on paired-benchmark confidence alone.

**Hypothesis:** If Phase 2.8 shows positive movement, 50 h of curated kirtan + SpecAugment + cosine LR + LoRA r=32 + weight decay + 3 seeds should push surt-small-v3 to ≥ 85 % benchmark and ≥ 80 % OOS — within 24–48 h M4 Pro wall-clock per run.

**Approach:**
- Pull 50 h slice at `min_score ≥ 0.85`
- Add SpecAugment (freq + time masking) and ±10 % speed perturbation to [src/path_b/dataset.py](src/path_b/dataset.py) — currently no augmentation, single view per epoch
- LoRA r=32, target_modules = `q_proj/k_proj/v_proj/out_proj` **plus** `fc1/fc2` (MLP) — current default is attention-only
- Cosine LR with `warmup_ratio=0.1`, `weight_decay=0.01`, `max_grad_norm=1.0`
- `gradient_checkpointing=True` to free memory for `batch_size=8` (currently False, sub-optimal)
- 3 epochs, early stop on val patience=2
- **3 seeds** for variance estimate

**Success criteria:** Best-of-3 seeds: benchmark ≥ 85 %, OOS ≥ 80 %; cross-seed variance < 2 pts (otherwise regularize harder before scaling); wall-clock < 48 h.

**Cost:** 3 × ~24 h = ~72 h M4 Pro. Overnight + weekend.

**Failure mode → Phase 4 ablations:** if best-of-3 < 85 %, capacity or chunk granularity is the bottleneck, not data scale.

### Phase 4 — Ablation + capacity scaling
**Role:** ML Scientist (experimentation).

**Hypothesis:** The marginal lifts come from data scale, target_modules coverage, and augmentation. Quantify each independently to find the Pareto frontier of accuracy vs wall-clock.

**Approach (independent wandb sweeps):**
- Data scale curve: 10 / 25 / 50 / 100 h → fit log-linear, predict the 200 h+ payoff
- LoRA rank sweep: 8 / 16 / 32 / 64
- Target modules: attn-only / attn+MLP / encoder-only / decoder-only / full
- LR sweep on 25 h subset: {5e-6, 1e-5, 3e-5, 5e-5}
- Augmentation ablation: none / specaug / specaug+speed / specaug+speed+noise

**Success criteria:** Pareto frontier documented in `submissions/ablations/notes.md` with plot; clear next architecture bet identified.

**Decision gate to Phase 5b:** If best ablation < 88 % benchmark, switch acoustic backbone — `surt-medium` or `surindersinghssj/indicconformer-pa-v3-kirtan` (NeMo Conformer). The reference system at karanbirsingh.com uses a 118 M *Conformer*, not Whisper; LoRA on Whisper-small has a real ceiling.

**Cost:** ~5 sweeps × ~12 h = ~60 h M4 Pro.

### Phase 5 — Honest evaluation & generalization audit
**Role:** ML Test Engineer (lead) + Generalization Researcher.

**Hypothesis:** A production-ready model maintains accuracy across (a) unseen shabads, (b) unseen ragis (singers), (c) unseen recording conditions (live vs studio, mic quality, harmonium-heavy vs voice-forward).

**Approach:**
- OOS eval v2 with **≥ 20 shabads, ≥ 5 ragis**, mixed conditions
- Per-slice frame accuracy reporting (per-shabad / per-ragi / per-condition)
- Calibration curve: predicted confidence vs actual correctness at each threshold
- Worst-slice analysis — what failure modes correlate (tempo, instrumentation, raga, etc.)
- Adversarial set: noisy clips, harmonium-heavy, fast tempo, sehaj-style

**Success criteria:** OOS v2 mean ≥ 80 %, worst-slice ≥ 70 %; no catastrophic regression on the 4 benchmark shabads (regression test); calibration honest at deployment thresholds.

**Cost:** 1–2 days data collection, 1 day automation, ~6 h compute.

### Phase 6 — Cloud handoff
**Role:** MLOps Engineer (lead) + Cloud Platform Engineer.

**Hypothesis:** Exceeding the Mac ceiling needs 200 h+ data and/or distributed training. Cloud GPU turns weeks of wall-clock into hours.

**Approach:**
- Reproducible Dockerfile mirroring [requirements-cloud.txt](requirements-cloud.txt)
- Pick one cloud target — **Modal recommended** (cheap, ephemeral, great DX) with RunPod as the documented fallback per [docs/cloud_training.md](docs/cloud_training.md)
- `scripts/train_remote.py` mirrors local args and spins a cloud job
- Accelerate + DeepSpeed ZeRO-2 config for multi-GPU (A100 8x scale)
- S3 / R2 for dataset + adapter sync; **same wandb project** so local & cloud runs are directly comparable
- `make train-cloud` target wired into the Makefile

**Success criteria:** `make train-cloud DATA=200h` reproduces a Phase 3 result in < 4 h instead of 24 h; per-run $ cost logged; adapter pulled back to local automatically.

**Cost:** 2–3 days engineering; ~$50–200 cloud spend for first runs.

### Phase 7 — Model registry + serving (others-use-it surface)
**Role:** ML Platform Engineer (serving) (lead) + Backend Engineer.

**Hypothesis:** "Others can use it" requires (a) a stable, versioned artifact, and (b) a hosted endpoint with sane SLOs.

**Approach:**
- Publish adapter to `<org>/surt-small-v3-kirtan-lora-v1` on HF Hub with a **proper model card**: training data sources + hours, hyperparameters, holdout policy, benchmark + OOS scores, intended use, known failure modes, ethical considerations, license
- Inference container (FastAPI + transformers + peft); LoRA merged at container build time via `peft.merge_and_unload()`
- Deploy on Modal / HF Inference Endpoints / SageMaker (Modal recommended for cost)
- API: `POST /caption {audio_url, mode: live|offline}` → segments JSON in the existing schema
- Versioning: semantic version + git SHA + adapter hash in every response

**Success criteria:** External `curl` returns captions for a Sikhnet Radio recording; p95 latency < 2× audio duration (real-time feasible); $/minute logged.

**Cost:** 2–3 days for v1 endpoint.

### Phase 8 — On-device (iOS / Apple Silicon)
**Role:** Edge ML Engineer (Apple Silicon specialist).

**Hypothesis:** Gurdwaras with poor connectivity need on-device inference. ANE makes Whisper-small fast enough on iPhone 14+.

**Approach (largely scaffolded — Phase 8 finishes the polish):**
- LoRA-merged surt → `whisperkittools` → `.mlpackage` via [scripts/export_coreml.py](scripts/export_coreml.py)
- ANE palettization profiles (6-bit / 4-bit) A/B
- Numerical parity test (Mac CoreML vs HF Python, target < 1 % WER divergence) — script stub referenced in [docs/ios_deployment.md](docs/ios_deployment.md) needs to land
- Real-device iPhone latency benchmark (not just simulator)
- End-to-end parity vs cloud endpoint output

**Success criteria:** iPhone real-time factor < 1.0; < 1 % WER divergence from cloud; app ships, captions display.

**Cost:** ~1 week.

### Phase 9 — Continuous improvement loop
**Role:** Applied ML Scientist (lead) + Product / domain advisor.

**Hypothesis:** Real users surface failure modes the OOS set doesn't. The model that ships v1 won't be the model that ships v3.

**Approach:**
- App + cloud endpoint log low-confidence regions (anonymized audio fragments + predicted ID); opt-in only, with explicit user consent
- Weekly active-learning review queue → hand-label → add to training set
- Drift monitor: track per-shabad accuracy over time on a fixed eval set
- Monthly LoRA refresh cadence, gated on OOS regression test (no regression > 1 pt)
- Sewadar feedback channel: "wrong shabad" reports → ticket → labeled correction → next batch

**Success criteria:** OOS accuracy curve improves quarter-over-quarter; drift never exceeds 5 pt on any shabad before being addressed.

**Cost:** Continuous; ~10 % engineering FTE.

### Expert role map (per phase)

| # | Primary role | Secondary role |
|---|---|---|
| 0 | ML Infrastructure Engineer | Senior ML Engineer |
| 1 | Speech Data Engineer | Gurmukhi domain expert |
| 2 | ML Scientist | — |
| 3 | ML Scientist (acoustic modeling) | Optimization Engineer |
| 4 | ML Scientist (experimentation) | — |
| 5 | ML Test Engineer | Generalization Researcher |
| 6 | MLOps Engineer | Cloud Platform Engineer |
| 7 | ML Platform Engineer (serving) | Backend Engineer |
| 8 | Edge ML Engineer (Apple Silicon) | — |
| 9 | Applied ML Scientist | Product / domain advisor |

### Cross-cutting guarantees

These apply across every phase and aren't optional:

- **Every adapter committed has a `run_card.json`** with git_sha + config_hash + data_hash + seed + scores. No artifact without lineage.
- **Every score reported specifies benchmark vs OOS.** Mixing them is intellectually dishonest; the 12-case benchmark is overfit-prone.
- **No phase ships on a single seed.** Cross-seed variance ≥ 2 pts means add regularization or augmentation, don't ship.
- **No data scrape from the 4 benchmark shabads, ever** — enforced in `configs/datasets.yaml` + pytest.
- **W&B is the source of truth for training curves**; HF Hub is the source of truth for artifacts; git is the source of truth for code.
- **Ensembles and route tables are research-only.** Production is a *single model* serving a *single deployment*. The leaderboard's x5/x6 routes don't count toward production.

## Repo layout

The repo holds **two engine implementations side-by-side** (Path A and Path B). They share infrastructure (audio fetcher, corpus cache, scoring) but their engines live in separate folders so changes to one never affect the other.

```
src/                        Path A engine (FROZEN at v3.2 = 86.5% live blind)
  asr.py                    Dual-backend ASR wrapper:
                              - faster_whisper (default): CPU, Silero VAD, produces v3.2
                              - mlx_whisper: Apple Silicon GPU, faster but different chunking
  matcher.py                rapidfuzz scoring + TfidfScorer + score_chunk()
  smoother.py               smooth() and smooth_with_stay_bias() — causal
  shabad_id.py              identify_shabad() (chunk_vote) + per_chunk_global_match
  path_b/                   Path B engine (in development)
    __init__.py             Scaffold; CTC scorer + loop-aware HMM go here
scripts/
  run_benchmark.py          Stage 0 empty submitter
  fetch_audio.py            yt-dlp + ffmpeg → audio/*.wav
  build_corpus.py           BaniDB API → corpus_cache/<id>.json
  run_path_a.py             Path A runner (--backend flag selects fw or mlx)
  run_path_b.py             (future) Path B runner
submissions/v<N>_<name>/    one folder per experiment + notes.md + tiles.html
audio/                      16kHz mono WAVs (gitignored)
corpus_cache/               BaniDB shabad lines (gitignored)
asr_cache/                  cached transcripts, key includes backend (gitignored)
```

### Reproducing Path A v3.2 (86.5% blind + live)

```bash
python scripts/run_path_a.py \
  --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 \
  --stay-bias 6 --blind --blind-aggregate chunk_vote \
  --blind-lookback 30 --live --tentative-emit \
  --out-dir submissions/v3_2_pathA_no_title
```

Default `--backend faster_whisper` is the canonical Path A backend. `--backend mlx_whisper` is available but produces different chunk granularity that requires retuning (see v4_mlx_* notes).

### Path B status

| Phase | Status | Result |
|---|---|---|
| B0-B3: MMS-1B + HMM | ✅ | 61.0% (below Path A 86.5%) |
| X1: Demucs vocal separation | ✅ | fw -20.5, mlx +10.5 — not a universal win |
| X3.0: swap to w2v-bert-punjabi | ✅ | 70.3% (+9.3 vs MMS, still below Path A) |
| B.1: LoRA fine-tune pipeline | ✅ | smoke-tested, ready for real data |
| B.2-B.6: collect kirtan data + fine-tune | ⏳ next | target 90-95% |

### Fine-tuning (LoRA on surt-small-v3) — Mac-first plan

The ASR plateau (Path A 86.5%, Path B 70.3%) is structural and only escapes via domain-adapted acoustic modeling. The production target is `surt-small-v3` (Whisper-small architecture, kirtan-fine-tuned by surindersinghssj). Path B (w2v-bert + HMM) remains as research scaffolding but is not the deployment target.

**Pipeline (built):**
- `src/path_b/dataset.py` — manifest-driven (audio, text) data loader; has separate paths for CTC and Whisper Seq2Seq
- `scripts/pull_dataset.py` — unified data acquisition with subcommands (`kirtan`, `sehaj`, `sehajpath`); enforces benchmark holdout
- `scripts/build_training_dataset.py` — YouTube-source fallback when HF isn't enough
- `scripts/finetune_path_b.py` — LoRA fine-tune; auto-detects Whisper vs CTC from `--model-id` (or override with `--model-type`). Loads pinned hyperparameters from `configs/training/surt_lora_mac.yaml`.
- `scripts/build_smoke_manifest.py` — tiny pipeline-validation manifest (NOT for real training — contaminates test set)
- `scripts/run_path_a.py --backend huggingface_whisper --adapter-dir <path>` — inference with a saved LoRA adapter against the canonical Path A pipeline
- Training cache: `lora_adapters/<name>/` (gitignored)
- Training data: `training_data/<manifest>/` (gitignored — keep audio off the public repo)

**Canonical end-to-end workflow (Mac):**

```bash
# 1) Pull a labeled slice (benchmark shabads filtered out automatically):
python scripts/pull_dataset.py kirtan \
  --out-dir training_data/kirtan_v1 \
  --num-samples 200 --min-score 0.8

# 2) LoRA fine-tune surt-small-v3 (Whisper Seq2Seq path auto-detected):
python scripts/finetune_path_b.py \
  --config configs/training/surt_lora_mac.yaml \
  --manifest training_data/kirtan_v1/manifest.json \
  --output-dir lora_adapters/surt_mac_v1

# 3) Evaluate against the paired benchmark:
python scripts/run_path_a.py \
  --backend huggingface_whisper \
  --model surindersinghssj/surt-small-v3 \
  --adapter-dir lora_adapters/surt_mac_v1 \
  --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 \
  --stay-bias 6 --blind --blind-aggregate chunk_vote \
  --blind-lookback 30 --live --tentative-emit \
  --out-dir submissions/v5_surt_mac_v1

python ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/v5_surt_mac_v1 \
  --gt   ../live-gurbani-captioning-benchmark-v1/test/

# 4) Out-of-set evaluation (honest accuracy — required before claiming a number):
python scripts/eval_oos.py \
  --data-dir eval_data/oos_v1 \
  --pred-dir submissions/oos_v1_surt_mac_v1 \
  --engine-config configs/inference/v3_2.yaml
```

**Smoke-test (validated on prior w2v-bert path):** `PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/finetune_path_b.py --manifest <smoke.json> --output-dir /tmp/lora_smoke --max-steps 20 --batch-size 1`. ~1.4 steps/sec on Apple Silicon. For Whisper Seq2Seq specifically: cross-entropy loss is MPS-native (no CPU fallback like CTC has), so throughput is higher — measure on first real run.

**Data sources (sorted by current relevance):**

1. **`surindersinghssj` on HuggingFace** — primary source.
   - [`surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical`](https://huggingface.co/datasets/surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical) — 300h kirtan audio + line text, 208k samples, Apache 2.0
   - [`surindersinghssj/gurbani-sehajpath-yt-captions-canonical`](https://huggingface.co/datasets/surindersinghssj/gurbani-sehajpath-yt-captions-canonical) — ~160h spoken paath
   - [`surindersinghssj/surt-small-v3`](https://huggingface.co/surindersinghssj/surt-small-v3) — our base model, Whisper-small fine-tuned on 660h Gurbani
   - Phase X4 (committed) confirmed: pure surt substitution hits 74.0% overall but wins +10-17 points on kchMJPK9Axs vs Path A v3.2. Oracle ensemble of {v3.2, X4} per case ≈ 90.3%.
2. **YouTube kirtan with shabad-level metadata** — `scripts/build_training_dataset.py` covers this fallback path.
3. **AI4Bharat IndicVoices Punjabi subset** — general Punjabi speech regularization (stub subcommand in `pull_dataset.py`, lands when needed).
4. **SikhiToTheMax / Khalis Foundation archives** — potential broadcast-quality source for future iterations.

**Compute reality:**
- M4 Pro 48GB (or any M-series ≥16GB unified): fits surt-small-v3 LoRA comfortably. ~4-8 hours per epoch for a 50h dataset. **Primary path.**
- Cloud GPU (Colab A100, RunPod) covered by `requirements-cloud.txt` + [`docs/cloud_training.md`](docs/cloud_training.md). Use only if Mac wall-clock becomes the bottleneck.

**Anti-overfitting hygiene:**
- Hold out by *shabad identity*, not by recording
- Strict separation: benchmark's 4 shabads (4377, 1821, 1341, 3712) NEVER appear in train (enforced by `pull_dataset.py` and `configs/datasets.yaml`)
- 80/10/10 train/val/test with shabad-level boundaries
- Early stopping on val loss
- LoRA's parameter efficiency (~1% trainable) is its own regularization

**Expected lift if executed well:** lyrics-alignment and Quran-recitation literature consistently report +10-15 points from in-domain fine-tuning. Target: 90-95% on the paired benchmark, 85-90% on OOS eval.

**Cloud training**: see [docs/cloud_training.md](docs/cloud_training.md) for cell-by-cell Google Colab + RunPod walkthrough. Mac is fine for smoke tests; real fine-tune (20+ hours of data) needs cloud GPU. A100 turns days of CPU into hours.

## External references

- **BaniDB** — https://api.banidb.com (e.g. `api.banidb.com/v2/shabads/{shabad_id}`). Source of canonical Gurmukhi lines and `verse_id`s.
- **SikhiToTheMax** — uses the same `shabad_id` and `line_idx` as BaniDB; either service can resolve a prediction to displayable text.

## Conventions for this repo

- **Never overwrite a submission folder.** Each experiment lives at `submissions/v<N>_<short_name>/` and ships with a `notes.md` describing the engine config (model, params, mode, score). The score is part of the experiment record — don't mutate a folder after scoring it.
- **Always run the scorer and visualizer on every new run.** After producing a submission, run `eval.py` and also generate `tiles.html` with the benchmark's `visualize.py`:

  ```bash
  python ../live-gurbani-captioning-benchmark-v1/visualize.py \
    --pred submissions/<run_name>/ \
    --gt   ../live-gurbani-captioning-benchmark-v1/test/ \
    --audio-dir audio/ \
    --out submissions/<run_name>/tiles.html
  ```

  The HTML strip view catches systematic failures that an aggregate accuracy number hides.
- **Keep the engine deterministic where possible** so re-runs are reproducible against committed submission JSONs.
