# Phase 2 — Mac smoke baseline + first real fine-tune

Companion to the high-level Phase 2 entry in [`CLAUDE.md`](../CLAUDE.md#phase-2--mac-smoke-baseline). This is the operational detail — pre-flight, exact `make` invocations, expected outputs at each step, success gate, failure modes, decision branches.

**Status:** Completed on 2026-05-16. Phase 2 validated the training/eval pipeline but did **not** promote as a model improvement. `v5_mac_baseline` scored `74.0%`, equal to `x4_pathA_surt` and below the `>= 75.0%` gate. The current go-forward plan is [`docs/phase2_outcome.md`](phase2_outcome.md): run Phase 2.5 before Phase 3.

**Role:** ML Scientist.

**Original hypothesis:** A clean 200-sample fine-tune of `surt-small-v3` on curated kirtan should reproducibly move:
- **Paired benchmark frame accuracy:** `+1 to +3 pts` above the `x4_pathA_surt` baseline of 74.0%
- **OOS frame accuracy (if `oos_v1` is curated):** `+1 to +2 pts` above v3.2's OOS baseline (TBD, run on v3.2 first)

**Actual result:** the loop is sound, but the 200-clip run was neutral. Training loss decreased, reproducibility was bit-identical on the smoke gate, and PEFT inference loaded the adapter. The more likely bottleneck is data scale/diversity plus blind-ID/cold-window behavior, so Phase 3 now requires the Phase 2.5 diagnostic bridge.

## Pre-flight (one-time, ~30 min)

These steps must complete before any Phase 2 step runs. They're system-level setup, not training.

| Step | Command | Done when | Wall-clock |
|---|---|---|---|
| Python 3.10+ | `brew install python@3.12` | `python3 --version` reports ≥3.10 | 5–10 min |
| Update Makefile PYTHON | `PYTHON=python3.12 make doctor-train` returns `✓` | doctor-train passes | <1 min |
| ffmpeg (for audio fetch) | `brew install ffmpeg` | `ffmpeg -version` works | 2–5 min |
| Paired benchmark repo cloned | `git clone <benchmark-repo-url> ../live-gurbani-captioning-benchmark-v1` | `make doctor-dev` passes | <1 min |
| Install Python deps | `make install` (or `pip install -r requirements-mac.txt`) | `python3 -c "import torch, transformers, peft, soundfile, yaml"` works | 5–10 min (torch is large) |
| Fetch benchmark audio | `make fetch-audio` | 4 WAVs exist in `audio/` | 5–15 min depending on connection |
| Build the smoke manifest | `make smoke-manifest` | `training_data/smoke/manifest.json` exists with 4 entries | <1 min |
| Verify full test suite passes WITH deps | `make test` | 83/83 pass, **0 skipped** on the current Python 3.12 venv | <20 sec |

**Gate before proceeding:** if any pre-flight step fails, fix before continuing. Phase 2.A failing on missing deps is wasted wall-clock.

## Phase 2.A — Smoke validation of the trained loop (~5 min wall-clock)

**Goal:** prove the end-to-end training loop runs on this Mac WITH the new Phase 0 instrumentation, before burning hours on a real fine-tune.

```bash
make smoke
```

This invokes `scripts/finetune_path_b.py` with `--max-steps 20 --batch-size 1 --report-to none` against the 4-clip smoke manifest. Expected to run ~30 seconds on M4 Pro.

**Success criteria:**
1. Exit code 0
2. `/tmp/lora_smoke/adapter_model.safetensors` exists (LoRA adapter weights)
3. `/tmp/lora_smoke/run_card.json` exists (Phase 0.B's lineage emission worked)
4. Inspect `run_card.json`: `status == "completed"`, `seed == 42`, `wall_clock_s` is plausible (~5–30 s), `peak_mem_gb` is non-null, `data_hash` matches the smoke manifest hash

**Failure modes & responses:**
| Symptom | Likely cause | Action |
|---|---|---|
| `ModuleNotFoundError: torch` | `make install` not run | Run pre-flight step 5 |
| `RuntimeError: MPS backend out of memory` | Other apps holding RAM | Quit apps; M4 Pro 48 GB should easily fit batch=1 |
| `AttributeError: 'NoneType' object has no attribute 'forced_decoder_ids'` | transformers/peft version skew | Pin transformers to a known-good version (4.46+) |
| Hangs on first eval (no eval manifest → shouldn't hang) | Misconfigured eval_strategy | Verify `--eval-manifest` not silently set in config; `make smoke` should run with eval_strategy=no |
| `run_card.json` has `status="crashed"` | Loop crashed mid-train | Inspect Python traceback; the run_card itself confirms the try/finally wiring works |

## Phase 2.B — Reproducibility gate (~10 min wall-clock)

**Goal:** verify Phase 0's "same-seed → same loss" claim on real hardware.

```bash
rm -rf /tmp/lora_smoke
make smoke && cp /tmp/lora_smoke/run_card.json /tmp/run_card_A.json
rm -rf /tmp/lora_smoke
make smoke && cp /tmp/lora_smoke/run_card.json /tmp/run_card_B.json

python3 -c "
import json
a = json.load(open('/tmp/run_card_A.json'))
b = json.load(open('/tmp/run_card_B.json'))
assert a['config_hash'] == b['config_hash'], 'config hashes diverge'
assert a['data_hash'] == b['data_hash'], 'data hashes diverge'
ta = a['final_train_loss']; tb = b['final_train_loss']
delta = abs(ta - tb) / abs(ta) if ta else float('inf')
print(f'run A loss: {ta:.4f}  run B loss: {tb:.4f}  rel delta: {delta*100:.2f}%')
assert delta <= 0.02, f'PHASE 0 GATE FAILED: MPS delta {delta*100:.2f}% exceeds 2%'
print('PHASE 0 GATE: PASS')
"
```

**Success criteria:** script prints `PHASE 0 GATE: PASS`. MPS isn't bit-deterministic so we expect non-zero delta; the bar is ≤ 2 %.

**Failure modes:**
- **Delta > 2 %:** seed isn't propagating somewhere. Likely places: PEFT's LoRA init may not respect `set_seed`. Check by adding `print(model.state_dict()[<adapter_key>][:5])` after `get_peft_model` in both runs.
- **config_hash diverges:** something time-dependent leaked into the args dict. Inspect `args` field in both `run_card.json`; diff to find the offender.

If the gate fails, **Phase 2 stops here.** No point training a real adapter you can't reproduce. Fix the seed wiring first (likely a Phase 0.D patch PR).

## Phase 2.C — Data pull (~10–15 min wall-clock + download time)

```bash
make data DATA_DIR=training_data/v5_mac_baseline DATA_SAMPLES=200
```

This runs `scripts/pull_dataset.py kirtan --out-dir training_data/v5_mac_baseline --num-samples 200 --min-score 0.8`.

**Success criteria:**
1. `training_data/v5_mac_baseline/manifest.json` exists with ~200 records
2. `training_data/v5_mac_baseline/data_card.md` exists (Phase 1.D output) with rejection counts, top shabads, holdout enforcement confirmed
3. Inspect data_card: holdout shows the 4 benchmark shabads and 4 benchmark videos listed.
4. Confirm all three holdout layers are active: `holdout_shabad`, `holdout_video`, and `holdout_content`.
5. Confirm no rows overlap benchmark canonical lines. The kirtan dataset uses 3-character `canonical_shabad_id` tokens, not BaniDB integer IDs, so content-based holdout is the reliable shabad-level guard.

**Note on split:** for Phase 2 we use the default `--split-by none` (single manifest). The shabad-level split (Phase 1.B) is for Phase 3 where we want a real val set. Phase 2's signal is from benchmark + OOS eval, not from per-step val loss.

## Phase 2.D — Real fine-tune (~20–45 min train wall-clock on M4 Pro)

```bash
make train \
  DATA_DIR=training_data/v5_mac_baseline \
  TRAIN_OUT=lora_adapters/v5_mac_baseline
```

This is `scripts/finetune_path_b.py --config configs/training/surt_lora_mac.yaml --manifest training_data/v5_mac_baseline/manifest.json --output-dir lora_adapters/v5_mac_baseline`. Default config: 3 epochs, lr 1e-5, batch 4, grad_accum 2, LoRA r=16, cosine schedule, weight_decay 0.01.

**Expected throughput on M4 Pro 48GB:** roughly 1.5–3.0 train steps/sec. 200 samples × 3 epochs / (batch 4 × grad_accum 2) ≈ 75 optimizer steps, ~5–15 minutes per epoch. **Wall-clock estimate: 20–45 minutes for the train itself, +overhead.**

(The full Phase 2 loop took longer because it included data pull, dependency debugging, evaluation, and documentation. The training-only segment for `v5_mac_baseline` was 232 s.)

**Run in background; monitor via:**
- Stream tensorboard from `lora_adapters/v5_mac_baseline/runs/*` (Phase 0.A default when no `WANDB_API_KEY` is set)
- Or `tail -f` the make output for `loss=X.XX` lines

**Success criteria:**
1. Train completes (exit 0)
2. `lora_adapters/v5_mac_baseline/adapter_model.safetensors` exists
3. `run_card.json` shows `status="completed"`, `final_train_loss` is a sane number (typically 0.5–2.0 for Whisper LoRA on this scale)
4. Loss trajectory in tensorboard/wandb is monotone decreasing across epochs (occasional bumps OK; should not be flat or increasing)

**Failure modes:**
| Symptom | Action |
|---|---|
| Loss flat from step 0 | Likely LR is too low; bump to 3e-5 in a YAML override for Phase 2.D' |
| Loss explodes / NaN | LR too high or grad_clip not engaging; verify `max_grad_norm=1.0` in run_card |
| MPS OOM mid-training | Memory leak in HF Trainer's eval (shouldn't fire since eval_strategy=no); reduce batch_size to 2 |
| Run dies, run_card.json shows `status="crashed"` | Inspect the saved partial loss + stack trace; the lineage is preserved by Phase 0.B's wiring |

## Phase 2.E — Evaluation (paired benchmark + OOS) (~30–60 min wall-clock)

```bash
make eval \
  TRAIN_OUT=lora_adapters/v5_mac_baseline \
  EVAL_OUT=submissions/v5_mac_baseline
```

This invokes `scripts/run_path_a.py --backend huggingface_whisper --model surindersinghssj/surt-small-v3 --adapter-dir lora_adapters/v5_mac_baseline ...` followed by the benchmark's `eval.py`. The fair comparison to `x4_pathA_surt` uses `HF_WINDOW_SECONDS=10`; 30-second windows reproduce the known-bad X4 regime and scored 46.2% during the v5 audit.

**Success criteria (TWO-ARM gate):**

**Arm 1 — Benchmark only (minimum bar, always required):**
- Frame accuracy ≥ **75 %** on the paired benchmark (+1 pt over `x4_pathA_surt` baseline of 74.0%)

**Arm 2 — OOS validation (required only if `oos_v1` has been curated per Phase 1.5):**
- Frame accuracy reported (any honest number) on `oos_v1`
- Mean ≥ **v3.2's OOS baseline + 1 pt** (the v3.2 OOS baseline itself is a TBD number that gets established the first time we run v3.2 against the curated `oos_v1`)
- No catastrophic per-case regression (no single case drops > 10 pts vs v3.2)

If `oos_v1` is not yet curated when Phase 2 runs, document this in `submissions/v5_mac_baseline/notes.md` as "OOS owed" — the Phase 2 score then promotes provisionally, with OOS measured the moment 1.5 lands.

## Phase 2.F — Documentation (~15 min wall-clock)

The submission folder must include:

```
submissions/v5_mac_baseline/
├── *.json                  # one per benchmark case (engine output)
├── tiles.html              # from visualize.py — strip view of pred vs GT
├── notes.md                # human-readable summary
└── run_card.json           # symlink or copy from lora_adapters/v5_mac_baseline/run_card.json
```

`notes.md` template:

```markdown
# v5_mac_baseline — Phase 2 first real Mac fine-tune

**Date:** YYYY-MM-DD
**Adapter:** lora_adapters/v5_mac_baseline
**Run card:** see run_card.json (config_hash, data_hash, seed, all there)

## Config

- Base model: surindersinghssj/surt-small-v3
- Data: training_data/v5_mac_baseline (200 samples, min_score 0.8, no split)
- Hyperparameters: see configs/training/surt_lora_mac.yaml
- Hardware: M4 Pro 48 GB, MPS, fp32 (current torch 2.5 / accelerate window cannot use MPS fp16 safely)
- Wall-clock: <X> minutes train + <Y> minutes eval

## Scores

- Paired benchmark (blind + live): **XX.X %** (baseline x4_pathA_surt: 74.0 %)
- OOS v1: **XX.X %** OR "owed — oos_v1 not curated at run time"

## Generalization annotation

Generalizes? Yes / No / Conditional — explain.

## Observations

- (anything surprising about loss trajectory, per-case scores, etc.)
```

## Decision gate after Phase 2

**If Arm 1 ≥ 75 % AND (Arm 2 ≥ +1 pt OR Arm 2 owed):**
→ Phase 2 would promote. This did not happen for `v5_mac_baseline`.

**If Arm 1 < 75 %:**
→ Phase 2 does NOT promote. For `v5_mac_baseline`, the loop was validated and the adapter path was active, so the next diagnosis is not "fix training from scratch." Run Phase 2.5:
- curate OOS v1 and establish v3.2/x4/v5 baselines,
- pull a larger, more diverse `v5b_mac_diverse` slice,
- keep the same config for the first retry,
- compare benchmark, OOS, and transcript deltas.

**If Arm 1 ≥ 75 % but Arm 2 catastrophically regresses on a specific case:**
→ Do not promote as a model improvement. Document the per-case failure in `notes.md`, then diagnose with Phase 2.5 / OOS before scaling. Do NOT change the engine to "fix" a single case — that's overfitting in disguise.

## What Phase 2 explicitly does NOT do

- **Hyperparameter sweep.** That's Phase 4's job. Phase 2 uses the defaults from `configs/training/surt_lora_mac.yaml` and shows the pipeline can move the needle at all. One run, one seed, one config.
- **OOS pack curation.** Phase 1.5 owns that workstream. Phase 2 runs whether or not 1.5 has landed.
- **`gradient_checkpointing=true` flip.** Phase 0 deliberately kept it off. Phase 3 verifies it works with PEFT+MPS before the YAML default flips.
- **`eval_strategy=steps` flip.** Same reasoning — needs Phase 1.B val manifest first; Phase 2 uses no eval (faster, simpler smoke).
- **Multiple seeds.** Phase 3 runs 3 seeds. Phase 2 runs one to prove "the loop runs and moves the needle"; cross-seed variance is a Phase 3 measurement.

## Estimated total wall-clock

| Phase | Wall-clock |
|---|---|
| 2.A smoke validation | 5 min |
| 2.B reproducibility gate | 10 min (two smoke runs back to back) |
| 2.C data pull | 10–15 min (depends on connection) |
| 2.D real fine-tune | 20–45 min train + overhead |
| 2.E evaluation | 30–60 min |
| 2.F documentation | 15 min |
| **Total** | **~2–3 hours, run in background; user attention ~30 min total** |

This is half a day in elapsed time but only ~30 min of active hands-on. The rest is the M4 Pro working while you do other things.

## Sequencing vs Phase 1.5

Phase 1.5 is now the front door to Phase 2.5. OOS curation ([`docs/oos_v1_curation.md`](oos_v1_curation.md)) should happen before the next model-improvement claim.

Recommended order after the v5 result:
1. Implement or extend the OOS audio-fetch/bootstrap helper for arbitrary URLs.
2. Curate the 5-case OOS v1 pack and establish v3.2/x4/v5 baselines.
3. Add diversity-aware data-pull controls if the existing first-shard pull cannot meet source-video/shabad-token floors.
4. Pull `v5b_mac_diverse` (`1k-5k` clips, `min_score >= 0.85`, content holdout active).
5. Train/evaluate with the same config first; only then consider LoRA rank/LR/augmentation changes.

The Mac pipelines for OOS curation and training do not conflict, but OOS should no longer be optional for promotion.
