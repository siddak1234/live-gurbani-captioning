# Fine-tuning `surt-small-v3` on Apple Silicon

This is the recipe for LoRA-fine-tuning surt-small-v3 on an M-series Mac (validated against M4 Pro, 48 GB unified memory). The output is a small LoRA adapter that gets merged back into the base model at inference time, or used directly via PEFT.

## Why this works on Mac

- **Model size:** 240M params, ~500 MB in fp16. Trivially fits 16 GB; 48 GB is overkill but comfortable.
- **LoRA trainable params:** ~3M (about 1% of the base). Memory bound is dominated by activations + batch, not model weights.
- **Apple GPU (MPS) acceleration:** Whisper Seq2Seq loss is plain cross-entropy — natively supported on MPS. Unlike CTC, there is no CPU fallback for the loss op. Throughput on M4 Pro: ~1-3 steps/sec at batch 4 (estimated, run the smoke step to verify on your machine).

## Prereqs

```bash
# Mac dependencies
pip install -r requirements-mac.txt

# Environment hints (already set by .claude/settings.json inside Claude Code)
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_ENABLE_HF_TRANSFER=1
```

## 1. Pull training data

```bash
python scripts/pull_dataset.py kirtan \
  --out-dir training_data/kirtan_v1 \
  --num-samples 200 \
  --min-score 0.8
```

`--num-samples` controls how many qualifying clips to keep. The first parquet shard of the source dataset holds ~1500 rows (~2-3h of audio); raise `--max-scan` to widen the pool. Benchmark shabads, benchmark videos, and benchmark canonical line text are filtered out automatically.

For multi-source training mixes (kirtan + sehaj + sehajpath), pull each separately and concat the resulting manifests. See `configs/datasets.yaml` for the source registry.

For the Phase 2.5 diagnostic pull, prefer the diversity-gated target:

```bash
make data-v5b
```

This scans multiple parquet shards, keeps `min_score >= 0.85`, and fails unless the kept manifest has enough source-video and shabad-token diversity. With diversity floors active, `DATA_SAMPLES` is a minimum rather than a hard cap; the pull can keep extra clips until the floors pass. Inspect `training_data/v5b_mac_diverse/data_card.md` before training.

Current checkpoint: `v5b_mac_diverse` passed the data gate with `2,544` clips,
`4.936 h`, `20` videos, `195` shabad tokens, and `0` benchmark video/content
leaks. It trained successfully, but blind/live benchmark eval regressed to
`65.6%`. Phase 2.6 found that oracle-shabad/live0 alignment improves to `87.4%`
(x4/v5 oracle baseline: `85.2%`) and a v3.2-ID-lock proxy scores `87.1%`, so
the problem moved to runtime ID-lock/alignment rather than "can the Mac train?"

That runtime work has now produced `phase2_9_loop_align` at `91.2%` paired and
Phase 2.13 evidence fusion with `5/5` assisted-OOS locks, `59.9%` assisted-OOS
frame accuracy, and `84.1%` paired under the opt-in fusion policy. The remaining
lock failure is full-start `zOtIpxMT9hU -> 4892`; tail-window overfits can fix it
on paired but hurt OOS, so they are rejected.

Therefore the next training step was a **controlled Phase 3 warm-start**, not
all-300h training:

```bash
make data-v6-scale20
make train-v6-scale20
```

`data-v6-scale20` pulls a large fresh slice from shards `20-49`, preserving
shards `0-9` for prior v5b history and `10-19` for silver evaluation. It keeps
`min_score >= 0.88` and requires at least 40 source videos and 300 shabad tokens
before the pull is considered valid. Inspect
`training_data/v6_mac_scale20/data_card.md` before training.
The scientific gates and decision table live in
[`phase3_warm_start_plan.md`](phase3_warm_start_plan.md).

Current Phase 3 warm-start checkpoint, 2026-05-17:

- data: `12,216` clips, `24.593 h`, `524` shabad tokens, `40` source videos,
  diversity gate `PASS`
- adapter: `lora_adapters/v6_mac_scale20/`
- training: `4,581` optimizer steps, `3.0` epochs, `3 h 46 m` wall-clock on MPS
- memory: `27.24 GB` peak MPS driver memory on the 48 GB M4 Pro
- losses: final logged train loss `0.028`; trainer mean `train_loss=0.1272`

This confirms the 48 GB M4 Pro is being used correctly for the approved large
warm-start. The low train loss is not an accuracy claim. The required next step
is held-out silver evaluation before paired benchmark promotion.

## 2. Smoke-test the pipeline (~5 minutes)

Before committing a multi-hour training run, verify the script gets through 20 steps:

```bash
python scripts/finetune_path_b.py \
  --config configs/training/surt_lora_mac.yaml \
  --manifest training_data/smoke/manifest.json \
  --output-dir /tmp/surt_smoke \
  --max-steps 20 --batch-size 1
```

The smoke manifest (`scripts/build_smoke_manifest.py`) is intentionally tiny (4 snippets, 5 seconds each). The resulting adapter is contaminated with benchmark audio — **do not score it against the benchmark**.

If this completes, the pipeline is green.

## 3. Run the real fine-tune

```bash
python scripts/finetune_path_b.py \
  --config configs/training/surt_lora_mac.yaml \
  --manifest training_data/kirtan_v1/manifest.json \
  --output-dir lora_adapters/surt_mac_v1
```

The config (`configs/training/surt_lora_mac.yaml`) supplies:

- `model_id = surindersinghssj/surt-small-v3`
- `lora_r = 16`, `lora_alpha = 32`, target_modules covering Whisper attention projections
- `batch_size = 4`, `grad_accum = 2` (effective batch 8)
- `epochs = 3`, `lr = 1e-5`, `warmup_ratio = 0.1`
- `fp16: false` on the current Mac stack. Torch 2.5 + accelerate <1.11 is the known-good window; re-enable MPS fp16 only after upgrading torch to 2.8+ and revalidating torchaudio / torchcodec / ctranslate2 compatibility.

Any flag passed on the CLI overrides the YAML value.

**Wall-clock estimate** for a 50-hour dataset, 3 epochs, batch 8:
- M4 Pro at 2-3 steps/sec: 6-12 hours per epoch, 18-36 hours total.
- Faster if you raise `batch_size` to 8 and drop `grad_accum` to 1; slower if MPS hits memory pressure.

## 4. Evaluate

Use the LoRA adapter directly via `scripts/run_path_a.py` with `--backend huggingface_whisper`. Keep `HF_WINDOW_SECONDS=10` for fair comparison to `x4_pathA_surt`; 30-second windows are known-bad for this backend.

```bash
HF_WINDOW_SECONDS=10 \
python scripts/run_path_a.py \
  --backend huggingface_whisper \
  --model surindersinghssj/surt-small-v3 \
  --adapter-dir lora_adapters/surt_mac_v1 \
  --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 \
  --stay-bias 6 --blind --blind-aggregate chunk_vote \
  --blind-lookback 30 --live --tentative-emit \
  --out-dir submissions/v5_surt_mac_v1
```

Score on the benchmark:

```bash
python ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/v5_surt_mac_v1/ \
  --gt   ../live-gurbani-captioning-benchmark-v1/test/
```

**Important:** the benchmark is 4 shabads. A score here only validates that the fine-tune didn't *regress* against Path A v3.2 (86.5%). The honest measurement is `scripts/eval_oos.py` against shabads outside the benchmark (planned, M2).

## 5. Iterate

- Loss not decreasing → check `--lr` (try 5e-6), check `--warmup-steps` (try 50).
- OOM → drop `batch_size` to 2 or 1, raise `grad_accum` accordingly.
- Throughput too slow → measure steps/sec from the trainer logs. Current validated MPS mode is fp32 and prints `Precision: bf16=False, fp16=False`. If under 0.5/s, check for CPU fallback, memory pressure, or a dependency drift from the known-good versions below.
- Catastrophic regression on one shabad → likely overfitting to a narrow training distribution. Pull more diverse data (raise `--num-samples`, include sehaj source).

## 6. When to escape to cloud

If wall-clock pushes past ~24h for a single training run, or you want to sweep hyperparameters in parallel, consider Colab Pro+ with an A100 or L4. The same script and config work on CUDA — `--bf16` flips on automatically, `--fp16` flips off. Move `requirements-cloud.txt` instead of `requirements-mac.txt`. See [`docs/cloud_training.md`](cloud_training.md) for the cloud recipe.

## Known-good environment

| Component | Version |
|---|---|
| macOS | 14.0+ |
| Python | 3.12.13 |
| torch | 2.5.0 |
| transformers | 4.46.3 |
| peft | 0.19.1 |
| accelerate | 1.10.1 |

If you upgrade these, rerun `make smoke`, the same-seed reproducibility gate, and one benchmark eval before trusting a real training run.

## Troubleshooting

- **"unable to load native MPS backend"** → upgrade to PyTorch 2.4+. The MPS backend significantly improved in 2.3 and 2.4.
- **Training crashes after a few steps with `MPSNDArray` errors** → set `PYTORCH_ENABLE_MPS_FALLBACK=1`. A handful of ops still fall back to CPU; the env var lets them silently do so instead of failing.
- **Training step time spikes after a save checkpoint** → the unified memory pool is filling. Lower `save_steps` so checkpoints land less often, or reboot between training runs.
