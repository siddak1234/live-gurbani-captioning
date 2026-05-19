# Phase 3 large 300h runbook

This is the next acoustic-scaling step after the Phase 3 warm-start and
confirmed loop-align runtime checkpoint.

The 300h run is not a blind bet. It is now justified as a controlled
acoustic-scaling experiment because:

- `v6_mac_scale20` proved the M4 Pro training stack works at 24.6 h scale;
- recency-guarded shabad lock fixed the remaining paired false lock;
- confirmed loop-align moved paired `91.0% -> 92.8%` and assisted-OOS
  `59.9% -> 60.8%` without more training;
- a small confirmed-loop parameter sweep plateaued, so the next likely source
  of gain is better acoustic evidence, not another local smoother knob.

## Commands

Pull the large filtered slice from the full canonical 300h dataset:

```bash
make data-v7-300h
```

If Hugging Face/Xet stalls on a `.incomplete` blob with no clip-count growth,
restart the pull through the regular HTTP path:

```bash
HF_HUB_DISABLE_XET=1 make data-v7-300h
```

That workaround was required on 2026-05-17: the first attempt hung on shard 81
with a CloudFront socket in `CLOSE_WAIT`; the HTTP-path restart completed.

This writes:

- `training_data/v7_mac_300h/manifest_train.json`
- `training_data/v7_mac_300h/manifest_val.json`
- `training_data/v7_mac_300h/manifest_test.json`
- `training_data/v7_mac_300h/data_card.md`

Then run the first large adapter as **one epoch**, not three:

```bash
make train-v7-300h-epoch1
```

The one-epoch run is deliberate. It checks whether the full data scale improves
held-out eval loss and runtime behavior before spending the full multi-day /
multi-seed budget.

If a long run is interrupted after a checkpoint, resume explicitly rather than
starting over:

```bash
make train-v7-300h-epoch1 \
  PHASE3_RESUME=lora_adapters/v7_mac_300h_epoch1/checkpoint-1000
```

The resume path restores model, optimizer, scheduler, RNG, and trainer state.
Use the latest complete `checkpoint-*` directory.

## Expected cost

Based on the v6 run:

- v6: 24.6 h training audio, 3 epochs, 3.8 h wall-clock, 27.2 GB peak MPS memory.
- v7 pull result (2026-05-17): 116,246 clean clips, 235.96 h total, 188.61 h
  train, 23.79 h val, 23.57 h test. The target name remains `v7_mac_300h`
  because it scans the full canonical 300h source, but after quality/duration
  filters the usable training split is ~189 h.
- v7 epoch-1 estimate: roughly 8-12 h wall-clock on M4 Pro for the 188.61 h
  train split, extrapolating from the v6 wall-clock and allowing overhead for
  validation/checkpointing.
- Disk: 35-60 GB for clips plus parquet cache is expected and acceptable on
  this machine (`df -h` showed >700 GB free at checkpoint time).

## Gates before training

After `make data-v7-300h`, inspect `data_card.md`:

- holdout shabads/videos/content must be enforced;
- shabad-level split must exist;
- unique videos >= 100;
- unique shabad tokens >= 1000;
- no OOS audio is used for training.

If the data card fails, do not train. Adjust shard range / score threshold /
diversity floors and re-pull.

The 2026-05-17 v7 pull passed the pre-training gates:

- holdout counters: `holdout_shabad=0`, `holdout_video=0`,
  `holdout_content=0`;
- diversity: 296 videos, 1,751 shabad tokens;
- train/val/test split: `split_by=shabad`, 0 shabad overlap across splits;
- score floor: min score 0.800, split mean scores ~0.96;
- `manifest.json` is identical to `manifest_train.json` for back-compat.

## 2026-05-17 training checkpoint

The first v7 epoch-1 attempt reached step 1000 and completed the full
validation sweep over 11,541 held-out clips, then crashed while saving the
checkpoint:

- train loss fell from roughly `1.16` to the `0.25-0.30` range by step 1000;
- validation ran to completion in `1996 s` (`5.78 samples/s`, `0.72 steps/s`);
- peak MPS driver memory reported by `run_card.json`: `38.82 GB`;
- `checkpoint-1000` was written successfully;
- failure cause: Transformers could not find `eval_loss` while
  `--load-best-model-at-end` requested `metric_for_best_model=eval_loss`.

Root cause: PEFT wraps Whisper with a generic forward signature, so
Transformers did not infer that `labels` is the evaluation label column. The
trainer still learned normally, but evaluation emitted runtime-only metrics.

Fix landed in `scripts/finetune_path_b.py`: pass `label_names=["labels"]` to
`Seq2SeqTrainingArguments` for both CTC and Whisper paths, and add explicit
`--resume-from-checkpoint` support. A 1-step Whisper smoke run with
`--eval-manifest`, `--save-steps 1`, and `--load-best-model-at-end` verified
that `eval_loss` is now emitted (`eval_loss=1.9093`) and checkpoint ranking no
longer crashes.

Resume command for this run:

```bash
make train-v7-300h-epoch1 \
  PHASE3_RESUME=lora_adapters/v7_mac_300h_epoch1/checkpoint-1000
```

Operational note: the crash happened after the step-1000 adapter/optimizer/
scheduler/RNG files were saved, but before `trainer_state.json` existed. For
this one recovery, a minimal `trainer_state.json` was reconstructed with
`global_step=1000` and no best checkpoint, then the run resumed from
`checkpoint-1000`. This is acceptable for epoch-1 continuation because the model
weights, optimizer state, scheduler state, RNG state, and training arguments were
present; it only means best-checkpoint ranking starts from the first successful
post-fix eval.

Live recovery validation: the resumed run reached step 2000 on 2026-05-17,
completed the full validation sweep, emitted `eval_loss=0.036197841`, saved a
complete `checkpoint-2000/trainer_state.json`, set
`best_model_checkpoint=lora_adapters/v7_mac_300h_epoch1/checkpoint-2000`, and
continued training past step 2000. That proves the PEFT eval-loss fix on the
real v7 run, not just the 1-step smoke.

Step-3000 validation then emitted `eval_loss=0.035645694`, saved a complete
`checkpoint-3000/trainer_state.json`, and moved `best_model_checkpoint` to
`checkpoint-3000`. Early held-out loss is therefore still improving during the
large one-epoch run (`0.03620 -> 0.03565`) rather than showing immediate
overfit drift.

Step-4000 validation emitted `eval_loss=0.035433434` with
`eval_runtime=2103.088s`, `eval_samples_per_second=5.488`, and
`eval_steps_per_second=0.686`. The run resumed after validation and continued
training past step 4000. This keeps the trend positive
(`0.03620 -> 0.03565 -> 0.03543`), so the recommended action remains: let the
epoch finish, then score the final/best v7 adapter on paired benchmark plus
assisted OOS before promoting or changing architecture.

Step-5000 validation emitted `eval_loss=0.035315320` with
`eval_runtime=2107.6419s`, `eval_samples_per_second=5.476`, and
`eval_steps_per_second=0.685`. The improvement is smaller than the earlier
steps but still positive (`0.03620 -> 0.03565 -> 0.03543 -> 0.03532`), and the
run continued training past step 5000. This is a normal late-epoch flattening
pattern, not an overfit signal. Continue the epoch; do not change architecture
until paired + assisted-OOS scoring says the ASR improvement is failing to move
the line-alignment runtime.

Step-6000 validation emitted `eval_loss=0.035235524` with
`eval_runtime=2104.6417s`, `eval_samples_per_second=5.484`, and
`eval_steps_per_second=0.686`. The slope continues to flatten, but it is still
moving in the right direction (`0.03620 -> 0.03565 -> 0.03543 -> 0.03532 ->
0.03524`). This keeps the current plan valid: finish epoch 1, then evaluate the
best/final v7 adapter against paired benchmark and assisted OOS before deciding
whether to promote, continue to multi-epoch 300h training, or diagnose runtime
bottlenecks.

Step-7000 validation emitted `eval_loss=0.035162665` with
`eval_runtime=2095.2946s`, `eval_samples_per_second=5.508`, and
`eval_steps_per_second=0.689`.

Step-8000 validation emitted `eval_loss=0.035139494` with
`eval_runtime=2098.0578s`, `eval_samples_per_second=5.501`, and
`eval_steps_per_second=0.688`. The curve is now nearly flat but still improving
(`0.03620 -> 0.03565 -> 0.03543 -> 0.03532 -> 0.03524 -> 0.03516 ->
0.03514`). This does **not** justify stopping early: there is no overfit signal
yet, and the remaining compute is cheaper than introducing a new variable
mid-run. It does mean the next decision must come from paired + assisted-OOS
runtime metrics. If those metrics do not move, the bottleneck is likely
alignment/runtime behavior rather than acoustic loss alone.

Step-9000 validation emitted `eval_loss=0.035123996` with
`eval_runtime=2134.5091s`, `eval_samples_per_second=5.407`, and
`eval_steps_per_second=0.676`. `trainer_state.json` moved
`best_model_checkpoint` to `checkpoint-9000`.

The 2026-05-18 resume from `checkpoint-9000` finished the epoch cleanly:

- step 10000: `eval_loss=0.035112731`, `eval_runtime=2004.4807s`,
  `eval_samples_per_second=5.758`, `eval_steps_per_second=0.720`;
- step 11000: `eval_loss=0.035107065`, `eval_runtime=2016.0926s`,
  `eval_samples_per_second=5.724`, `eval_steps_per_second=0.716`;
- final step: `11661`, `run_card.status=completed`, final logged train loss
  `0.0214`, peak MPS memory `35.53 GB`;
- `best_model_checkpoint` is now
  `lora_adapters/v7_mac_300h_epoch1/checkpoint-11000`.

The validation curve stayed monotonic but essentially flat (`0.03620 -> 0.03565
-> 0.03543 -> 0.03532 -> 0.03524 -> 0.03516 -> 0.03514 -> 0.03512 ->
0.03511 -> 0.035107`). This is useful evidence that acoustic scaling did not
hurt held-out loss, but it is not a promotion signal by itself. The key
experimental question is now entirely runtime-facing: does the lower acoustic
loss move the confirmed paired/OOS metrics toward the 95% target?

## Gates after training

Evaluate the adapter through the current confirmed runtime, not the older
runtime:

```bash
make eval-paired-recency-guard-confirmed-v6 \
  CONFIRMED_ADAPTER_DIR=lora_adapters/v7_mac_300h_epoch1 \
  CONFIRMED_PAIRED_OUT=submissions/phase3_confirmed_v7_300h_paired

make eval-oos-recency-guard-confirmed-v6-assisted \
  CONFIRMED_ADAPTER_DIR=lora_adapters/v7_mac_300h_epoch1 \
  CONFIRMED_OOS_OUT=submissions/oos_v1_assisted_phase3_confirmed_v7_300h
```

The Make targets default to the current v6 confirmed checkpoint, so the
`CONFIRMED_ADAPTER_DIR` override is required when scoring the v7 adapter. The
metric gate is:

- paired must beat `92.8%`;
- assisted-OOS must beat `60.8%`;
- locks remain `12/12` paired and `5/5` assisted-OOS;
- no catastrophic case falls below the current confirmed-runtime floor.

## Decision after epoch 1

- If paired and assisted-OOS both move up: run the full 3-epoch 300h training
  and then a seed-variance check.
- If silver/val loss improves but runtime metrics do not: the adapter helps ASR
  but the line-path runtime is still the bottleneck.
- If val loss and runtime both stall: stop scaling on Mac and pivot to the next
  architecture bet before spending cloud/300h budget.
