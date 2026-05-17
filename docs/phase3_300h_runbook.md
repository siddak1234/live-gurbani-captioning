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

## Expected cost

Based on the v6 run:

- v6: 24.6 h training audio, 3 epochs, 3.8 h wall-clock, 27.2 GB peak MPS memory.
- v7 epoch-1 estimate: roughly 12-18 h wall-clock if the filtered pull lands
  near the full 300h scale.
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
