# Pause checkpoint — 2026-05-19

## v7 training paused

- adapter dir: `lora_adapters/v7_mac_300h_epoch1`
- latest saved checkpoint: `checkpoint-19000`
- best validated checkpoint: `checkpoint-19000`
- last saved train/eval step: `19000`
- last observed unsaved training step before SIGTERM: about `19530`
- last logged train loss before SIGTERM: `0.04` at about step `19530`
- best validation loss: `0.03500748425722122` at step `19000`
- wall-clock so far for the continuation run: about `11:40:04`
- status from `run_card.json`: `completed` (non-crashed; this file is the epoch-1 card, mtime `2026-05-18 21:20:59`)
- stop method: `SIGTERM` to trainer PID `52672`; `make` exited with `Terminated: 15`

The continuation was stopped safely after the step-19000 checkpoint and before
step 20000. Progress from roughly steps 19001-19530 was not checkpointed and is
expected to be replayed when resuming. No durable checkpoint data was deleted.

## To resume training

Use the repo-local interpreter explicitly; this folder lives under
`~/Desktop/Personal Project/live-gurbani-captioning`.

```bash
cd "$HOME/Desktop/Personal Project/live-gurbani-captioning"
git pull --ff-only origin main
export PYTHON=.venv/bin/python3
make train-v7-300h \
  PHASE3_EPOCHS=3 \
  PHASE3_RESUME=lora_adapters/v7_mac_300h_epoch1/checkpoint-19000
```

This resumes from the last durable optimizer checkpoint. Expected behavior:
training restarts from step 19000 and continues toward total step 34983.

## If you want to score before resuming

The best adapter currently lives at `checkpoint-19000`. To measure whether this
new validation best improves runtime metrics before spending more compute:

```bash
cd "$HOME/Desktop/Personal Project/live-gurbani-captioning"
git pull --ff-only origin main

export PYTHON=.venv/bin/python3
make eval-paired-recency-guard-confirmed-v6 \
  CONFIRMED_ADAPTER_DIR=lora_adapters/v7_mac_300h_epoch1 \
  CONFIRMED_PAIRED_OUT=submissions/phase3_confirmed_v7_300h_ckpt19000_paired

make eval-oos-recency-guard-confirmed-v6-assisted \
  CONFIRMED_ADAPTER_DIR=lora_adapters/v7_mac_300h_epoch1 \
  CONFIRMED_OOS_OUT=submissions/oos_v1_assisted_phase3_confirmed_v7_300h_ckpt19000
```

## In-flight workstreams (state-of-the-world)

- Phase 3 v7 continuation: paused safely at `checkpoint-19000`.
- Validation trend during continuation:
  - step 15000: `0.03506891429424286`
  - step 16000: `0.035056933760643005`
  - step 17000: `0.03505650907754898`
  - step 18000: `0.03503436595201492`
  - step 19000: `0.03500748425722122` (current best)
- Prior runtime gate from v7 epoch-1: paired `93.1%`, assisted-OOS `61.3%`.
- OOS v1 status: drafts preserved; all five
  `eval_data/oos_v1/test/case_*.json` files remain committed with
  `curation_status="NEEDS_HUMAN_CORRECTION"`.
- Phase 2.10 silver: completed; diagnostics supported the confirmed runtime
  path, but promotion still requires paired + assisted-OOS runtime scoring.
- Pending decisions:
  - Resume from `checkpoint-19000` to finish the controlled 3-epoch run, or
    score `checkpoint-19000` first if you want a runtime check before more compute.
  - Promote only if paired/OOS runtime metrics improve and locks stay stable.
  - Replace machine-assisted OOS with gold-corrected OOS before any public 95%+
    generalization claim.

## Commits added during this monitoring/pause window

- `d670b0e` `docs(phase3): record v7 checkpoint-13000 watch point`
- `88cf3fc` `docs(phase3): record v7 checkpoint-14000 recovery`
- `6daab4a` `docs(phase3): record v7 checkpoint-15000 new best`
- `4b879dc` `Merge remote-tracking branch 'origin/main'`
- `640e398` `docs(phase3): record v7 checkpoints 16000-19000 trend`
- this checkpoint-status update commit records the safe stop at `checkpoint-19000`.

## What to tell Codex next time

```text
Open ~/Desktop/Personal Project/live-gurbani-captioning. Read docs/CHECKPOINT_STATUS.md and docs/phase3_300h_runbook.md. We safely stopped v7 training at checkpoint-19000 after SIGTERM. Verify git status is clean, verify lora_adapters/v7_mac_300h_epoch1/checkpoint-19000 exists, then either resume training from checkpoint-19000 or score checkpoint-19000 first. Do not delete caches or adapter/training_data directories.
```
