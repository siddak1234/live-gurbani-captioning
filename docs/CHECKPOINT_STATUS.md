# Checkpoint status — 2026-05-18

## v7 training completed

- adapter dir: `lora_adapters/v7_mac_300h_epoch1`
- final saved checkpoint: `checkpoint-11661`
- best validated checkpoint: `checkpoint-11000`
- last logged train step: `11661`
- last logged train loss: `0.0214`
- best validation loss: `0.03510706499218941` at step `11000`
- final-resume wall-clock: `3:20:45`
- total known wall-clock through pause + resume: about `15:20:24`
- status from `run_card.json`: `completed`
- train/eval clips: `93,292` train, `11,541` eval
- peak MPS memory: `35.53 GB`

## Result interpretation

The resumed epoch finished cleanly from `checkpoint-9000` and improved the
held-out validation curve slightly:

```text
checkpoint-9000   eval_loss=0.03512399643659592
checkpoint-10000  eval_loss=0.03511273115873337
checkpoint-11000  eval_loss=0.03510706499218941  <-- best
```

The improvement is real but very small. The next expert decision cannot come
from acoustic validation loss alone. The correct next gate is runtime scoring
through the confirmed paired + assisted-OOS paths.

## To score the best v7 adapter

This folder was moved to `~/Desktop/Personal Project/live-gurbani-captioning`.
Use the repo-local interpreter explicitly, because `.venv/bin/activate` still
contains the old path from before the folder move.

```bash
cd "$HOME/Desktop/Personal Project/live-gurbani-captioning"
git pull --ff-only origin main

PYTHON=.venv/bin/python3 make eval-paired-recency-guard-confirmed-v6 \
  CONFIRMED_ADAPTER_DIR=lora_adapters/v7_mac_300h_epoch1 \
  CONFIRMED_PAIRED_OUT=submissions/phase3_confirmed_v7_300h_paired

PYTHON=.venv/bin/python3 make eval-oos-recency-guard-confirmed-v6-assisted \
  CONFIRMED_ADAPTER_DIR=lora_adapters/v7_mac_300h_epoch1 \
  CONFIRMED_OOS_OUT=submissions/oos_v1_assisted_phase3_confirmed_v7_300h
```


## Runtime scoring result — 2026-05-18

The best v7 adapter cleared both current confirmed-runtime gates:

- Paired benchmark: `93.1%` (3190/3425), `12/12` locks.
- Assisted-OOS diagnostic: `61.3%` (539/880), `5/5` locks.
- Prior gates were `92.8%` paired and `60.8%` assisted-OOS.

This is a positive acoustic-scaling signal, but the OOS margin is only `+0.5`
points and the OOS labels are still machine-assisted. Treat this as permission
to continue controlled 300h training, not as a production accuracy claim.

## To continue v7 to 3 epochs

Resume from the final optimizer checkpoint, not the best scoring adapter:

```bash
cd "$HOME/Desktop/Personal Project/live-gurbani-captioning"
git pull --ff-only origin main
PYTHON=.venv/bin/python3 make train-v7-300h \
  PHASE3_EPOCHS=3 \
  PHASE3_RESUME=lora_adapters/v7_mac_300h_epoch1/checkpoint-11661
```

After completion, rerun the paired + assisted-OOS commands above and compare
against `93.1%` / `61.3%`.


## Active continuation run — 2026-05-18

The controlled continuation to 3 epochs has been started in a detached `screen`
session so it can survive closing VS Code:

- screen session: `v7_epoch23`
- launch PID observed: `52656` (`SCREEN`), trainer PID observed: `52672`
- log file: `/tmp/phase3_v7_epochs2_3.log`
- resume checkpoint: `lora_adapters/v7_mac_300h_epoch1/checkpoint-11661`
- target total steps for 3 epochs: `34983`

Useful checks:

```bash
screen -ls
sed -n '1,180p' /tmp/phase3_v7_epochs2_3.log
pgrep -fl 'scripts/finetune_path_b.py|v7_epoch23'
```

To attach interactively:

```bash
screen -r v7_epoch23
```

Detach again with `Ctrl-a` then `d`.

## In-flight workstreams (state-of-the-world)

- Phase 2.9 best honest runtime: `phase2_9_loop_align` at `91.2%`.
- Current confirmed-runtime gate before promotion:
  - paired must beat `92.8%`;
  - assisted-OOS must beat `60.8%`;
  - locks remain `12/12` paired and `5/5` assisted-OOS.
- OOS v1 status: drafts are preserved; all five
  `eval_data/oos_v1/test/case_*.json` files were committed with
  `curation_status="NEEDS_HUMAN_CORRECTION"`.
- Phase 2.10 silver: completed; diagnostics supported the confirmed runtime
  path and the v6/v7 acoustic-scaling gate, but promotion still requires paired
  + assisted-OOS runtime scoring.
- Pending decisions:
  - Continue v7 from `checkpoint-11661` toward 3 epochs because both runtime
    gates moved up and locks stayed stable.
  - Re-score after the continuation; promote only if paired/OOS improve again
    and no case regresses catastrophically.
  - Replace machine-assisted OOS with gold-corrected OOS before any public
    95%+ generalization claim.

## Commits added during the prior pause

`3e8c8a2` `wip(oos): preserve machine-seeded GT working files at pause checkpoint`
