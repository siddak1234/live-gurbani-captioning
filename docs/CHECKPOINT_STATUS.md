# Pause checkpoint — 2026-05-18

## v7 training paused

- adapter dir: lora_adapters/v7_mac_300h_epoch1
- latest saved checkpoint: checkpoint-9000
- last logged train step: 9000
- last logged train loss: 0.0608
- wall-clock so far: 11:59:39
- status from run_card.json: interrupted

## To resume training

```bash
cd ~/Desktop/live-gurbani-captioning
source .venv/bin/activate
git pull origin main
python scripts/finetune_path_b.py \
  --config configs/training/surt_lora_mac.yaml \
  --manifest training_data/v7_mac_300h/manifest_train.json \
  --eval-manifest training_data/v7_mac_300h/manifest_val.json \
  --output-dir lora_adapters/v7_mac_300h_epoch1 \
  --epochs 1 \
  --eval-strategy steps --eval-steps 1000 --save-steps 1000 \
  --load-best-model-at-end \
  --resume-from-checkpoint lora_adapters/v7_mac_300h_epoch1/checkpoint-9000
```

## In-flight workstreams (state-of-the-world)

- Phase 2.9 best honest runtime: phase2_9_loop_align @ 91.2%
- OOS v1 status: drafts seeded; all five `eval_data/oos_v1/test/case_*.json`
  files are preserved with `curation_status="NEEDS_HUMAN_CORRECTION"`
- Phase 2.10 silver: completed; silver diagnostics supported the confirmed
  runtime path and v6/v7 acoustic-scaling gate, but promotion still requires
  paired + assisted-OOS runtime scoring
- Pending decisions:
  - Resume v7 epoch-1 from `checkpoint-9000` and finish the remaining steps.
  - Evaluate final/best v7 through confirmed runtime gates.
  - Promote only if paired beats `92.8%`, assisted-OOS beats `60.8%`, and locks
    stay `12/12` paired plus `5/5` assisted-OOS.
  - If val loss improves but runtime metrics do not, return to line
    alignment/candidate-resolution diagnostics rather than blindly extending
    training.

## Commits added during this pause

3e8c8a2 wip(oos): preserve machine-seeded GT working files at pause checkpoint

