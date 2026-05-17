# M4 Pro compute utilization audit

## Hardware and runtime

| Item | Value |
|---|---|
| Model | MacBook Pro |
| Chip | Apple M4 Pro |
| Unified memory | 48 GB |
| Torch | 2.5.0 |
| MPS built | True |
| MPS available in this process | True |
| MPS smoke ok | True |

## Completed training runs

| Adapter | Clips | Loss | Wall clock | Peak MPS memory | Device |
|---|---:|---:|---:|---:|---|
| `v5_mac_baseline` | 200 | 0.6465 | 3.9 min | 22.34 GB (mps_driver) | mps |
| `v5b_mac_diverse` | 2544 | 0.2486 | 47.8 min | 27.05 GB (mps_driver) | mps |

## Data and artifact footprint

| Path | Size |
|---|---:|
| `training_data/` | 2.4G |
| `lora_adapters/` | 153M |
| `submissions/` | 3.7M |
| `asr_cache/` | 200K |

## Audit decision

- Highest completed training memory use was 27.05 GB, about 56.4% of 48 GB unified memory.
- The M4 Pro is being used correctly for the training work we have actually approved: PyTorch MPS, not CPU.
- We are not currently compute-bound. The current blocker is still validation
  quality, but Phase 2.13's UEM-aware evidence-fusion result is strong enough
  to justify one controlled Phase 3 warm-start data/training run.
- The 48 GB headroom is useful for the future Phase 3 plan (larger batches, gradient checkpointing experiments, longer runs), but Phase 3 is intentionally gated.
- Do not pull/train on all 300h right now. The silver audit found label-risk
  rows, not clean ASR failures, and `v5b_mac_diverse` was neutral/regressive
  outside oracle alignment.
- Next recommended compute use: `make data-v6-scale20`, validate the data card,
  then `make train-v6-scale20` if diversity/holdout gates pass. This is a
  warm-start scaling experiment, not a promotion claim.

## If Phase 3 is unblocked later

- Re-enable MPS fp16 only after a torch >= 2.8 / accelerate >= 1.11 compatibility pass.
- Verify `gradient_checkpointing=true` with PEFT+MPS in isolation before changing the main YAML.
- Use the 48 GB machine for 50h/3-seed runs only after OOS v1 passes or a
  deliberate pivot is documented.
