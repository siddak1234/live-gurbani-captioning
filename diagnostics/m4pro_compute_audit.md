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

## Completed / in-progress training runs

| Adapter | Clips | Loss | Wall clock | Peak MPS memory | Device |
|---|---:|---:|---:|---:|---|
| `v5_mac_baseline` | 200 | 0.6465 | 3.9 min | 22.34 GB (mps_driver) | mps |
| `v5b_mac_diverse` | 2544 | 0.2486 | 47.8 min | 27.05 GB (mps_driver) | mps |
| `v6_mac_scale20` | 12216 | 0.0280 | 226.1 min | 27.24 GB (mps_driver) | mps |
| `v7_mac_300h_epoch1` | 93292 | in progress | in progress (latest checkpoint 8000) | 1.38 GB (mps_driver) | mps |

## Data and artifact footprint

| Path | Size |
|---|---:|
| `training_data/` | 31G |
| `lora_adapters/` | 904M |
| `submissions/` | 4.1M |
| `asr_cache/` | 308K |

## Audit decision

- Highest completed training memory use was 27.24 GB, about 56.7% of 48 GB unified memory.
- The M4 Pro is being used correctly for the training work we have actually approved: PyTorch MPS, not CPU.
- The confirmed loop-align runtime reached a local plateau at 92.8% paired / 60.8% assisted-OOS.
- That plateau justified the current controlled v7 300h-source epoch-1 run; this is no longer an underuse-of-M4 question.
- v7 epoch 1 is currently in progress with complete trainer state through checkpoint 8000.
- The interrupted first v7 attempt recorded ~38.82 GB peak MPS driver memory at step 1000, so this workload is using most of the 48 GB M4 Pro envelope.
- The correct action is to finish epoch 1, then evaluate the final/best v7 adapter through the confirmed paired + assisted-OOS gates.
- Do not automatically expand to 3 epochs or multiple seeds. Promotion requires beating 92.8% paired and 60.8% assisted-OOS with locks preserved.
- If held-out loss improves but runtime metrics do not, the remaining bottleneck is line alignment / candidate resolution, not acoustic capacity or M4 Pro underuse.

## If Phase 3 is unblocked later

- Re-enable MPS fp16 only after a torch >= 2.8 / accelerate >= 1.11 compatibility pass.
- Verify `gradient_checkpointing=true` with PEFT+MPS in isolation before changing the main YAML.
- Use the 48 GB machine for full-slice or multi-seed runs only after silver/OOS gates pass or a deliberate pivot is documented.
