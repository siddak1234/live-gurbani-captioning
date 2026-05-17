# M4 Pro compute utilization audit

## Hardware and runtime

| Item | Value |
|---|---|
| Model | MacBook Pro |
| Chip | Apple M4 Pro |
| Unified memory | 48 GB |
| Torch | 2.5.0 |
| MPS built | True |
| MPS available in this process | False |
| MPS smoke ok | False |

## Completed training runs

| Adapter | Clips | Loss | Wall clock | Peak MPS memory | Device |
|---|---:|---:|---:|---:|---|
| `v5_mac_baseline` | 200 | 0.6465 | 3.9 min | 22.34 GB (mps_driver) | mps |
| `v5b_mac_diverse` | 2544 | 0.2486 | 47.8 min | 27.05 GB (mps_driver) | mps |
| `v6_mac_scale20` | 12216 | 0.0280 | 226.1 min | 27.24 GB (mps_driver) | mps |

## Data and artifact footprint

| Path | Size |
|---|---:|
| `training_data/` | 5.1G |
| `lora_adapters/` | 577M |
| `submissions/` | 4.0M |
| `asr_cache/` | 308K |

## Audit decision

- Highest completed training memory use was 27.24 GB, about 56.7% of 48 GB unified memory.
- The M4 Pro is being used correctly for the training work we have actually approved: PyTorch MPS, not CPU.
- The controlled Phase 3 warm-start completed and passed the silver non-regression gate modestly.
- A generic recency-consistency guarded fusion runtime lifted paired accuracy to 91.0% / 12-of-12 locks without assisted-OOS regression.
- Assisted-OOS remains flat at 59.9% despite 5-of-5 locks, so the active blocker is line timing/alignment under the correct shabad, not M4 Pro capacity.
- The 48 GB headroom remains useful for future larger batches, gradient checkpointing experiments, and longer runs, but full 300h / multi-seed training is not justified until OOS alignment improves or diagnostics prove true ASR misses.
- Do not pull/train on all 300h right now. The next valid experiment is OOS/paired line-alignment error analysis under the recency-guarded runtime.
- Next recommended compute use: cached-output alignment diagnostics on `submissions/phase3_recency_guard_paired` and `submissions/oos_v1_assisted_phase3_recency_guard`.

## If Phase 3 is unblocked later

- Re-enable MPS fp16 only after a torch >= 2.8 / accelerate >= 1.11 compatibility pass.
- Verify `gradient_checkpointing=true` with PEFT+MPS in isolation before changing the main YAML.
- Use the 48 GB machine for full-slice or multi-seed runs only after silver/OOS gates pass or a deliberate pivot is documented.
