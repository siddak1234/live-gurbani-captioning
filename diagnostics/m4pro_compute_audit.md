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
| `v6_mac_scale20` | 12216 | 0.0280 | 226.1 min | 27.24 GB (mps_driver) | mps |

## Data and artifact footprint

| Path | Size |
|---|---:|
| `training_data/` | 5.1G |
| `lora_adapters/` | 577M |
| `submissions/` | 3.8M |
| `asr_cache/` | 260K |

## Audit decision

- Highest completed training memory use was 27.24 GB, about 56.7% of 48 GB unified memory.
- The M4 Pro is being used correctly for the training work we have actually approved: PyTorch MPS, not CPU.
- The controlled Phase 3 warm-start completed. We are not currently compute-bound; the active blocker is held-out validation quality for `v6_mac_scale20`.
- The 48 GB headroom remains useful for future larger batches, gradient checkpointing experiments, and longer runs, but more training is not justified until silver/OOS gates are checked.
- Do not pull/train on all 300h right now. The next valid experiment is held-out silver evaluation of `lora_adapters/v6_mac_scale20` against base `surt-small-v3` and `v5b_mac_diverse`.
- Next recommended compute use: `make eval-silver-300h SILVER_ADAPTER_DIR=lora_adapters/v6_mac_scale20 SILVER_OUT=submissions/silver_300h_v6_mac_scale20.json`.

## If Phase 3 is unblocked later

- Re-enable MPS fp16 only after a torch >= 2.8 / accelerate >= 1.11 compatibility pass.
- Verify `gradient_checkpointing=true` with PEFT+MPS in isolation before changing the main YAML.
- Use the 48 GB machine for full-slice or multi-seed runs only after silver/OOS gates pass or a deliberate pivot is documented.
