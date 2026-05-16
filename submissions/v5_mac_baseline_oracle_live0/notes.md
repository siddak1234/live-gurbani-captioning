# v5_mac_baseline_oracle_live0 — Phase 2.6 oracle baseline

**Decision:** diagnostic only. Not deployable because the GT shabad ID is provided and `blind_lookback=0`.

Overall: **85.2%** frame accuracy (`2918/3425`), collar `1s`.

This matches `x4_pathA_surt_oracle_live0`, confirming the 200-clip `v5_mac_baseline` adapter was neutral even when routing is fixed. The larger/diverse `v5b_mac_diverse` adapter is the first one that changes oracle alignment meaningfully.

Command:

```bash
HF_WINDOW_SECONDS=10 python3 scripts/run_path_a.py \
  --backend huggingface_whisper \
  --model surindersinghssj/surt-small-v3 \
  --adapter-dir lora_adapters/v5_mac_baseline \
  --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 \
  --stay-bias 6 --live --blind-lookback 0 \
  --out-dir submissions/v5_mac_baseline_oracle_live0
```
