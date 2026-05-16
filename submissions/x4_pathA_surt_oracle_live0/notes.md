# x4_pathA_surt_oracle_live0 — Phase 2.6 oracle baseline

**Decision:** diagnostic only. Not deployable because the GT shabad ID is provided and `blind_lookback=0`.

Overall: **85.2%** frame accuracy (`2918/3425`), collar `1s`.

This is the base `surindersinghssj/surt-small-v3` oracle-shabad/live0 alignment baseline. It establishes that `v5b_mac_diverse_oracle_live0` at `87.4%` is a real +2.2 pt oracle-alignment lift rather than just an oracle-mode artifact.

Command:

```bash
HF_WINDOW_SECONDS=10 python3 scripts/run_path_a.py \
  --backend huggingface_whisper \
  --model surindersinghssj/surt-small-v3 \
  --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 \
  --stay-bias 6 --live --blind-lookback 0 \
  --out-dir submissions/x4_pathA_surt_oracle_live0
```
