# v5b_mac_diverse_oracle_live0 — Phase 2.6 oracle diagnostic

**Decision:** diagnostic only. Not a deployable score because the GT shabad ID is provided and `blind_lookback=0`.

## Result

Overall: **87.4%** frame accuracy (`2992/3425`), collar `1s`.

This is +21.8 pt over `v5b_mac_diverse` blind/live (`65.6%`) and +2.2 pt over the base surt oracle-live0 diagnostic (`85.2%`). The adapter is not globally worse; it helps alignment once the shabad is known.

## Command

```bash
HF_WINDOW_SECONDS=10 python3 scripts/run_path_a.py \
  --backend huggingface_whisper \
  --model surindersinghssj/surt-small-v3 \
  --adapter-dir lora_adapters/v5b_mac_diverse \
  --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 \
  --stay-bias 6 --live --blind-lookback 0 \
  --out-dir submissions/v5b_mac_diverse_oracle_live0

python3 ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/v5b_mac_diverse_oracle_live0/ \
  --gt ../live-gurbani-captioning-benchmark-v1/test/
```

## Scores

| Case | Accuracy |
|---|---:|
| IZOsmkdmmcg | 90.3% |
| IZOsmkdmmcg_cold33 | 87.0% |
| IZOsmkdmmcg_cold66 | 80.8% |
| kZhIA8P6xWI | 84.5% |
| kZhIA8P6xWI_cold33 | 83.1% |
| kZhIA8P6xWI_cold66 | 78.1% |
| kchMJPK9Axs | 93.2% |
| kchMJPK9Axs_cold33 | 94.3% |
| kchMJPK9Axs_cold66 | 96.4% |
| zOtIpxMT9hU | 85.4% |
| zOtIpxMT9hU_cold33 | 76.5% |
| zOtIpxMT9hU_cold66 | 50.5% |

## Interpretation

The v5b regression is primarily shabad-ID/integration failure. Continue with ID-lock and alignment work; do not scale training until this boundary is fixed and OOS-tested.
