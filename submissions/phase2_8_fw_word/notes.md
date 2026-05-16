# phase2_8_fw_word — faster-whisper word-timestamp probe

**Decision:** useful for blind shabad ID, not a promoted captioning path.

Overall: **72.0%** frame accuracy (`2465/3425`), collar `1s`.

## Command

```bash
python3 scripts/run_path_a.py \
  --blend "token_sort_ratio:0.5,WRatio:0.5" \
  --threshold 0 \
  --stay-bias 6 \
  --blind \
  --blind-aggregate chunk_vote \
  --blind-lookback 30 \
  --live \
  --tentative-emit \
  --word-timestamps \
  --out-dir submissions/phase2_8_fw_word

python3 ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/phase2_8_fw_word \
  --gt ../live-gurbani-captioning-benchmark-v1/test
```

## Scores

| Case | Accuracy |
|---|---:|
| IZOsmkdmmcg | 86.8% |
| IZOsmkdmmcg_cold33 | 88.0% |
| IZOsmkdmmcg_cold66 | 89.1% |
| kZhIA8P6xWI | 87.1% |
| kZhIA8P6xWI_cold33 | 86.5% |
| kZhIA8P6xWI_cold66 | 83.8% |
| kchMJPK9Axs | 74.3% |
| kchMJPK9Axs_cold33 | 67.6% |
| kchMJPK9Axs_cold66 | 50.5% |
| zOtIpxMT9hU | 53.3% |
| zOtIpxMT9hU_cold33 | 33.2% |
| zOtIpxMT9hU_cold66 | 21.2% |

## Interpretation

This fixes the Phase 2.7 blind-ID failure: all 12 starts commit to the correct shabad. But the word-timestamp segment shape hurts line tracking badly, especially `zOtIpxMT9hU` and cold `kchMJPK9Axs`.

The signal is architectural: use word timestamps for the shabad-ID window, not as the full captioning segmentation path.
