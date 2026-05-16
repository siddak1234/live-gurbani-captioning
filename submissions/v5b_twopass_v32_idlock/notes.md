# v5b_twopass_v32_idlock — Phase 2.6 ID-lock proxy

**Decision:** promising diagnostic, not yet a runtime engine.

Overall: **87.1%** frame accuracy (`2983/3425`), collar `1s`.

This submission merges:

- pre-lock segments from `submissions/v3_2_pathA_no_title`;
- post-lock segments from `submissions/v5b_mac_diverse_oracle_live0`;
- commit time = `GT uem.start + 30s`, matching the live blind-ID buffer.

It estimates a conservative two-pass architecture: use the proven v3.2/faster-whisper path for blind ID and tentative captions, then use the v5b adapter for line tracking after shabad lock. Because v3.2's blind ID is documented as 12/12 correct on the paired benchmark, this is a valid benchmark proxy for the ID-lock idea, but not yet a deployable implementation.

## Command

```bash
python3 scripts/merge_idlock_submissions.py \
  --gt-dir ../live-gurbani-captioning-benchmark-v1/test \
  --pre-dir submissions/v3_2_pathA_no_title \
  --post-dir submissions/v5b_mac_diverse_oracle_live0 \
  --out-dir submissions/v5b_twopass_v32_idlock \
  --lookback-seconds 30

python3 ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/v5b_twopass_v32_idlock/ \
  --gt ../live-gurbani-captioning-benchmark-v1/test/
```

## Scores

| Case | Accuracy |
|---|---:|
| IZOsmkdmmcg | 90.3% |
| IZOsmkdmmcg_cold33 | 89.0% |
| IZOsmkdmmcg_cold66 | 85.3% |
| kZhIA8P6xWI | 86.5% |
| kZhIA8P6xWI_cold33 | 76.8% |
| kZhIA8P6xWI_cold66 | 77.1% |
| kchMJPK9Axs | 93.2% |
| kchMJPK9Axs_cold33 | 95.7% |
| kchMJPK9Axs_cold66 | 96.0% |
| zOtIpxMT9hU | 80.5% |
| zOtIpxMT9hU_cold33 | 73.0% |
| zOtIpxMT9hU_cold66 | 52.5% |

## Interpretation

This rescues the v5b drop (`65.6% -> 87.1%`) and slightly beats v3.2 (`86.5%`). The result supports building a real ID-lock integration layer, but it is not close to the 95% target. The remaining lift likely needs word timestamps, hybrid timing, or full-shabad forced alignment.
