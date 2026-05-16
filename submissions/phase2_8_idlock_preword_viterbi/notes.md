# phase2_8_idlock_preword_viterbi — post-lock Viterbi smoother probe

**Decision:** negative diagnostic. Do not promote.

Overall: **77.2%** frame accuracy (`2645/3425`), collar `1s`.

This probe kept the best Phase 2.8 ID-lock setup:

- pre-lock engine: faster-whisper `medium`, `word_timestamps=True`, 30s blind lookback;
- post-lock engine: `surindersinghssj/surt-small-v3` + `v5b_mac_diverse`;
- post context: `buffered`;
- new variable: post/pre smoother set to `viterbi` (line-continuity penalty, no null state).

## Command

```bash
HF_WINDOW_SECONDS=10 python3 scripts/run_idlock_path.py \
  --out-dir submissions/phase2_8_idlock_preword_viterbi \
  --post-adapter-dir lora_adapters/v5b_mac_diverse \
  --post-context buffered \
  --pre-word-timestamps \
  --smoother viterbi

python3 ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/phase2_8_idlock_preword_viterbi \
  --gt ../live-gurbani-captioning-benchmark-v1/test
```

## Scores

| Case | Accuracy |
|---|---:|
| IZOsmkdmmcg | 93.8% |
| IZOsmkdmmcg_cold33 | 92.2% |
| IZOsmkdmmcg_cold66 | 98.1% |
| kZhIA8P6xWI | 69.6% |
| kZhIA8P6xWI_cold33 | 63.3% |
| kZhIA8P6xWI_cold66 | 54.3% |
| kchMJPK9Axs | 78.3% |
| kchMJPK9Axs_cold33 | 73.7% |
| kchMJPK9Axs_cold66 | 62.6% |
| zOtIpxMT9hU | 80.5% |
| zOtIpxMT9hU_cold33 | 73.5% |
| zOtIpxMT9hU_cold66 | 37.4% |

## Interpretation

The sequence penalty helped `IZOsmkdmmcg` dramatically, but it damaged `kZhIA8P6xWI`
and `kchMJPK9Axs`. This is exactly the failure mode we wanted to detect: a
single generic line-distance penalty is not a robust replacement for a real
alignment model. It over-regularizes shabads whose sung structure includes
refrain loops or non-local line returns.

Keep `smooth_with_viterbi()` as a tested diagnostic primitive, but do not use it
as the production path. The next step is full-shabad alignment with explicit
loop/refrain handling.
