# phase2_8_idlock_preword_viterbi_null45 — post-lock Viterbi + null-state probe

**Decision:** negative diagnostic. Do not promote.

Overall: **77.1%** frame accuracy (`2640/3425`), collar `1s`.

This probe tested whether low-confidence filler chunks (especially `ਵਾਹਿਗੁਰੂ`
sections in `zOtIpxMT9hU_cold66`) should be represented as "no lyric line"
instead of being forced into the nearest canonical line.

## Command

```bash
HF_WINDOW_SECONDS=10 python3 scripts/run_idlock_path.py \
  --out-dir submissions/phase2_8_idlock_preword_viterbi_null45 \
  --post-adapter-dir lora_adapters/v5b_mac_diverse \
  --post-context buffered \
  --pre-word-timestamps \
  --smoother viterbi \
  --viterbi-null-score 45

python3 ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/phase2_8_idlock_preword_viterbi_null45 \
  --gt ../live-gurbani-captioning-benchmark-v1/test
```

## Scores

| Case | Accuracy |
|---|---:|
| IZOsmkdmmcg | 87.5% |
| IZOsmkdmmcg_cold33 | 82.8% |
| IZOsmkdmmcg_cold66 | 85.3% |
| kZhIA8P6xWI | 71.0% |
| kZhIA8P6xWI_cold33 | 68.1% |
| kZhIA8P6xWI_cold66 | 65.7% |
| kchMJPK9Axs | 78.9% |
| kchMJPK9Axs_cold33 | 80.1% |
| kchMJPK9Axs_cold66 | 72.5% |
| zOtIpxMT9hU | 77.3% |
| zOtIpxMT9hU_cold33 | 71.9% |
| zOtIpxMT9hU_cold66 | 42.4% |

## Interpretation

The null state did suppress some bad filler, but it also removed useful weak
evidence elsewhere. The net result is far below the current best
`phase2_8_idlock_preword` score of **86.6%**.

This closes the "maybe a smarter per-chunk smoother fixes it" branch. The
remaining lift needs a real alignment layer that reasons about the full shabad
sequence, not only adjacent chunk labels.
