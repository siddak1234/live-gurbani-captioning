# x7_surt_only — surt standalone with longer blind buffer (negative result)

**Score: 68.6%** — regressed from x4_pathA_surt's 74.0%.

## Hypothesis tested

surt-small-v3 produces 10s chunks (vs fw's 5-10s segments). At `--blind-lookback 30`, surt has only ~3 chunks per buffer → noisy shabad ID on cold variants. **Hypothesis:** longer buffer (60s) → more chunks → reliable ID → higher score.

## Result

Score went *down*. The 60s buffer eats more of the cold variants' UEM (cold66 has only 99 frames; losing 60s = 60% of the scored region). Net negative: ID became no more reliable AND fewer frames remained scorable.

## Per-shabad

| | X4 surt (30s lookback) | X7 surt (60s lookback) |
|---|---|---|
| IZOsmkdmmcg | 90 / 10 / 17 | 90 / 20 / 35 |
| kZhIA8P6xWI | 79 / 73 / 65 | 78 / 69 / 65 |
| kchMJPK9Axs | 92 / 93 / 93 | 92 / 90 / 88 |
| zOtIpxMT9hU | 80 / 68 / 38 | **10** / 63 / 38 |
| Overall | 74.0% | 68.6% |

zOtIpxMT9hU cold0 collapsed from 80 → 10. The 60s buffer caused a new blind-ID failure.

## Conclusion

**surt-small-v3 is held back by ASR-matcher chunk-granularity mismatch, not by buffer length.** The chunks contain canonical Gurmukhi but each one spans multiple GT segments, so per-chunk line classification loses transitions.

**Real fix:** word-level timestamps from the model (untested — model was fine-tuned without timestamp tokens), or a hybrid (surt for text, faster-whisper for timing).

Kept as a recorded negative result so future sessions don't re-run this.
