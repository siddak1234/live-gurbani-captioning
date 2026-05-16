# phase2_8_idlock_preword — word-timestamp ID-lock hybrid

**Decision:** best current runtime architecture, but still below promotion gate and OOS is owed.

Overall: **86.6%** frame accuracy (`2965/3425`), collar `1s`.

This is the strongest Phase 2.8 signal so far:

- pre-lock engine: faster-whisper `medium`, `word_timestamps=True`, blind/live, 30s lookback;
- post-lock engine: `surindersinghssj/surt-small-v3` + `v5b_mac_diverse`, locked to the pre-lock shabad ID;
- switch: `uem.start + 30s`;
- implementation: `src/idlock_engine.py` + `scripts/run_idlock_path.py`.

## Command

```bash
HF_WINDOW_SECONDS=10 python3 scripts/run_idlock_path.py \
  --out-dir submissions/phase2_8_idlock_preword \
  --post-adapter-dir lora_adapters/v5b_mac_diverse \
  --post-context buffered \
  --pre-word-timestamps

python3 ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/phase2_8_idlock_preword \
  --gt ../live-gurbani-captioning-benchmark-v1/test
```

## Scores

| Case | Accuracy | Runtime lock |
|---|---:|---|
| IZOsmkdmmcg | 90.1% | 4377 -> 4377 |
| IZOsmkdmmcg_cold33 | 86.7% | 4377 -> 4377 |
| IZOsmkdmmcg_cold66 | 85.3% | 4377 -> 4377 |
| kZhIA8P6xWI | 82.8% | 1821 -> 1821 |
| kZhIA8P6xWI_cold33 | 82.6% | 1821 -> 1821 |
| kZhIA8P6xWI_cold66 | 78.1% | 1821 -> 1821 |
| kchMJPK9Axs | 93.2% | 1341 -> 1341 |
| kchMJPK9Axs_cold33 | 95.9% | 1341 -> 1341 |
| kchMJPK9Axs_cold66 | 96.4% | 1341 -> 1341 |
| zOtIpxMT9hU | 80.5% | 3712 -> 3712 |
| zOtIpxMT9hU_cold33 | 73.5% | 3712 -> 3712 |
| zOtIpxMT9hU_cold66 | 37.4% | 3712 -> 3712 |

## Interpretation

Word timestamps solve the blind-ID failure: 12/12 shabad locks are correct. The post-lock v5b alignment then recovers most of the Phase 2.6 proxy behavior.

It still misses the `>=87.0%` Phase 2.8 paired-benchmark gate by `0.4` points and remains well short of the 95% target. The main residual weakness is short cold-start timing, especially `zOtIpxMT9hU_cold66`.

Shorter lookback probes were not better:

| Variant | Overall | Reason |
|---|---:|---|
| 15s pre-word ID-lock | 79.7% | `kZhIA8P6xWI` open case mis-locks. |
| 20s pre-word ID-lock | 79.8% | `kZhIA8P6xWI` open case mis-locks. |
| 30s pre-word ID-lock | **86.6%** | Correct 12/12 shabad locks; best current runtime. |

Next step: keep this as the current best runtime candidate, but do not promote. The next lift likely requires a real alignment layer after shabad lock rather than more ASR chunk-shape tweaks.
