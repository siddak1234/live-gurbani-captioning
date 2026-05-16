# v5b_idlock_runtime — Phase 2.7 runtime ID-lock

**Decision:** failed benchmark gate; do not promote.

Overall: **75.6%** frame accuracy (`2589/3425`), collar `1s`.

This is the first real runtime implementation of the Phase 2.6 ID-lock idea. It is not a submission-level merge:

- pre-lock engine: faster-whisper `medium`, blind/live, tentative captions, chunk-vote shabad ID, 30s lookback;
- post-lock engine: `surindersinghssj/surt-small-v3` + `lora_adapters/v5b_mac_diverse`, constrained to the pre-lock committed shabad;
- switch: `uem.start + 30s`;
- implementation: `src/idlock_engine.py` (Layer 2 library) + `scripts/run_idlock_path.py` (Layer 3 benchmark runner).

## Command

```bash
HF_WINDOW_SECONDS=10 python3 scripts/run_idlock_path.py \
  --out-dir submissions/v5b_idlock_runtime \
  --post-adapter-dir lora_adapters/v5b_mac_diverse \
  --post-context buffered

python3 ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/v5b_idlock_runtime \
  --gt ../live-gurbani-captioning-benchmark-v1/test
```

## Scores

| Case | Accuracy | Runtime lock |
|---|---:|---|
| IZOsmkdmmcg | 90.3% | 4377 -> 4377 |
| IZOsmkdmmcg_cold33 | 89.0% | 4377 -> 4377 |
| IZOsmkdmmcg_cold66 | 85.3% | 4377 -> 4377 |
| kZhIA8P6xWI | 5.0% | **4377 -> 1821** |
| kZhIA8P6xWI_cold33 | 10.6% | **4377 -> 1821** |
| kZhIA8P6xWI_cold66 | 78.1% | 1821 -> 1821 |
| kchMJPK9Axs | 93.2% | 1341 -> 1341 |
| kchMJPK9Axs_cold33 | 90.9% | 1341 -> 1341 |
| kchMJPK9Axs_cold66 | 100.0% | 1341 -> 1341 |
| zOtIpxMT9hU | 80.5% | 3712 -> 3712 |
| zOtIpxMT9hU_cold33 | 73.5% | 3712 -> 3712 |
| zOtIpxMT9hU_cold66 | 52.5% | 3712 -> 3712 |

## Interpretation

The runtime architecture works mechanically, but the scientific hypothesis did not transfer. The two `kZhIA8P6xWI` shabad-ID failures collapse the score. Once the shabad is correct, the post-lock v5b alignment behaves like the Phase 2.6 diagnostic predicted (`kchMJPK9Axs` and most `IZOsmkdmmcg` cases are strong).

The more important finding is reproducibility drift: re-running the frozen v3.2 command with the current environment/cache produces `73.5%`, not the archived `86.5%`. That means Phase 2.6's `87.1%` proxy was built from a historical submission artifact, not from a currently reproducible runtime baseline.

Diagnostic ID aggregator probes:

| Runtime variant | Overall | Notes |
|---|---:|---|
| `chunk_vote`, 30s | 75.6% | Best of the checked runtime ID modes; still fails `kZhIA8P6xWI` cold0/cold33. |
| `tfidf`, 30s | 65.5% | Fixes `kZhIA8P6xWI`, breaks `IZOsmkdmmcg` and `zOtIpxMT9hU`. |
| `topk:3`, 30s | 58.7% | Breaks too many starts. |

**Next:** do not start Phase 3 LoRA scale-up. Move to Phase 2.8: recover/pin the ASR baseline and prototype timestamp/alignment paths (`vad_filter=False`, `word_timestamps=True`, hybrid surt text + faster-whisper timestamps, or full-shabad forced alignment).
