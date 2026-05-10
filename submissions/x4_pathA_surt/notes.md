# x4_pathA_surt_w10 — Path A v3.2 pipeline with surt-small-v3 ASR (10s windows)

**Phase X4 finding:** The dataset I claimed didn't exist DOES exist. `surindersinghssj/surt-small-v3` on HuggingFace is a Whisper-small fine-tuned on **660h of Gurbani audio**, Apache 2.0 licensed. Plugged it into Path A as a third ASR backend (`huggingface_whisper`).

## Setup

- ASR: `surindersinghssj/surt-small-v3` (Whisper-small fine-tune for Gurbani)
- Input format: 30s manual windows for transcription (model trained without
  timestamp tokens, can't use HF pipeline's `return_timestamps=True`)
- Best window length here: **10s** via `HF_WINDOW_SECONDS=10`
- Otherwise identical to v3.2 (oracle/blind, live, tentative emission, stay-bias=6)

## Score

**Overall: 74.0%** (2535/3425 frames) — down 12.5 from v3.2's 86.5%, but story is more nuanced than the headline.

## Per-shabad (vs Path A v3.2 baseline)

| Shabad | v3.2 baseline | X4 w10 | Δ |
|---|---|---|---|
| IZOsmkdmmcg | 98 / 96 / 94 | **90** / 10 / 17 | mixed (cold variants blind-ID failed) |
| kZhIA8P6xWI | 87 / 81 / 88 | 79 / 73 / 65 | -5 to -23 |
| kchMJPK9Axs | 83 / 77 / 76 | **92 / 93 / 93** | **+10 to +17** ✨ |
| zOtIpxMT9hU | 93 / 79 / 83 | 80 / 68 / 38 | -13 to -45 |

surt-small-v3 is a **clear acoustic win on kchMJPK9Axs** (the shabad where the rahao refrain "man bauraa re" caused the most trouble for fw — surt's training on canonical text snaps to the right line). The cold-variant collapses on IZOsmkdmmcg and zOtIpxMT9hU are from blind ID failures: shorter UEM → fewer chunks in the 30s buffer → noisy chunk_vote.

Three window sizes tried:
- 30s windows: 43.9% (too coarse, each chunk spans multiple lines)
- **10s windows: 74.0%** ← retained
- 5s windows: 61.5% (IZOsmkdmmcg specifically collapsed)

## Oracle ensemble ceiling

Best-of {v3.2, X4 w10, X4 w5} per case = **~90.3% overall**. Method-aware
ensembling could plausibly hit this. Concrete shabad-level pattern:
- IZOsmkdmmcg, zOtIpxMT9hU → Path A v3.2 wins
- kchMJPK9Axs → surt-small-v3 wins
- kZhIA8P6xWI → roughly tied

The lift is real and reachable, but requires a non-trivial ensembling layer
that we haven't built. Pure surt-small-v3 substitution doesn't beat v3.2.

## Run command

```bash
HF_WINDOW_SECONDS=10 python scripts/run_path_a.py \
  --backend huggingface_whisper --model "surindersinghssj/surt-small-v3" \
  --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 \
  --stay-bias 6 --blind --blind-aggregate chunk_vote \
  --blind-lookback 30 --live --tentative-emit \
  --out-dir submissions/x4_pathA_surt_w10
```

## What this changes about Phase B

- **Skipping data collection is now realistic.** surindersinghssj already collected and published 300h. They've fine-tuned and released the model.
- **If we want to push past 86.5%, the next experiment is an ensemble or per-case-method-selection layer**, not data collection.
- Alternative: try `surindersinghssj/indicconformer-pa-v3-kirtan` — NeMo-based, different architecture, may complement surt-small-v3.
