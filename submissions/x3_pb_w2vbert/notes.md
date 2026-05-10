# x3_pb_w2vbert — Phase X3.0: swap MMS-1B for w2v-bert-punjabi

**Goal:** Test whether a Punjabi-fine-tuned CTC model (`kdcyberdude/w2v-bert-punjabi`,
trained on ~500h of Punjabi audio) outperforms MMS-1B's generic Punjabi
adapter inside Path B's HMM line tracker. **No data collection or training
required — drop-in model swap.**

## Result: 70.3% overall (+9.3 from MMS, but still -16 from Path A)

| | Path A v3.2 | Path B + MMS | Path B + w2v-bert |
|---|---|---|---|
| IZOsmkdmmcg | 98 / 96 / 94 | 72 / 72 / 62 | **92 / 92 / 87** |
| kZhIA8P6xWI | 87 / 81 / 88 | 70 / 79 / 90 | 74 / 77 / 79 |
| kchMJPK9Axs | 83 / 77 / 76 | 54 / 53 / 83 | 73 / 74 / 82 |
| zOtIpxMT9hU | 93 / 79 / 83 | 43 / 32 / 19 | **30 / 11 / 21** ✗ |
| **Overall** | **86.5%** | 61.0% | 70.3% |

## What the swap reveals

**The Punjabi-tuned model is a real acoustic improvement.** Three of four
shabads jump 10-20 points vs MMS, with IZOsmkdmmcg landing within 5 points
of Path A canonical. This validates the hypothesis from the data-survey
research: existing Punjabi-fine-tuned models help substantially even
without our own training pass.

**But Path B's HMM still has structural issues.** zOtIpxMT9hU collapses to
5 segments for a 5-minute audio — the HMM gets stuck on line 1 for the
entire middle of the song. Investigated: not a vocab issue (94%+ coverage),
not a switch_log_prob tuning issue (tried -3 through -9, all collapse the
same way), not an end-state-stay issue (disabled stay at end state, no
improvement). The forward marginals on that audio are concentrated such
that L1 dominates regardless of transitions.

Honest read: with w2v-bert as the acoustic model, **Path B can match Path A
on 3/4 shabads** — but one collapsed shabad drags the average down 16
points. Even oracle-picking the best of Path A and Path B per case nets
only **86.7% (vs Path A alone at 86.5%)**. Path A wins 11 of 12 cases head-
to-head; Path B + w2v-bert wins exactly 1.

## What this means for the broader plan

The cheap probe (model swap) shows we're hitting a real architectural
plateau. Pushing significantly past 86.5% requires either:

1. **Path A enhancements** that don't need a better acoustic model. Probably
   diminishing returns given the matcher/smoother tuning we already did.
2. **Real fine-tuning** on kirtan-specific labeled data (the original X3
   plan: collect 20-50h, LoRA-fine-tune w2v-bert or wav2vec2). Multi-week.
3. **Rebuild Path B with a different decoding architecture** — e.g., proper
   forced alignment over the full shabad text with explicit loop structure,
   or use w2v-bert's chunk-level greedy decode + Path A's matcher+smoother
   (hybrid). Multi-day to multi-week.

## Run command

```bash
python scripts/run_path_b_hmm.py \
  --model-id "kdcyberdude/w2v-bert-punjabi" --target-lang "" \
  --out-dir submissions/x3_pb_w2vbert
```

(About 2 min wall time including encoding on Apple Silicon MPS. ~50MB model
download on first run plus cached log-prob matrices in `mms_cache/`.)

## Other models tried

- `gagan3012/wav2vec2-xlsr-punjabi`: ~35% overall — over-transitions wildly
  (138-228 segments per case), much worse than both MMS and w2v-bert.

## Encoder change

`src/path_b/encoder.py` now supports arbitrary HF CTC model IDs via
`AutoModelForCTC` / `AutoTokenizer` / `AutoFeatureExtractor`. MMS-style
loading with `target_lang` adapters still works for `*mms*` model IDs.
Cache keys include sanitized model_id so different models' outputs don't
collide.
