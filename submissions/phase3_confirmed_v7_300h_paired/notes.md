# phase3_confirmed_v7_300h_paired

Confirmed-runtime paired benchmark scoring for the completed v7 300h-source epoch-1 adapter.

## Config

- Mode: blind + live ID-lock runtime, retro-buffered post-lock finalization
- Pre-lock ASR: `faster_whisper:medium` with word timestamps
- Post-lock ASR: `huggingface_whisper:surindersinghssj/surt-small-v3`
- Adapter: `lora_adapters/v7_mac_300h_epoch1` (top-level adapter saved from best checkpoint)
- Best validation checkpoint: `lora_adapters/v7_mac_300h_epoch1/checkpoint-11000`
- Runtime smoother: `loop_align_confirmed`
- Lock aggregation: `guarded_fusion:tfidf_45+0.5*chunk_vote_90|offset=90|low=0.15|min=0.5`
- Confirm chunks: `2`
- Hard jump margin: `15`

## Run

```bash
PYTHON=.venv/bin/python3 make eval-paired-recency-guard-confirmed-v6 \
  CONFIRMED_ADAPTER_DIR=lora_adapters/v7_mac_300h_epoch1 \
  CONFIRMED_PAIRED_OUT=submissions/phase3_confirmed_v7_300h_paired
```

## Score

**Overall: 93.1%** frame accuracy (3190/3425 frames, 12 videos, collar=1s).

This beats the pre-v7 confirmed-runtime paired gate of **92.8%** while preserving **12/12** correct shabad locks.

| Case | Score |
|---|---:|
| IZOsmkdmmcg | 94.7% |
| IZOsmkdmmcg_cold33 | 96.1% |
| IZOsmkdmmcg_cold66 | 94.2% |
| kZhIA8P6xWI | 86.1% |
| kZhIA8P6xWI_cold33 | 88.9% |
| kZhIA8P6xWI_cold66 | 84.8% |
| kchMJPK9Axs | 92.6% |
| kchMJPK9Axs_cold33 | 94.3% |
| kchMJPK9Axs_cold66 | 100.0% |
| zOtIpxMT9hU | 95.8% |
| zOtIpxMT9hU_cold33 | 93.9% |
| zOtIpxMT9hU_cold66 | 87.9% |

## Assisted-OOS paired with this run

The matching assisted-OOS diagnostic, written to the gitignored folder
`submissions/oos_v1_assisted_phase3_confirmed_v7_300h`, scored **61.3%**
(539/880 frames) with **5/5** correct locks. That beats the prior assisted-OOS
gate of **60.8%**, but only by 0.5 points, so it is a positive scaling signal,
not a production claim.

## Interpretation

This is the first non-route-table runtime to exceed the old overfit `x6_ensemble`
paired score while also moving assisted-OOS in the right direction. The gain is
real but modest: paired moved +0.3 points and assisted-OOS moved +0.5 points.
The expert next step is controlled continuation from `checkpoint-11661` to a
3-epoch v7 run, with eval/checkpoint every 1000 steps, followed by the same
paired + assisted-OOS gates. Do not promote solely from paired accuracy; the
OOS labels remain machine-assisted and need gold correction before any public
95%+ claim.

## Artifacts

- 12 submission JSONs in this directory
- `tiles.html` visualizer generated with `--no-fetch`
- Training lineage: `lora_adapters/v7_mac_300h_epoch1/run_card.json`
