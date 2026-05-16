# v3_2_repro_current — current-runtime reproducibility check

**Decision:** archived v3.2 is not currently reproducible; treat the historical `86.5%` as a frozen artifact until the ASR/runtime drift is pinned.

Overall: **73.5%** frame accuracy (`2516/3425`), collar `1s`.

This reruns the exact command documented in `submissions/v3_2_pathA_no_title/notes.md` under the current environment:

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
  --out-dir submissions/v3_2_repro_current

python3 ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/v3_2_repro_current \
  --gt ../live-gurbani-captioning-benchmark-v1/test
```

## Scores

| Case | Accuracy | Runtime blind ID |
|---|---:|---|
| IZOsmkdmmcg | 100.0% | 4377 -> 4377 |
| IZOsmkdmmcg_cold33 | 98.7% | 4377 -> 4377 |
| IZOsmkdmmcg_cold66 | 99.4% | 4377 -> 4377 |
| kZhIA8P6xWI | 5.3% | **4377 -> 1821** |
| kZhIA8P6xWI_cold33 | 19.3% | **4377 -> 1821** |
| kZhIA8P6xWI_cold66 | 60.0% | 1821 -> 1821 |
| kchMJPK9Axs | 73.8% | 1341 -> 1341 |
| kchMJPK9Axs_cold33 | 73.3% | 1341 -> 1341 |
| kchMJPK9Axs_cold66 | 81.5% | 1341 -> 1341 |
| zOtIpxMT9hU | 90.2% | 3712 -> 3712 |
| zOtIpxMT9hU_cold33 | 86.7% | 3712 -> 3712 |
| zOtIpxMT9hU_cold66 | 73.7% | 3712 -> 3712 |

## Interpretation

The historical `v3_2_pathA_no_title` submission still evaluates to `86.5%`, but the same command today evaluates to `73.5%`. The main collapse is the same blind-ID failure seen in Phase 2.7 runtime ID-lock: `kZhIA8P6xWI` cold0/cold33 commit to shabad 4377.

This strongly suggests one or more unpinned runtime variables changed:

- faster-whisper / CTranslate2 version;
- model snapshot / conversion behavior;
- VAD behavior;
- local ASR transcript cache availability;
- environment flags around timestamping or silence filtering.

Before any result can be called production-valid, Phase 2.8 needs to make the ASR baseline reproducible: pin versions, preserve transcript checksums, and record model/backend metadata in submission notes.
