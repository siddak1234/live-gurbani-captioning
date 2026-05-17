# Locked-shabad line-path audit

**Diagnostic only.** Shabad lock is treated as mostly solved here; this
report asks how the line tracker fails *inside* the locked shabad. The
goal is to decide whether the next high-accuracy move is aligner logic or
another broad acoustic training run.

## Inputs

- Predictions: `submissions/phase3_recency_guard_paired`
- Ground truth: `../live-gurbani-captioning-benchmark-v1/test`
- Corpus: `/Users/sbhatia/Desktop/live-gurbani-captioning/corpus_cache`
- Scorer collar: `1s`

## Summary

- Overall frame accuracy: `91.0%` (3118/3425)
- Error frames: `307`

### Error Kinds

| Kind | Frames | Share of errors |
|---|---:|---:|
| `wrong_line` | 194 | 63.2% |
| `boundary_wrong` | 98 | 31.9% |
| `missing_pred` | 15 | 4.9% |

### Line-Path Relations

| Relation | Frames | Share of errors |
|---|---:|---:|
| `adjacent_backtrack` | 98 | 31.9% |
| `predicted_during_unlabeled_gt` | 92 | 30.0% |
| `future_jump` | 54 | 17.6% |
| `adjacent_future` | 30 | 9.8% |
| `backtrack_jump` | 18 | 5.9% |
| `missing_prediction` | 15 | 4.9% |

## Per Case

| Case | Accuracy | Error frames | Pred segs | GT segs | Dominant relations |
|---|---:|---:|---:|---:|---|
| `kZhIA8P6xWI` | 82.8% | 52 | 19 | 17 | future_jump=18, adjacent_backtrack=13, adjacent_future=12, predicted_during_unlabeled_gt=5 |
| `kZhIA8P6xWI_cold33` | 84.1% | 33 | 14 | 13 | future_jump=13, adjacent_backtrack=6, adjacent_future=6, predicted_during_unlabeled_gt=4 |
| `kZhIA8P6xWI_cold66` | 84.8% | 16 | 7 | 7 | future_jump=7, adjacent_future=6, adjacent_backtrack=3 |
| `zOtIpxMT9hU_cold66` | 87.9% | 12 | 5 | 7 | adjacent_backtrack=7, missing_prediction=5 |
| `kchMJPK9Axs` | 88.6% | 74 | 20 | 18 | predicted_during_unlabeled_gt=37, adjacent_backtrack=26, future_jump=5, adjacent_future=3 |
| `kchMJPK9Axs_cold33` | 90.6% | 41 | 15 | 13 | predicted_during_unlabeled_gt=27, adjacent_backtrack=6, future_jump=5, backtrack_jump=3 |
| `zOtIpxMT9hU_cold33` | 93.9% | 12 | 8 | 10 | adjacent_backtrack=7, missing_prediction=5 |
| `IZOsmkdmmcg_cold66` | 94.2% | 9 | 6 | 6 | adjacent_backtrack=7, future_jump=2 |
| `IZOsmkdmmcg` | 94.7% | 24 | 16 | 15 | predicted_during_unlabeled_gt=9, adjacent_backtrack=9, adjacent_future=2, backtrack_jump=2 |
| `kchMJPK9Axs_cold66` | 95.5% | 10 | 6 | 6 | predicted_during_unlabeled_gt=10 |
| `zOtIpxMT9hU` | 95.8% | 12 | 9 | 11 | adjacent_backtrack=7, missing_prediction=5 |
| `IZOsmkdmmcg_cold33` | 96.1% | 12 | 11 | 11 | adjacent_backtrack=7, backtrack_jump=2, future_jump=2, adjacent_future=1 |

## Top Line Confusions

| GT line | Pred line | Relation | Frames |
|---:|---:|---|---:|
| null | 8 | `predicted_during_unlabeled_gt` | 69 |
| 4 | 3 | `adjacent_backtrack` | 38 |
| 3 | 4 | `adjacent_future` | 25 |
| 2 | 1 | `adjacent_backtrack` | 21 |
| 6 | 5 | `adjacent_backtrack` | 18 |
| 3 | 8 | `future_jump` | 14 |
| null | 3 | `predicted_during_unlabeled_gt` | 14 |
| 3 | 2 | `adjacent_backtrack` | 12 |
| 5 | 8 | `future_jump` | 12 |
| 3 | 10 | `future_jump` | 12 |
| 6 | null | `missing_prediction` | 12 |
| 3 | 6 | `future_jump` | 10 |
| null | 4 | `predicted_during_unlabeled_gt` | 9 |
| 7 | 6 | `adjacent_backtrack` | 9 |
| 6 | 3 | `backtrack_jump` | 8 |

## Longest Error Spans

| Case | Start | End | Dur | Relation | Kind | GT | Pred |
|---|---:|---:|---:|---|---|---|---|
| `kchMJPK9Axs` | 50 | 70 | 20 | `adjacent_backtrack` | `wrong_line` | 4: ਨਿਰਭੈ ਹੋਇ ਨ ਹਰਿ ਭਜੇ ਮਨ ਬਉਰਾ ਰੇ ਗਹਿਓ ਨ ਰਾਮ ਜਹਾਜੁ ॥੧॥ ਰ... | 3: ਬਿਖੈ ਬਾਚੁ ਹਰਿ ਰਾਚੁ ਸਮਝੁ ਮਨ ਬਉਰਾ ਰੇ ॥ |
| `kchMJPK9Axs` | 130 | 140 | 10 | `predicted_during_unlabeled_gt` | `boundary_wrong` | null | 8: ਜੈਸਾ ਰੰਗੁ ਕਸੁੰਭ ਕਾ ਮਨ ਬਉਰਾ ਰੇ ਤਿਉ ਪਸਰਿਓ ਪਾਸਾਰੁ ॥੩॥ |
| `kchMJPK9Axs` | 260 | 270 | 10 | `predicted_during_unlabeled_gt` | `boundary_wrong` | null | 8: ਜੈਸਾ ਰੰਗੁ ਕਸੁੰਭ ਕਾ ਮਨ ਬਉਰਾ ਰੇ ਤਿਉ ਪਸਰਿਓ ਪਾਸਾਰੁ ॥੩॥ |
| `kchMJPK9Axs` | 490 | 500 | 10 | `predicted_during_unlabeled_gt` | `boundary_wrong` | null | 8: ਜੈਸਾ ਰੰਗੁ ਕਸੁੰਭ ਕਾ ਮਨ ਬਉਰਾ ਰੇ ਤਿਉ ਪਸਰਿਓ ਪਾਸਾਰੁ ॥੩॥ |
| `kchMJPK9Axs_cold33` | 260 | 270 | 10 | `predicted_during_unlabeled_gt` | `boundary_wrong` | null | 8: ਜੈਸਾ ਰੰਗੁ ਕਸੁੰਭ ਕਾ ਮਨ ਬਉਰਾ ਰੇ ਤਿਉ ਪਸਰਿਓ ਪਾਸਾਰੁ ॥੩॥ |
| `kchMJPK9Axs_cold33` | 490 | 500 | 10 | `predicted_during_unlabeled_gt` | `boundary_wrong` | null | 8: ਜੈਸਾ ਰੰਗੁ ਕਸੁੰਭ ਕਾ ਮਨ ਬਉਰਾ ਰੇ ਤਿਉ ਪਸਰਿਓ ਪਾਸਾਰੁ ॥੩॥ |
| `kchMJPK9Axs_cold66` | 490 | 500 | 10 | `predicted_during_unlabeled_gt` | `boundary_wrong` | null | 8: ਜੈਸਾ ਰੰਗੁ ਕਸੁੰਭ ਕਾ ਮਨ ਬਉਰਾ ਰੇ ਤਿਉ ਪਸਰਿਓ ਪਾਸਾਰੁ ॥੩॥ |
| `IZOsmkdmmcg` | 90 | 99 | 9 | `predicted_during_unlabeled_gt` | `boundary_wrong` | null | 4: ਜਿਸਹਿ ਕ੍ਰਿਪਾਲੁ ਹੋਇ ਮੇਰਾ ਸੁਆਮੀ ਪੂਰਨ ਤਾ ਕੋ ਕਾਮੁ ॥੧॥ |
| `kZhIA8P6xWI` | 43 | 50 | 7 | `adjacent_backtrack` | `wrong_line` | 4: ਜਾ ਕੀ ਭਗਤਿ ਕਰਹਿ ਜਨ ਪੂਰੇ ਮੁਨਿ ਜਨ ਸੇਵਹਿ ਗੁਰ ਵੀਚਾਰਿ ॥੧॥ ... | 3: ਪ੍ਰੀਤਮ ਕਿਉ ਬਿਸਰਹਿ ਮੇਰੇ ਪ੍ਰਾਣ ਅਧਾਰ ॥ |
| `kchMJPK9Axs` | 283 | 290 | 7 | `predicted_during_unlabeled_gt` | `boundary_wrong` | null | 3: ਬਿਖੈ ਬਾਚੁ ਹਰਿ ਰਾਚੁ ਸਮਝੁ ਮਨ ਬਉਰਾ ਰੇ ॥ |
| `kchMJPK9Axs_cold33` | 283 | 290 | 7 | `predicted_during_unlabeled_gt` | `boundary_wrong` | null | 3: ਬਿਖੈ ਬਾਚੁ ਹਰਿ ਰਾਚੁ ਸਮਝੁ ਮਨ ਬਉਰਾ ਰੇ ॥ |
| `kZhIA8P6xWI` | 64 | 70 | 6 | `adjacent_future` | `wrong_line` | 3: ਪ੍ਰੀਤਮ ਕਿਉ ਬਿਸਰਹਿ ਮੇਰੇ ਪ੍ਰਾਣ ਅਧਾਰ ॥ | 4: ਜਾ ਕੀ ਭਗਤਿ ਕਰਹਿ ਜਨ ਪੂਰੇ ਮੁਨਿ ਜਨ ਸੇਵਹਿ ਗੁਰ ਵੀਚਾਰਿ ॥੧॥ ... |
| `kZhIA8P6xWI` | 294 | 300 | 6 | `adjacent_future` | `wrong_line` | 3: ਪ੍ਰੀਤਮ ਕਿਉ ਬਿਸਰਹਿ ਮੇਰੇ ਪ੍ਰਾਣ ਅਧਾਰ ॥ | 4: ਜਾ ਕੀ ਭਗਤਿ ਕਰਹਿ ਜਨ ਪੂਰੇ ਮੁਨਿ ਜਨ ਸੇਵਹਿ ਗੁਰ ਵੀਚਾਰਿ ॥੧॥ ... |
| `kZhIA8P6xWI_cold33` | 294 | 300 | 6 | `adjacent_future` | `wrong_line` | 3: ਪ੍ਰੀਤਮ ਕਿਉ ਬਿਸਰਹਿ ਮੇਰੇ ਪ੍ਰਾਣ ਅਧਾਰ ॥ | 4: ਜਾ ਕੀ ਭਗਤਿ ਕਰਹਿ ਜਨ ਪੂਰੇ ਮੁਨਿ ਜਨ ਸੇਵਹਿ ਗੁਰ ਵੀਚਾਰਿ ॥੧॥ ... |
| `kZhIA8P6xWI_cold66` | 294 | 300 | 6 | `adjacent_future` | `wrong_line` | 3: ਪ੍ਰੀਤਮ ਕਿਉ ਬਿਸਰਹਿ ਮੇਰੇ ਪ੍ਰਾਣ ਅਧਾਰ ॥ | 4: ਜਾ ਕੀ ਭਗਤਿ ਕਰਹਿ ਜਨ ਪੂਰੇ ਮੁਨਿ ਜਨ ਸੇਵਹਿ ਗੁਰ ਵੀਚਾਰਿ ॥੧॥ ... |
| `IZOsmkdmmcg` | 325 | 330 | 5 | `adjacent_backtrack` | `wrong_line` | 2: ਸਾਧਸੰਗਿ ਗਾਵਹਿ ਗੁਣ ਗੋਬਿੰਦ ਪੂਰਨ ਬ੍ਰਹਮ ਗਿਆਨੁ ॥੧॥ ਰਹਾਉ ॥ | 1: ਪੋਥੀ ਪਰਮੇਸਰ ਕਾ ਥਾਨੁ ॥ |
| `IZOsmkdmmcg_cold33` | 325 | 330 | 5 | `adjacent_backtrack` | `wrong_line` | 2: ਸਾਧਸੰਗਿ ਗਾਵਹਿ ਗੁਣ ਗੋਬਿੰਦ ਪੂਰਨ ਬ੍ਰਹਮ ਗਿਆਨੁ ॥੧॥ ਰਹਾਉ ॥ | 1: ਪੋਥੀ ਪਰਮੇਸਰ ਕਾ ਥਾਨੁ ॥ |
| `IZOsmkdmmcg_cold66` | 325 | 330 | 5 | `adjacent_backtrack` | `wrong_line` | 2: ਸਾਧਸੰਗਿ ਗਾਵਹਿ ਗੁਣ ਗੋਬਿੰਦ ਪੂਰਨ ਬ੍ਰਹਮ ਗਿਆਨੁ ॥੧॥ ਰਹਾਉ ॥ | 1: ਪੋਥੀ ਪਰਮੇਸਰ ਕਾ ਥਾਨੁ ॥ |
| `kZhIA8P6xWI` | 135 | 140 | 5 | `future_jump` | `wrong_line` | 5: ਰਵਿ ਸਸਿ ਦੀਪਕ ਜਾ ਕੇ ਤ੍ਰਿਭਵਣਿ ਏਕਾ ਜੋਤਿ ਮੁਰਾਰਿ ॥ | 8: ਅੰਤਰਿ ਜੋਤਿ ਸਬਦੁ ਧੁਨਿ ਜਾਗੈ ਸਤਿਗੁਰੁ ਝਗਰੁ ਨਿਬੇਰੈ ॥੩॥ |
| `kZhIA8P6xWI_cold33` | 135 | 140 | 5 | `future_jump` | `wrong_line` | 5: ਰਵਿ ਸਸਿ ਦੀਪਕ ਜਾ ਕੇ ਤ੍ਰਿਭਵਣਿ ਏਕਾ ਜੋਤਿ ਮੁਰਾਰਿ ॥ | 8: ਅੰਤਰਿ ਜੋਤਿ ਸਬਦੁ ਧੁਨਿ ਜਾਗੈ ਸਤਿਗੁਰੁ ਝਗਰੁ ਨਿਬੇਰੈ ॥੩॥ |
| `kchMJPK9Axs` | 345 | 350 | 5 | `future_jump` | `wrong_line` | 3: ਬਿਖੈ ਬਾਚੁ ਹਰਿ ਰਾਚੁ ਸਮਝੁ ਮਨ ਬਉਰਾ ਰੇ ॥ | 6: ਛੂਟਨ ਕੋ ਸਹਸਾ ਪਰਿਆ ਮਨ ਬਉਰਾ ਰੇ ਨਾਚਿਓ ਘਰ ਘਰ ਬਾਰਿ ॥੨॥ |
| `kchMJPK9Axs_cold33` | 345 | 350 | 5 | `future_jump` | `wrong_line` | 3: ਬਿਖੈ ਬਾਚੁ ਹਰਿ ਰਾਚੁ ਸਮਝੁ ਮਨ ਬਉਰਾ ਰੇ ॥ | 6: ਛੂਟਨ ਕੋ ਸਹਸਾ ਪਰਿਆ ਮਨ ਬਉਰਾ ਰੇ ਨਾਚਿਓ ਘਰ ਘਰ ਬਾਰਿ ॥੨॥ |
| `kZhIA8P6xWI` | 16 | 20 | 4 | `future_jump` | `wrong_line` | 3: ਪ੍ਰੀਤਮ ਕਿਉ ਬਿਸਰਹਿ ਮੇਰੇ ਪ੍ਰਾਣ ਅਧਾਰ ॥ | 8: ਅੰਤਰਿ ਜੋਤਿ ਸਬਦੁ ਧੁਨਿ ਜਾਗੈ ਸਤਿਗੁਰੁ ਝਗਰੁ ਨਿਬੇਰੈ ॥੩॥ |
| `kZhIA8P6xWI` | 130 | 134 | 4 | `predicted_during_unlabeled_gt` | `boundary_wrong` | null | 8: ਅੰਤਰਿ ਜੋਤਿ ਸਬਦੁ ਧੁਨਿ ਜਾਗੈ ਸਤਿਗੁਰੁ ਝਗਰੁ ਨਿਬੇਰੈ ॥੩॥ |
| `kZhIA8P6xWI` | 160 | 164 | 4 | `backtrack_jump` | `wrong_line` | 6: ਗੁਰਮੁਖਿ ਹੋਇ ਸੁ ਅਹਿਨਿਸਿ ਨਿਰਮਲੁ ਮਨਮੁਖਿ ਰੈਣਿ ਅੰਧਾਰਿ ॥੨॥ | 3: ਪ੍ਰੀਤਮ ਕਿਉ ਬਿਸਰਹਿ ਮੇਰੇ ਪ੍ਰਾਣ ਅਧਾਰ ॥ |
| `kZhIA8P6xWI` | 266 | 270 | 4 | `future_jump` | `wrong_line` | 3: ਪ੍ਰੀਤਮ ਕਿਉ ਬਿਸਰਹਿ ਮੇਰੇ ਪ੍ਰਾਣ ਅਧਾਰ ॥ | 10: ਨਾਨਕ ਸਹਜਿ ਮਿਲੇ ਜਗਜੀਵਨ ਨਦਰਿ ਕਰਹੁ ਨਿਸਤਾਰਾ ॥੪॥੨॥ |
| `kZhIA8P6xWI_cold33` | 130 | 134 | 4 | `predicted_during_unlabeled_gt` | `boundary_wrong` | null | 8: ਅੰਤਰਿ ਜੋਤਿ ਸਬਦੁ ਧੁਨਿ ਜਾਗੈ ਸਤਿਗੁਰੁ ਝਗਰੁ ਨਿਬੇਰੈ ॥੩॥ |
| `kZhIA8P6xWI_cold33` | 160 | 164 | 4 | `backtrack_jump` | `wrong_line` | 6: ਗੁਰਮੁਖਿ ਹੋਇ ਸੁ ਅਹਿਨਿਸਿ ਨਿਰਮਲੁ ਮਨਮੁਖਿ ਰੈਣਿ ਅੰਧਾਰਿ ॥੨॥ | 3: ਪ੍ਰੀਤਮ ਕਿਉ ਬਿਸਰਹਿ ਮੇਰੇ ਪ੍ਰਾਣ ਅਧਾਰ ॥ |
| `kZhIA8P6xWI_cold33` | 266 | 270 | 4 | `future_jump` | `wrong_line` | 3: ਪ੍ਰੀਤਮ ਕਿਉ ਬਿਸਰਹਿ ਮੇਰੇ ਪ੍ਰਾਣ ਅਧਾਰ ॥ | 10: ਨਾਨਕ ਸਹਜਿ ਮਿਲੇ ਜਗਜੀਵਨ ਨਦਰਿ ਕਰਹੁ ਨਿਸਤਾਰਾ ॥੪॥੨॥ |
| `kZhIA8P6xWI_cold66` | 266 | 270 | 4 | `future_jump` | `wrong_line` | 3: ਪ੍ਰੀਤਮ ਕਿਉ ਬਿਸਰਹਿ ਮੇਰੇ ਪ੍ਰਾਣ ਅਧਾਰ ॥ | 10: ਨਾਨਕ ਸਹਜਿ ਮਿਲੇ ਜਗਜੀਵਨ ਨਦਰਿ ਕਰਹੁ ਨਿਸਤਾਰਾ ॥੪॥੨॥ |

## Decision Use

- High `adjacent_future` / `adjacent_backtrack`: tune transition penalties and boundary smoothing before larger training.
- High `future_jump` / `backtrack_jump`: line-state path is unstable; Viterbi/loop-align constraints should be tightened.
- High `outside_gt_line_set`: inspect whether the clip labels omit sung repeats; if labels are acceptable, add a no-line/end-of-clip guard.
- High `wrong_shabad`: return to shabad-lock evidence; this report is no longer in the locked-shabad regime.
