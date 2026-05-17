# Locked-shabad line-path audit

**Diagnostic only.** Shabad lock is treated as mostly solved here; this
report asks how the line tracker fails *inside* the locked shabad. The
goal is to decide whether the next high-accuracy move is aligner logic or
another broad acoustic training run.

## Inputs

- Predictions: `submissions/oos_v1_assisted_phase3_confirmed`
- Ground truth: `eval_data/oos_v1/assisted_test`
- Corpus: `/Users/sbhatia/Desktop/live-gurbani-captioning/corpus_cache`
- Scorer collar: `1s`

## Summary

- Overall frame accuracy: `60.8%` (535/880)
- Error frames: `345`

### Error Kinds

| Kind | Frames | Share of errors |
|---|---:|---:|
| `wrong_line` | 193 | 55.9% |
| `outside_gt_line` | 131 | 38.0% |
| `boundary_wrong` | 21 | 6.1% |

### Line-Path Relations

| Relation | Frames | Share of errors |
|---|---:|---:|
| `outside_gt_line_set` | 108 | 31.3% |
| `future_jump` | 84 | 24.3% |
| `backtrack_jump` | 45 | 13.0% |
| `adjacent_future` | 42 | 12.2% |
| `adjacent_backtrack` | 40 | 11.6% |
| `predicted_during_unlabeled_gt` | 26 | 7.5% |

## Per Case

| Case | Accuracy | Error frames | Pred segs | GT segs | Dominant relations |
|---|---:|---:|---:|---:|---|
| `case_001` | 47.2% | 95 | 10 | 8 | outside_gt_line_set=60, backtrack_jump=25, future_jump=8, adjacent_future=1 |
| `case_004` | 58.3% | 75 | 9 | 4 | outside_gt_line_set=48, predicted_during_unlabeled_gt=23, backtrack_jump=4 |
| `case_002` | 60.0% | 64 | 4 | 9 | future_jump=42, adjacent_future=11, backtrack_jump=10, predicted_during_unlabeled_gt=1 |
| `case_005` | 63.3% | 66 | 12 | 13 | adjacent_future=30, adjacent_backtrack=22, future_jump=10, backtrack_jump=4 |
| `case_003` | 75.0% | 45 | 7 | 6 | future_jump=24, adjacent_backtrack=18, backtrack_jump=2, predicted_during_unlabeled_gt=1 |

## Top Line Confusions

| GT line | Pred line | Relation | Frames |
|---:|---:|---|---:|
| 1 | 2 | `adjacent_future` | 40 |
| 2 | 7 | `future_jump` | 40 |
| 6 | 4 | `outside_gt_line_set` | 37 |
| 6 | 5 | `adjacent_backtrack` | 31 |
| 2 | 10 | `outside_gt_line_set` | 30 |
| 6 | 3 | `backtrack_jump` | 25 |
| null | 9 | `predicted_during_unlabeled_gt` | 21 |
| 2 | 5 | `future_jump` | 20 |
| 6 | 3 | `outside_gt_line_set` | 17 |
| 1 | 4 | `outside_gt_line_set` | 10 |
| 3 | 4 | `outside_gt_line_set` | 10 |
| 1 | 5 | `future_jump` | 10 |
| 1 | 3 | `future_jump` | 8 |
| 2 | 1 | `adjacent_backtrack` | 7 |
| 6 | 2 | `backtrack_jump` | 6 |

## Longest Error Spans

| Case | Start | End | Dur | Relation | Kind | GT | Pred |
|---|---:|---:|---:|---|---|---|---|
| `case_004` | 50 | 80 | 30 | `outside_gt_line_set` | `outside_gt_line` | 2: ਏਕ ਨੂਰ ਤੇ ਸਭੁ ਜਗੁ ਉਪਜਿਆ ਕਉਨ ਭਲੇ ਕੋ ਮੰਦੇ ॥੧॥ | 10: ਕਹਿ ਕਬੀਰ ਮੇਰੀ ਸੰਕਾ ਨਾਸੀ ਸਰਬ ਨਿਰੰਜਨੁ ਡੀਠਾ ॥੪॥੩॥ |
| `case_002` | 1 | 29 | 28 | `future_jump` | `wrong_line` | 2: ਤੁਮ ਮਾਤ ਪਿਤਾ ਹਮ ਬਾਰਿਕ ਤੇਰੇ ॥ | 7: ਤੁਮ ਤੇ ਹੋਇ ਸੁ ਆਗਿਆਕਾਰੀ ॥ |
| `case_004` | 80 | 101 | 21 | `predicted_during_unlabeled_gt` | `outside_gt_line` | null | 9: ਅਲਹੁ ਅਲਖੁ ਨ ਜਾਈ ਲਖਿਆ ਗੁਰਿ ਗੁੜੁ ਦੀਨਾ ਮੀਠਾ ॥ |
| `case_001` | 10 | 30 | 20 | `outside_gt_line_set` | `outside_gt_line` | 6: ਅਪੁਨਾ ਦਾਸੁ ਹਰਿ ਆਪਿ ਉਬਾਰਿਆ ਨਾਨਕ ਨਾਮ ਅਧਾਰਾ ॥੨॥੬॥੩੪॥ | 4: ਹਰਿ ਹਰਿ ਨਾਮੁ ਅਉਖਧੁ ਮੁਖਿ ਦੇਵੈ ਕਾਟੈ ਜਮ ਕੀ ਫੰਧਾ ॥੧॥ ਰਹਾਉ... |
| `case_003` | 140 | 160 | 20 | `future_jump` | `wrong_line` | 2: ਕ੍ਰਿਪਾ ਕਟਾਖੵ ਅਵਲੋਕਨੁ ਕੀਨੋ ਦਾਸ ਕਾ ਦੂਖੁ ਬਿਦਾਰਿਓ ॥੧॥ | 5: ਜੋ ਮਾਗਹਿ ਠਾਕੁਰ ਅਪੁਨੇ ਤੇ ਸੋਈ ਸੋਈ ਦੇਵੈ ॥ |
| `case_005` | 1 | 20 | 19 | `adjacent_future` | `wrong_line` | 1: ਕੋਈ ਬੋਲੈ ਰਾਮ ਰਾਮ ਕੋਈ ਖੁਦਾਇ ॥ | 2: ਕੋਈ ਸੇਵੈ ਗੁਸਈਆ ਕੋਈ ਅਲਾਹਿ ॥੧॥ |
| `case_004` | 101 | 118 | 17 | `outside_gt_line_set` | `outside_gt_line` | 6: ਨਾ ਕਛੁ ਪੋਚ ਮਾਟੀ ਕੇ ਭਾਂਡੇ ਨਾ ਕਛੁ ਪੋਚ ਕੁੰਭਾਰੈ ॥੨॥ | 3: ਲੋਗਾ ਭਰਮਿ ਨ ਭੂਲਹੁ ਭਾਈ ॥ |
| `case_005` | 146 | 160 | 14 | `adjacent_backtrack` | `wrong_line` | 6: ਕੋਈ ਕਰੈ ਪੂਜਾ ਕੋਈ ਸਿਰੁ ਨਿਵਾਇ ॥੨॥ | 5: ਕੋਈ ਨਾਵੈ ਤੀਰਥਿ ਕੋਈ ਹਜ ਜਾਇ ॥ |
| `case_001` | 30 | 41 | 11 | `backtrack_jump` | `wrong_line` | 6: ਅਪੁਨਾ ਦਾਸੁ ਹਰਿ ਆਪਿ ਉਬਾਰਿਆ ਨਾਨਕ ਨਾਮ ਅਧਾਰਾ ॥੨॥੬॥੩੪॥ | 3: ਮੇਰਾ ਬੈਦੁ ਗੁਰੂ ਗੋਵਿੰਦਾ ॥ |
| `case_002` | 79 | 90 | 11 | `adjacent_future` | `wrong_line` | 1: ਜੀਉ ਪਿੰਡੁ ਸਭੁ ਤੇਰੀ ਰਾਸਿ ॥ | 2: ਤੁਮ ਮਾਤ ਪਿਤਾ ਹਮ ਬਾਰਿਕ ਤੇਰੇ ॥ |
| `case_001` | 50 | 60 | 10 | `outside_gt_line_set` | `outside_gt_line` | 1: ਜਨਮ ਜਨਮ ਕੇ ਦੂਖ ਨਿਵਾਰੈ ਸੂਕਾ ਮਨੁ ਸਾਧਾਰੈ ॥ | 4: ਹਰਿ ਹਰਿ ਨਾਮੁ ਅਉਖਧੁ ਮੁਖਿ ਦੇਵੈ ਕਾਟੈ ਜਮ ਕੀ ਫੰਧਾ ॥੧॥ ਰਹਾਉ... |
| `case_001` | 113 | 123 | 10 | `outside_gt_line_set` | `outside_gt_line` | 6: ਅਪੁਨਾ ਦਾਸੁ ਹਰਿ ਆਪਿ ਉਬਾਰਿਆ ਨਾਨਕ ਨਾਮ ਅਧਾਰਾ ॥੨॥੬॥੩੪॥ | 4: ਹਰਿ ਹਰਿ ਨਾਮੁ ਅਉਖਧੁ ਮੁਖਿ ਦੇਵੈ ਕਾਟੈ ਜਮ ਕੀ ਫੰਧਾ ॥੧॥ ਰਹਾਉ... |
| `case_001` | 1 | 10 | 9 | `backtrack_jump` | `wrong_line` | 6: ਅਪੁਨਾ ਦਾਸੁ ਹਰਿ ਆਪਿ ਉਬਾਰਿਆ ਨਾਨਕ ਨਾਮ ਅਧਾਰਾ ॥੨॥੬॥੩੪॥ | 3: ਮੇਰਾ ਬੈਦੁ ਗੁਰੂ ਗੋਵਿੰਦਾ ॥ |
| `case_003` | 1 | 10 | 9 | `adjacent_backtrack` | `wrong_line` | 6: ਨਾਨਕ ਦਾਸੁ ਮੁਖ ਤੇ ਜੋ ਬੋਲੈ ਈਹਾ ਊਹਾ ਸਚੁ ਹੋਵੈ ॥੨॥੧੪॥੪੫॥ | 5: ਜੋ ਮਾਗਹਿ ਠਾਕੁਰ ਅਪੁਨੇ ਤੇ ਸੋਈ ਸੋਈ ਦੇਵੈ ॥ |
| `case_005` | 20 | 29 | 9 | `future_jump` | `wrong_line` | 1: ਕੋਈ ਬੋਲੈ ਰਾਮ ਰਾਮ ਕੋਈ ਖੁਦਾਇ ॥ | 5: ਕੋਈ ਨਾਵੈ ਤੀਰਥਿ ਕੋਈ ਹਜ ਜਾਇ ॥ |
| `case_005` | 120 | 128 | 8 | `adjacent_future` | `wrong_line` | 1: ਕੋਈ ਬੋਲੈ ਰਾਮ ਰਾਮ ਕੋਈ ਖੁਦਾਇ ॥ | 2: ਕੋਈ ਸੇਵੈ ਗੁਸਈਆ ਕੋਈ ਅਲਾਹਿ ॥੧॥ |
| `case_001` | 43 | 50 | 7 | `future_jump` | `wrong_line` | 1: ਜਨਮ ਜਨਮ ਕੇ ਦੂਖ ਨਿਵਾਰੈ ਸੂਕਾ ਮਨੁ ਸਾਧਾਰੈ ॥ | 3: ਮੇਰਾ ਬੈਦੁ ਗੁਰੂ ਗੋਵਿੰਦਾ ॥ |
| `case_001` | 123 | 130 | 7 | `outside_gt_line_set` | `outside_gt_line` | 3: ਮੇਰਾ ਬੈਦੁ ਗੁਰੂ ਗੋਵਿੰਦਾ ॥ | 4: ਹਰਿ ਹਰਿ ਨਾਮੁ ਅਉਖਧੁ ਮੁਖਿ ਦੇਵੈ ਕਾਟੈ ਜਮ ਕੀ ਫੰਧਾ ॥੧॥ ਰਹਾਉ... |
| `case_001` | 150 | 157 | 7 | `outside_gt_line_set` | `outside_gt_line` | 6: ਅਪੁਨਾ ਦਾਸੁ ਹਰਿ ਆਪਿ ਉਬਾਰਿਆ ਨਾਨਕ ਨਾਮ ਅਧਾਰਾ ॥੨॥੬॥੩੪॥ | 4: ਹਰਿ ਹਰਿ ਨਾਮੁ ਅਉਖਧੁ ਮੁਖਿ ਦੇਵੈ ਕਾਟੈ ਜਮ ਕੀ ਫੰਧਾ ॥੧॥ ਰਹਾਉ... |
| `case_002` | 31 | 38 | 7 | `future_jump` | `wrong_line` | 2: ਤੁਮ ਮਾਤ ਪਿਤਾ ਹਮ ਬਾਰਿਕ ਤੇਰੇ ॥ | 7: ਤੁਮ ਤੇ ਹੋਇ ਸੁ ਆਗਿਆਕਾਰੀ ॥ |
| `case_005` | 108 | 114 | 6 | `adjacent_backtrack` | `wrong_line` | 2: ਕੋਈ ਸੇਵੈ ਗੁਸਈਆ ਕੋਈ ਅਲਾਹਿ ॥੧॥ | 1: ਕੋਈ ਬੋਲੈ ਰਾਮ ਰਾਮ ਕੋਈ ਖੁਦਾਇ ॥ |
| `case_003` | 40 | 44 | 4 | `adjacent_backtrack` | `wrong_line` | 6: ਨਾਨਕ ਦਾਸੁ ਮੁਖ ਤੇ ਜੋ ਬੋਲੈ ਈਹਾ ਊਹਾ ਸਚੁ ਹੋਵੈ ॥੨॥੧੪॥੪੫॥ | 5: ਜੋ ਮਾਗਹਿ ਠਾਕੁਰ ਅਪੁਨੇ ਤੇ ਸੋਈ ਸੋਈ ਦੇਵੈ ॥ |
| `case_004` | 140 | 144 | 4 | `backtrack_jump` | `wrong_line` | 4: ਖਾਲਿਕੁ ਖਲਕ ਖਲਕ ਮਹਿ ਖਾਲਿਕੁ ਪੂਰਿ ਰਹਿਓ ਸ੍ਰਬ ਠਾਂਈ ॥੧॥ ਰਹਾ... | 1: ਅਵਲਿ ਅਲਹ ਨੂਰੁ ਉਪਾਇਆ ਕੁਦਰਤਿ ਕੇ ਸਭ ਬੰਦੇ ॥ |
| `case_001` | 110 | 113 | 3 | `outside_gt_line_set` | `outside_gt_line` | 2: ਦਰਸਨੁ ਭੇਟਤ ਹੋਤ ਨਿਹਾਲਾ ਹਰਿ ਕਾ ਨਾਮੁ ਬੀਚਾਰੈ ॥੧॥ | 4: ਹਰਿ ਹਰਿ ਨਾਮੁ ਅਉਖਧੁ ਮੁਖਿ ਦੇਵੈ ਕਾਟੈ ਜਮ ਕੀ ਫੰਧਾ ॥੧॥ ਰਹਾਉ... |
| `case_001` | 157 | 160 | 3 | `outside_gt_line_set` | `outside_gt_line` | 3: ਮੇਰਾ ਬੈਦੁ ਗੁਰੂ ਗੋਵਿੰਦਾ ॥ | 4: ਹਰਿ ਹਰਿ ਨਾਮੁ ਅਉਖਧੁ ਮੁਖਿ ਦੇਵੈ ਕਾਟੈ ਜਮ ਕੀ ਫੰਧਾ ॥੧॥ ਰਹਾਉ... |
| `case_003` | 77 | 80 | 3 | `adjacent_backtrack` | `wrong_line` | 6: ਨਾਨਕ ਦਾਸੁ ਮੁਖ ਤੇ ਜੋ ਬੋਲੈ ਈਹਾ ਊਹਾ ਸਚੁ ਹੋਵੈ ॥੨॥੧੪॥੪੫॥ | 5: ਜੋ ਮਾਗਹਿ ਠਾਕੁਰ ਅਪੁਨੇ ਤੇ ਸੋਈ ਸੋਈ ਦੇਵੈ ॥ |
| `case_003` | 160 | 163 | 3 | `future_jump` | `wrong_line` | 2: ਕ੍ਰਿਪਾ ਕਟਾਖੵ ਅਵਲੋਕਨੁ ਕੀਨੋ ਦਾਸ ਕਾ ਦੂਖੁ ਬਿਦਾਰਿਓ ॥੧॥ | 6: ਨਾਨਕ ਦਾਸੁ ਮੁਖ ਤੇ ਜੋ ਬੋਲੈ ਈਹਾ ਊਹਾ ਸਚੁ ਹੋਵੈ ॥੨॥੧੪॥੪੫॥ |
| `case_001` | 148 | 150 | 2 | `backtrack_jump` | `wrong_line` | 6: ਅਪੁਨਾ ਦਾਸੁ ਹਰਿ ਆਪਿ ਉਬਾਰਿਆ ਨਾਨਕ ਨਾਮ ਅਧਾਰਾ ॥੨॥੬॥੩੪॥ | 3: ਮੇਰਾ ਬੈਦੁ ਗੁਰੂ ਗੋਵਿੰਦਾ ॥ |
| `case_002` | 29 | 31 | 2 | `future_jump` | `boundary_wrong` | 2: ਤੁਮ ਮਾਤ ਪਿਤਾ ਹਮ ਬਾਰਿਕ ਤੇਰੇ ॥ | 7: ਤੁਮ ਤੇ ਹੋਇ ਸੁ ਆਗਿਆਕਾਰੀ ॥ |
| `case_002` | 38 | 40 | 2 | `future_jump` | `boundary_wrong` | 2: ਤੁਮ ਮਾਤ ਪਿਤਾ ਹਮ ਬਾਰਿਕ ਤੇਰੇ ॥ | 7: ਤੁਮ ਤੇ ਹੋਇ ਸੁ ਆਗਿਆਕਾਰੀ ॥ |

## Decision Use

- High `adjacent_future` / `adjacent_backtrack`: tune transition penalties and boundary smoothing before larger training.
- High `future_jump` / `backtrack_jump`: line-state path is unstable; Viterbi/loop-align constraints should be tightened.
- High `outside_gt_line_set`: inspect whether the clip labels omit sung repeats; if labels are acceptable, add a no-line/end-of-clip guard.
- High `wrong_shabad`: return to shabad-lock evidence; this report is no longer in the locked-shabad regime.
