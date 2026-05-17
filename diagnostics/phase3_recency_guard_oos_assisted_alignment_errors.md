# Alignment error report

**Diagnostic only.** Uses the benchmark scorer's per-frame details and
groups incorrect frames into contiguous spans. This tells us whether the
next move should be aligner/timing work or acoustic training.

## Inputs

- Predictions: `submissions/oos_v1_assisted_phase3_recency_guard`
- Ground truth: `eval_data/oos_v1/assisted_test`
- Collar: `1s`

## Summary

- Overall frame accuracy: `59.9%` (527/880)
- Error frames: `353`

| Error kind | Frames | Share of errors |
|---|---:|---:|
| `wrong_line` | 199 | 56.4% |
| `unresolved_pred` | 131 | 37.1% |
| `boundary_wrong` | 23 | 6.5% |

## Per Case

| Case | Accuracy | Error frames | Pred segs | GT segs | Error mix |
|---|---:|---:|---:|---:|---|
| `case_001` | 47.2% | 95 | 10 | 8 | boundary_wrong=5, unresolved_pred=60, wrong_line=30 |
| `case_004` | 58.3% | 75 | 9 | 4 | unresolved_pred=71, wrong_line=4 |
| `case_005` | 58.9% | 74 | 13 | 13 | boundary_wrong=4, wrong_line=70 |
| `case_002` | 60.0% | 64 | 4 | 9 | boundary_wrong=11, wrong_line=53 |
| `case_003` | 75.0% | 45 | 7 | 6 | boundary_wrong=3, wrong_line=42 |

## Longest Error Spans

| Case | Start | End | Dur | Kind | GT | Pred |
|---|---:|---:|---:|---|---:|---:|
| `case_004` | 50 | 80 | 30 | `unresolved_pred` | 2 | __no_match__ |
| `case_002` | 1 | 29 | 28 | `wrong_line` | 2 | 7 |
| `case_004` | 80 | 101 | 21 | `unresolved_pred` | null | __no_match__ |
| `case_001` | 10 | 30 | 20 | `unresolved_pred` | 6 | __no_match__ |
| `case_003` | 140 | 160 | 20 | `wrong_line` | 2 | 5 |
| `case_005` | 1 | 20 | 19 | `wrong_line` | 1 | 2 |
| `case_004` | 101 | 118 | 17 | `unresolved_pred` | 6 | __no_match__ |
| `case_005` | 146 | 160 | 14 | `wrong_line` | 6 | 5 |
| `case_001` | 30 | 41 | 11 | `wrong_line` | 6 | 3 |
| `case_002` | 79 | 90 | 11 | `wrong_line` | 1 | 2 |
| `case_001` | 50 | 60 | 10 | `unresolved_pred` | 1 | __no_match__ |
| `case_001` | 113 | 123 | 10 | `unresolved_pred` | 6 | __no_match__ |
| `case_001` | 1 | 10 | 9 | `wrong_line` | 6 | 3 |
| `case_003` | 1 | 10 | 9 | `wrong_line` | 6 | 5 |
| `case_005` | 20 | 29 | 9 | `wrong_line` | 1 | 5 |
| `case_005` | 120 | 128 | 8 | `wrong_line` | 1 | 2 |
| `case_001` | 43 | 50 | 7 | `wrong_line` | 1 | 3 |
| `case_001` | 123 | 130 | 7 | `unresolved_pred` | 3 | __no_match__ |
| `case_001` | 150 | 157 | 7 | `unresolved_pred` | 6 | __no_match__ |
| `case_002` | 31 | 38 | 7 | `wrong_line` | 2 | 7 |
| `case_005` | 108 | 114 | 6 | `wrong_line` | 2 | 1 |
| `case_005` | 130 | 136 | 6 | `wrong_line` | 2 | 4 |
| `case_003` | 40 | 44 | 4 | `wrong_line` | 6 | 5 |
| `case_004` | 140 | 144 | 4 | `wrong_line` | 4 | 1 |
| `case_001` | 110 | 113 | 3 | `unresolved_pred` | 2 | __no_match__ |

## Interpretation Guide

- High `missing_pred`: aligner/no-line behavior is too conservative or ASR chunks are sparse.
- High `wrong_line`: locked-shabad line smoother is choosing the wrong pangti, often a loop/refrain issue.
- High `boundary_wrong`: segment boundaries need timing smoothing; label identity may be right nearby.
- High `unresolved_pred`: predictions are not resolving to the GT shabad's canonical line IDs/text.
