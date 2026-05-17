# Alignment error report

**Diagnostic only.** Uses the benchmark scorer's per-frame details and
groups incorrect frames into contiguous spans. This tells us whether the
next move should be aligner/timing work or acoustic training.

## Inputs

- Predictions: `submissions/phase3_recency_guard_paired`
- Ground truth: `../live-gurbani-captioning-benchmark-v1/test`
- Collar: `1s`

## Summary

- Overall frame accuracy: `91.0%` (3118/3425)
- Error frames: `307`

| Error kind | Frames | Share of errors |
|---|---:|---:|
| `wrong_line` | 194 | 63.2% |
| `boundary_wrong` | 98 | 31.9% |
| `missing_pred` | 15 | 4.9% |

## Per Case

| Case | Accuracy | Error frames | Pred segs | GT segs | Error mix |
|---|---:|---:|---:|---:|---|
| `kZhIA8P6xWI` | 82.8% | 52 | 19 | 17 | boundary_wrong=7, wrong_line=45 |
| `kZhIA8P6xWI_cold33` | 84.1% | 33 | 14 | 13 | boundary_wrong=5, wrong_line=28 |
| `kZhIA8P6xWI_cold66` | 84.8% | 16 | 7 | 7 | wrong_line=16 |
| `zOtIpxMT9hU_cold66` | 87.9% | 12 | 5 | 7 | missing_pred=5, wrong_line=7 |
| `kchMJPK9Axs` | 88.6% | 74 | 20 | 18 | boundary_wrong=38, wrong_line=36 |
| `kchMJPK9Axs_cold33` | 90.6% | 41 | 15 | 13 | boundary_wrong=28, wrong_line=13 |
| `zOtIpxMT9hU_cold33` | 93.9% | 12 | 8 | 10 | missing_pred=5, wrong_line=7 |
| `IZOsmkdmmcg_cold66` | 94.2% | 9 | 6 | 6 | wrong_line=9 |
| `IZOsmkdmmcg` | 94.7% | 24 | 16 | 15 | boundary_wrong=10, wrong_line=14 |
| `kchMJPK9Axs_cold66` | 95.5% | 10 | 6 | 6 | boundary_wrong=10 |
| `zOtIpxMT9hU` | 95.8% | 12 | 9 | 11 | missing_pred=5, wrong_line=7 |
| `IZOsmkdmmcg_cold33` | 96.1% | 12 | 11 | 11 | wrong_line=12 |

## Longest Error Spans

| Case | Start | End | Dur | Kind | GT | Pred |
|---|---:|---:|---:|---|---:|---:|
| `kchMJPK9Axs` | 50 | 70 | 20 | `wrong_line` | 4 | 3 |
| `kchMJPK9Axs` | 130 | 140 | 10 | `boundary_wrong` | null | 8 |
| `kchMJPK9Axs` | 260 | 270 | 10 | `boundary_wrong` | null | 8 |
| `kchMJPK9Axs` | 490 | 500 | 10 | `boundary_wrong` | null | 8 |
| `kchMJPK9Axs_cold33` | 260 | 270 | 10 | `boundary_wrong` | null | 8 |
| `kchMJPK9Axs_cold33` | 490 | 500 | 10 | `boundary_wrong` | null | 8 |
| `kchMJPK9Axs_cold66` | 490 | 500 | 10 | `boundary_wrong` | null | 8 |
| `IZOsmkdmmcg` | 90 | 99 | 9 | `boundary_wrong` | null | 4 |
| `kZhIA8P6xWI` | 43 | 50 | 7 | `wrong_line` | 4 | 3 |
| `kchMJPK9Axs` | 283 | 290 | 7 | `boundary_wrong` | null | 3 |
| `kchMJPK9Axs_cold33` | 283 | 290 | 7 | `boundary_wrong` | null | 3 |
| `kZhIA8P6xWI` | 64 | 70 | 6 | `wrong_line` | 3 | 4 |
| `kZhIA8P6xWI` | 294 | 300 | 6 | `wrong_line` | 3 | 4 |
| `kZhIA8P6xWI_cold33` | 294 | 300 | 6 | `wrong_line` | 3 | 4 |
| `kZhIA8P6xWI_cold66` | 294 | 300 | 6 | `wrong_line` | 3 | 4 |
| `IZOsmkdmmcg` | 325 | 330 | 5 | `wrong_line` | 2 | 1 |
| `IZOsmkdmmcg_cold33` | 325 | 330 | 5 | `wrong_line` | 2 | 1 |
| `IZOsmkdmmcg_cold66` | 325 | 330 | 5 | `wrong_line` | 2 | 1 |
| `kZhIA8P6xWI` | 135 | 140 | 5 | `wrong_line` | 5 | 8 |
| `kZhIA8P6xWI_cold33` | 135 | 140 | 5 | `wrong_line` | 5 | 8 |
| `kchMJPK9Axs` | 345 | 350 | 5 | `wrong_line` | 3 | 6 |
| `kchMJPK9Axs_cold33` | 345 | 350 | 5 | `wrong_line` | 3 | 6 |
| `kZhIA8P6xWI` | 16 | 20 | 4 | `wrong_line` | 3 | 8 |
| `kZhIA8P6xWI` | 130 | 134 | 4 | `boundary_wrong` | null | 8 |
| `kZhIA8P6xWI` | 160 | 164 | 4 | `wrong_line` | 6 | 3 |

## Interpretation Guide

- High `missing_pred`: aligner/no-line behavior is too conservative or ASR chunks are sparse.
- High `wrong_line`: locked-shabad line smoother is choosing the wrong pangti, often a loop/refrain issue.
- High `boundary_wrong`: segment boundaries need timing smoothing; label identity may be right nearby.
- High `unresolved_pred`: predictions are not resolving to the GT shabad's canonical line IDs/text.
