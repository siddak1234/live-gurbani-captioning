# Phase 2.12 silver lock-policy tuning

**Diagnostic only.** This report tunes shabad-lock policy variants using
the paired benchmark plus machine-assisted OOS labels. It is a learning
signal, not a production validation claim and not a replacement for gold
OOS correction.

## Inputs

- ASR cache: `asr_cache`
- ASR tag: `medium_word`
- Corpus dir: `corpus_cache` (10 cached shabads)
- paired: `../live-gurbani-captioning-benchmark-v1/test` — paired benchmark regression/dev labels
- assisted_oos: `eval_data/oos_v1/assisted_test` — machine-assisted OOS silver labels

## Top policies

| Rank | Policy | Macro | Paired | Assisted OOS |
|---:|---|---:|---:|---:|
| 1 | `chunk_vote@45s|min=0` | 67.5% | 9/12 (75.0%) | 3/5 (60.0%) |
| 2 | `chunk_vote@90s|min=0` | 67.5% | 9/12 (75.0%) | 3/5 (60.0%) |
| 3 | `chunk_vote@45s|min=1` | 67.5% | 9/12 (75.0%) | 3/5 (60.0%) |
| 4 | `chunk_vote@90s|min=1` | 67.5% | 9/12 (75.0%) | 3/5 (60.0%) |
| 5 | `chunk_vote@45s|min=10` | 67.5% | 9/12 (75.0%) | 3/5 (60.0%) |
| 6 | `chunk_vote@90s|min=10` | 67.5% | 9/12 (75.0%) | 3/5 (60.0%) |
| 7 | `chunk_vote@45s|min=25` | 67.5% | 9/12 (75.0%) | 3/5 (60.0%) |
| 8 | `chunk_vote@90s|min=25` | 67.5% | 9/12 (75.0%) | 3/5 (60.0%) |
| 9 | `chunk_vote@45s|min=50` | 67.5% | 9/12 (75.0%) | 3/5 (60.0%) |
| 10 | `chunk_vote@90s|min=50` | 67.5% | 9/12 (75.0%) | 3/5 (60.0%) |
| 11 | `chunk_vote@45s|min=85` | 67.5% | 9/12 (75.0%) | 3/5 (60.0%) |
| 12 | `chunk_vote@90s|min=85` | 67.5% | 9/12 (75.0%) | 3/5 (60.0%) |

## Guardrail views

| View | Policy | Macro | Paired | Assisted OOS |
|---|---|---:|---:|---:|
| best macro | `chunk_vote@45s|min=0` | 67.5% | 9/12 (75.0%) | 3/5 (60.0%) |
| best paired | `chunk_vote@45s|min=0` | 67.5% | 9/12 (75.0%) | 3/5 (60.0%) |
| best assisted OOS | `tfidf_then_topk3@45s|min=0` | 62.5% | 3/12 (25.0%) | 5/5 (100.0%) |
| best assisted OOS with paired >=75% | `chunk_vote@45s|min=0` | 67.5% | 9/12 (75.0%) | 3/5 (60.0%) |

## Current silver decision

Best macro policy: `chunk_vote@45s|min=0`.

- Paired lock accuracy: 9/12 (75.0%)
- Assisted-OOS lock accuracy: 3/5 (60.0%)
- Silver macro objective: 67.5%

Do **not** promote this as a runtime default yet. The paired set is
still a regression guardrail, and this search is intentionally tiny.
The useful output is the policy shape to test next under full frame
scoring, not a final architecture decision.

## Best-policy per-case decisions

| Dataset | Case | GT | Pred | Window | Mode | Score | Runner-up | Result |
|---|---|---:|---:|---:|---|---:|---|---|
| paired | IZOsmkdmmcg | 4377 | 4377 | 45s | `chunk_vote` | 332.7 | 1341 (0.0) | OK |
| paired | IZOsmkdmmcg_cold33 | 4377 | 4377 | 45s | `chunk_vote` | 332.7 | 1341 (0.0) | OK |
| paired | IZOsmkdmmcg_cold66 | 4377 | 4377 | 45s | `chunk_vote` | 332.7 | 1341 (0.0) | OK |
| paired | kZhIA8P6xWI | 1821 | 1821 | 45s | `chunk_vote` | 85.5 | 4892 (50.0) | OK |
| paired | kZhIA8P6xWI_cold33 | 1821 | 1821 | 45s | `chunk_vote` | 85.5 | 4892 (50.0) | OK |
| paired | kZhIA8P6xWI_cold66 | 1821 | 1821 | 45s | `chunk_vote` | 85.5 | 4892 (50.0) | OK |
| paired | kchMJPK9Axs | 1341 | 1341 | 45s | `chunk_vote` | 177.8 | 1821 (0.0) | OK |
| paired | kchMJPK9Axs_cold33 | 1341 | 1341 | 45s | `chunk_vote` | 177.8 | 1821 (0.0) | OK |
| paired | kchMJPK9Axs_cold66 | 1341 | 1341 | 45s | `chunk_vote` | 177.8 | 1821 (0.0) | OK |
| paired | zOtIpxMT9hU | 3712 | 4892 | 45s | `chunk_vote` | 342.0 | 1341 (0.0) | BAD |
| paired | zOtIpxMT9hU_cold33 | 3712 | 4892 | 45s | `chunk_vote` | 342.0 | 1341 (0.0) | BAD |
| paired | zOtIpxMT9hU_cold66 | 3712 | 4892 | 45s | `chunk_vote` | 342.0 | 1341 (0.0) | BAD |
| assisted_oos | case_001 | 2333 | 2361 | 45s | `chunk_vote` | 171.0 | 1821 (85.5) | BAD |
| assisted_oos | case_002 | 906 | 906 | 45s | `chunk_vote` | 430.5 | 1341 (0.0) | OK |
| assisted_oos | case_003 | 2600 | 2600 | 45s | `chunk_vote` | 308.6 | 4892 (60.0) | OK |
| assisted_oos | case_004 | 4892 | 1341 | 45s | `chunk_vote` | 171.0 | 1821 (0.0) | BAD |
| assisted_oos | case_005 | 3297 | 3297 | 45s | `chunk_vote` | 64.9 | 1341 (0.0) | OK |

## Interpretation

This report allows us to keep learning without waiting on human OOS
correction. It should be used to choose the next runtime experiment.
It should not be used to report final model quality.
