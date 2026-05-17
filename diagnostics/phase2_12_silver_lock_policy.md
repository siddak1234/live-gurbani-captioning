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
| 1 | `tfidf_then_topk3@45s|min=0` | 79.2% | 7/12 (58.3%) | 5/5 (100.0%) |
| 2 | `tfidf_then_topk3@45s|min=1` | 79.2% | 7/12 (58.3%) | 5/5 (100.0%) |
| 3 | `tfidf_then_topk3@45s|min=10` | 79.2% | 7/12 (58.3%) | 5/5 (100.0%) |
| 4 | `tfidf_then_topk3@45s|min=25` | 79.2% | 7/12 (58.3%) | 5/5 (100.0%) |
| 5 | `tfidf_then_topk3@45s|min=50` | 79.2% | 7/12 (58.3%) | 5/5 (100.0%) |
| 6 | `tfidf_then_topk3@45s|min=85` | 79.2% | 7/12 (58.3%) | 5/5 (100.0%) |
| 7 | `tfidf_then_topk3@45s|min=120` | 79.2% | 7/12 (58.3%) | 5/5 (100.0%) |
| 8 | `tfidf_then_topk3@45s|min=170` | 79.2% | 7/12 (58.3%) | 5/5 (100.0%) |
| 9 | `tfidf_then_topk3@45,60s|min=0` | 79.2% | 7/12 (58.3%) | 5/5 (100.0%) |
| 10 | `tfidf_then_topk3@45,60s|min=1` | 79.2% | 7/12 (58.3%) | 5/5 (100.0%) |
| 11 | `tfidf_then_topk3@45,60s|min=10` | 79.2% | 7/12 (58.3%) | 5/5 (100.0%) |
| 12 | `tfidf_then_topk3@30,45s|min=25` | 79.2% | 7/12 (58.3%) | 5/5 (100.0%) |

## Guardrail views

| View | Policy | Macro | Paired | Assisted OOS |
|---|---|---:|---:|---:|
| best macro | `tfidf_then_topk3@45s|min=0` | 79.2% | 7/12 (58.3%) | 5/5 (100.0%) |
| best paired | `chunk_vote@45s|min=0` | 75.8% | 11/12 (91.7%) | 3/5 (60.0%) |
| best assisted OOS | `tfidf_then_topk3@45s|min=0` | 79.2% | 7/12 (58.3%) | 5/5 (100.0%) |
| best assisted OOS with paired >=75% | `chunk_vote@45s|min=0` | 75.8% | 11/12 (91.7%) | 3/5 (60.0%) |

## Current silver decision

Best macro policy: `tfidf_then_topk3@45s|min=0`.

- Paired lock accuracy: 7/12 (58.3%)
- Assisted-OOS lock accuracy: 5/5 (100.0%)
- Silver macro objective: 79.2%

Do **not** promote this as a runtime default yet. The paired set is
still a regression guardrail, and this search is intentionally tiny.
The useful output is the policy shape to test next under full frame
scoring, not a final architecture decision.

## Best-policy per-case decisions

| Dataset | Case | GT | Pred | Window | Mode | Score | Runner-up | Result |
|---|---|---:|---:|---:|---|---:|---|---|
| paired | IZOsmkdmmcg | 4377 | 3712 | 45s | `tfidf` | 16.2 | 4377 (14.6) | BAD |
| paired | IZOsmkdmmcg_cold33 | 4377 | 4377 | 45s | `tfidf` | 17.7 | 3712 (15.1) | OK |
| paired | IZOsmkdmmcg_cold66 | 4377 | 4377 | 45s | `tfidf` | 20.0 | 3712 (14.4) | OK |
| paired | kZhIA8P6xWI | 1821 | 4377 | 45s | `tfidf` | 12.9 | 1821 (9.1) | BAD |
| paired | kZhIA8P6xWI_cold33 | 1821 | 1821 | 45s | `tfidf` | 26.3 | 4377 (5.7) | OK |
| paired | kZhIA8P6xWI_cold66 | 1821 | 906 | 45s | `tfidf` | 27.4 | 1821 (20.7) | BAD |
| paired | kchMJPK9Axs | 1341 | 1341 | 45s | `tfidf` | 62.8 | 2361 (5.2) | OK |
| paired | kchMJPK9Axs_cold33 | 1341 | 1341 | 45s | `tfidf` | 57.8 | 2361 (4.7) | OK |
| paired | kchMJPK9Axs_cold66 | 1341 | 3712 | 45s | `topk:3` | 256.5 | 2361 (217.8) | BAD |
| paired | zOtIpxMT9hU | 3712 | 4892 | 45s | `tfidf` | 24.5 | 1341 (0.0) | BAD |
| paired | zOtIpxMT9hU_cold33 | 3712 | 3712 | 45s | `topk:3` | 161.3 | 4377 (149.1) | OK |
| paired | zOtIpxMT9hU_cold66 | 3712 | 3712 | 45s | `topk:3` | 174.6 | 4377 (153.1) | OK |
| assisted_oos | case_001 | 2333 | 2333 | 45s | `tfidf` | 24.8 | 2361 (12.9) | OK |
| assisted_oos | case_002 | 906 | 906 | 45s | `tfidf` | 54.1 | 2361 (2.3) | OK |
| assisted_oos | case_003 | 2600 | 2600 | 45s | `tfidf` | 29.1 | 1341 (0.0) | OK |
| assisted_oos | case_004 | 4892 | 4892 | 45s | `tfidf` | 21.3 | 4377 (4.3) | OK |
| assisted_oos | case_005 | 3297 | 3297 | 45s | `topk:3` | 160.5 | 3712 (151.3) | OK |

## Interpretation

This report allows us to keep learning without waiting on human OOS
correction. It should be used to choose the next runtime experiment.
It should not be used to report final model quality.
