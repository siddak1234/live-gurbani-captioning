# Phase 2.13 lock-evidence fusion

**Diagnostic only.** This report searches sparse evidence-fusion
policies for shabad locking using paired benchmark labels and
machine-assisted OOS silver labels. It is not a production accuracy
claim.

## Inputs

- ASR cache: `asr_cache`
- ASR tag: `medium_word`
- Corpus dir: `corpus_cache` (10 cached shabads)
- Features: chunk_vote_30, tfidf_30, topk3_30, chunk_vote_45, tfidf_45, topk3_45, chunk_vote_60, tfidf_60, topk3_60, chunk_vote_90, tfidf_90, topk3_90, chunk_vote_120, tfidf_120, topk3_120, chunk_vote_150, tfidf_150, topk3_150, chunk_vote_180, tfidf_180, topk3_180
- paired: `../live-gurbani-captioning-benchmark-v1/test` — paired benchmark regression/dev labels
- assisted_oos: `eval_data/oos_v1/assisted_test` — machine-assisted OOS silver labels

## Top fusion policies

| Rank | Policy | Macro | Paired | Assisted OOS |
|---:|---|---:|---:|---:|
| 1 | `tfidf_45 + chunk_vote_180` | 95.8% | 11/12 (91.7%) | 5/5 (100.0%) |
| 2 | `tfidf_45 + chunk_vote_150` | 95.8% | 11/12 (91.7%) | 5/5 (100.0%) |
| 3 | `tfidf_45 + 0.5*chunk_vote_90` | 95.8% | 11/12 (91.7%) | 5/5 (100.0%) |
| 4 | `tfidf_45 + 0.5*chunk_vote_60` | 95.8% | 11/12 (91.7%) | 5/5 (100.0%) |
| 5 | `tfidf_45 + 0.5*chunk_vote_180` | 95.8% | 11/12 (91.7%) | 5/5 (100.0%) |
| 6 | `tfidf_45 + 0.5*chunk_vote_150` | 95.8% | 11/12 (91.7%) | 5/5 (100.0%) |
| 7 | `tfidf_45 + 0.5*chunk_vote_120` | 95.8% | 11/12 (91.7%) | 5/5 (100.0%) |
| 8 | `chunk_vote_30 + topk3_90` | 95.8% | 11/12 (91.7%) | 5/5 (100.0%) |
| 9 | `chunk_vote_30 + topk3_60` | 95.8% | 11/12 (91.7%) | 5/5 (100.0%) |
| 10 | `chunk_vote_30 + topk3_45` | 95.8% | 11/12 (91.7%) | 5/5 (100.0%) |
| 11 | `chunk_vote_30 + topk3_150` | 95.8% | 11/12 (91.7%) | 5/5 (100.0%) |
| 12 | `chunk_vote_30 + topk3_120` | 95.8% | 11/12 (91.7%) | 5/5 (100.0%) |

## Guardrail views

| View | Policy | Macro | Paired | Assisted OOS |
|---|---|---:|---:|---:|
| best macro | `tfidf_45 + chunk_vote_180` | 95.8% | 11/12 (91.7%) | 5/5 (100.0%) |
| best assisted OOS | `tfidf_45 + chunk_vote_180` | 95.8% | 11/12 (91.7%) | 5/5 (100.0%) |
| best assisted OOS with paired >=75% | `tfidf_45 + chunk_vote_180` | 95.8% | 11/12 (91.7%) | 5/5 (100.0%) |

## Current silver decision

Best macro fusion: `tfidf_45 + chunk_vote_180`.

- Paired lock accuracy: 11/12 (91.7%)
- Assisted-OOS lock accuracy: 5/5 (100.0%)
- Silver macro objective: 95.8%

This is evidence for the next opt-in runtime experiment only if it
improves over Phase 2.12 without paired regression. Gold OOS remains
required before promotion.

## Best-policy per-case decisions

| Dataset | Case | GT | Pred | Score | Runner-up | GT rank | Result |
|---|---|---:|---:|---:|---|---:|---|
| paired | IZOsmkdmmcg | 4377 | 4377 | 1.900 | 3712 (1.000) | 1 | OK |
| paired | IZOsmkdmmcg_cold33 | 4377 | 4377 | 2.000 | 3712 (1.065) | 1 | OK |
| paired | IZOsmkdmmcg_cold66 | 4377 | 4377 | 2.000 | 3712 (0.720) | 1 | OK |
| paired | kZhIA8P6xWI | 1821 | 1821 | 1.707 | 4377 (1.000) | 1 | OK |
| paired | kZhIA8P6xWI_cold33 | 1821 | 1821 | 2.000 | 4377 (0.216) | 1 | OK |
| paired | kZhIA8P6xWI_cold66 | 1821 | 1821 | 1.758 | 906 (1.195) | 1 | OK |
| paired | kchMJPK9Axs | 1341 | 1341 | 2.000 | 2361 (0.083) | 1 | OK |
| paired | kchMJPK9Axs_cold33 | 1341 | 1341 | 2.000 | 2361 (0.082) | 1 | OK |
| paired | kchMJPK9Axs_cold66 | 1341 | 1341 | 1.601 | 2361 (1.000) | 1 | OK |
| paired | zOtIpxMT9hU | 3712 | 4892 | 2.000 | 3712 (0.788) | 2 | BAD |
| paired | zOtIpxMT9hU_cold33 | 3712 | 3712 | 1.000 | 4892 (0.100) | 1 | OK |
| paired | zOtIpxMT9hU_cold66 | 3712 | 3712 | 1.000 | 906 (0.000) | 1 | OK |
| assisted_oos | case_001 | 2333 | 2333 | 2.000 | 2361 (1.521) | 1 | OK |
| assisted_oos | case_002 | 906 | 906 | 2.000 | 2361 (0.043) | 1 | OK |
| assisted_oos | case_003 | 2600 | 2600 | 2.000 | 2361 (0.447) | 1 | OK |
| assisted_oos | case_004 | 4892 | 4892 | 1.149 | 1341 (1.071) | 1 | OK |
| assisted_oos | case_005 | 3297 | 3297 | 1.000 | 1341 (0.314) | 1 | OK |
