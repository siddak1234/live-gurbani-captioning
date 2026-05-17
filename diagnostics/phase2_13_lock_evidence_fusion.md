# Phase 2.13 lock-evidence fusion

**Diagnostic only.** This report searches sparse evidence-fusion
policies for shabad locking using paired benchmark labels and
machine-assisted OOS silver labels. It is not a production accuracy
claim.

## Inputs

- ASR cache: `asr_cache`
- ASR tag: `medium_word`
- Corpus dir: `corpus_cache` (10 cached shabads)
- Features: chunk_vote_30, tfidf_30, topk3_30, chunk_vote_45, tfidf_45, topk3_45, chunk_vote_60, tfidf_60, topk3_60, chunk_vote_90, tfidf_90, topk3_90
- paired: `../live-gurbani-captioning-benchmark-v1/test` — paired benchmark regression/dev labels
- assisted_oos: `eval_data/oos_v1/assisted_test` — machine-assisted OOS silver labels

## Top fusion policies

| Rank | Policy | Macro | Paired | Assisted OOS |
|---:|---|---:|---:|---:|
| 1 | `tfidf_60 + 0.5*chunk_vote_90` | 87.5% | 9/12 (75.0%) | 5/5 (100.0%) |
| 2 | `tfidf_45 + 0.5*chunk_vote_90` | 87.5% | 9/12 (75.0%) | 5/5 (100.0%) |
| 3 | `tfidf_45 + 0.5*chunk_vote_60` | 87.5% | 9/12 (75.0%) | 5/5 (100.0%) |
| 4 | `chunk_vote_90 + 2*tfidf_90` | 87.5% | 9/12 (75.0%) | 5/5 (100.0%) |
| 5 | `chunk_vote_60 + 2*tfidf_90` | 87.5% | 9/12 (75.0%) | 5/5 (100.0%) |
| 6 | `chunk_vote_60 + 2*tfidf_60` | 87.5% | 9/12 (75.0%) | 5/5 (100.0%) |
| 7 | `chunk_vote_45 + 2*tfidf_90` | 87.5% | 9/12 (75.0%) | 5/5 (100.0%) |
| 8 | `chunk_vote_45 + 2*tfidf_60` | 87.5% | 9/12 (75.0%) | 5/5 (100.0%) |
| 9 | `chunk_vote_45 + 2*tfidf_45` | 87.5% | 9/12 (75.0%) | 5/5 (100.0%) |
| 10 | `chunk_vote_30 + topk3_90` | 87.5% | 9/12 (75.0%) | 5/5 (100.0%) |
| 11 | `chunk_vote_30 + topk3_60` | 87.5% | 9/12 (75.0%) | 5/5 (100.0%) |
| 12 | `chunk_vote_30 + topk3_45` | 87.5% | 9/12 (75.0%) | 5/5 (100.0%) |

## Guardrail views

| View | Policy | Macro | Paired | Assisted OOS |
|---|---|---:|---:|---:|
| best macro | `tfidf_60 + 0.5*chunk_vote_90` | 87.5% | 9/12 (75.0%) | 5/5 (100.0%) |
| best assisted OOS | `tfidf_60 + 0.5*chunk_vote_90` | 87.5% | 9/12 (75.0%) | 5/5 (100.0%) |
| best assisted OOS with paired >=75% | `tfidf_60 + 0.5*chunk_vote_90` | 87.5% | 9/12 (75.0%) | 5/5 (100.0%) |

## Current silver decision

Best macro fusion: `tfidf_60 + 0.5*chunk_vote_90`.

- Paired lock accuracy: 9/12 (75.0%)
- Assisted-OOS lock accuracy: 5/5 (100.0%)
- Silver macro objective: 87.5%

This is evidence for the next opt-in runtime experiment only if it
improves over Phase 2.12 without paired regression. Gold OOS remains
required before promotion.

## Best-policy per-case decisions

| Dataset | Case | GT | Pred | Score | Runner-up | GT rank | Result |
|---|---|---:|---:|---:|---|---:|---|
| paired | IZOsmkdmmcg | 4377 | 4377 | 1.400 | 3712 (1.000) | 1 | OK |
| paired | IZOsmkdmmcg_cold33 | 4377 | 4377 | 1.400 | 3712 (1.000) | 1 | OK |
| paired | IZOsmkdmmcg_cold66 | 4377 | 4377 | 1.400 | 3712 (1.000) | 1 | OK |
| paired | kZhIA8P6xWI | 1821 | 1821 | 1.387 | 4377 (1.000) | 1 | OK |
| paired | kZhIA8P6xWI_cold33 | 1821 | 1821 | 1.387 | 4377 (1.000) | 1 | OK |
| paired | kZhIA8P6xWI_cold66 | 1821 | 1821 | 1.387 | 4377 (1.000) | 1 | OK |
| paired | kchMJPK9Axs | 1341 | 1341 | 1.500 | 4892 (0.059) | 1 | OK |
| paired | kchMJPK9Axs_cold33 | 1341 | 1341 | 1.500 | 4892 (0.059) | 1 | OK |
| paired | kchMJPK9Axs_cold66 | 1341 | 1341 | 1.500 | 4892 (0.059) | 1 | OK |
| paired | zOtIpxMT9hU | 3712 | 4892 | 1.500 | 3712 (0.100) | 2 | BAD |
| paired | zOtIpxMT9hU_cold33 | 3712 | 4892 | 1.500 | 3712 (0.100) | 2 | BAD |
| paired | zOtIpxMT9hU_cold66 | 3712 | 4892 | 1.500 | 3712 (0.100) | 2 | BAD |
| assisted_oos | case_001 | 2333 | 2333 | 1.250 | 2361 (1.092) | 1 | OK |
| assisted_oos | case_002 | 906 | 906 | 1.500 | 2361 (0.043) | 1 | OK |
| assisted_oos | case_003 | 2600 | 2600 | 1.500 | 4892 (0.193) | 1 | OK |
| assisted_oos | case_004 | 4892 | 4892 | 1.000 | 1341 (0.579) | 1 | OK |
| assisted_oos | case_005 | 3297 | 3297 | 0.500 | 1341 (0.163) | 1 | OK |
