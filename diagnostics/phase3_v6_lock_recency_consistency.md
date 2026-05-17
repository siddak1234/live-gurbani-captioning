# Phase 3 lock recency-consistency audit

**Diagnostic only.** This report compares the current early shabad
lock against a later validation window. It is meant to guide the next
generic architecture step; it is not a promotion-grade accuracy claim
and it must not be turned into a case-specific route table.

## Inputs

- ASR cache: `/Users/sbhatia/Desktop/live-gurbani-captioning/asr_cache`
- ASR tag: `medium_word`
- Corpus dir: `/Users/sbhatia/Desktop/live-gurbani-captioning/corpus_cache` (10 cached shabads)
- Prefix policy: `tfidf_45 + 0.5*chunk_vote_90`
- Validation offset: `90s` after each case's `uem.start`
- Flag rule: prefix winner differs from validation winner, prefix winner
  late-window score <= `0.15`, validation winner
  score >= `0.5`
- paired: `../live-gurbani-captioning-benchmark-v1/test` — paired benchmark regression/dev labels
- assisted_oos: `eval_data/oos_v1/assisted_test` — machine-assisted OOS silver labels

## Summary

- Prefix lock accuracy: `16/17`
- Validation-window lock accuracy: `14/17`
- Flagged recency disagreements: `1`

Flagged rows are not automatic fixes. They are candidates for a
generic delayed/veto policy that must preserve paired + OOS behavior.

| Dataset | Case | GT | Prefix pred | Prefix score | Late pred | Late score | Prefix late support | Late GT rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| paired | zOtIpxMT9hU | 3712 | 4892 | 1.500 | 3712 | 0.500 | 0.090 | 1 |

## All Rows

| Dataset | Case | GT | Prefix pred | Prefix OK | Prefix score | Runner-up | Late pred | Late OK | Late score | Prefix late support | Late GT rank | Flag |
|---|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---|
| paired | IZOsmkdmmcg | 4377 | 4377 | yes | 1.400 | 3712 | 1821 | no | 1.418 | 0.801 | 3 |  |
| paired | IZOsmkdmmcg_cold33 | 4377 | 4377 | yes | 1.500 | 3712 | 4377 | yes | 1.500 | 1.500 | 1 |  |
| paired | IZOsmkdmmcg_cold66 | 4377 | 4377 | yes | 1.500 | 3712 | 3712 | no | 1.000 | 0.900 | 2 |  |
| paired | kZhIA8P6xWI | 1821 | 1821 | yes | 1.207 | 4377 | 1821 | yes | 1.500 | 1.500 | 1 |  |
| paired | kZhIA8P6xWI_cold33 | 1821 | 1821 | yes | 1.500 | 4377 | 1821 | yes | 1.258 | 1.258 | 1 |  |
| paired | kZhIA8P6xWI_cold66 | 1821 | 1821 | yes | 1.258 | 906 | 1821 | yes | 1.478 | 1.478 | 1 |  |
| paired | kchMJPK9Axs | 1341 | 1341 | yes | 1.500 | 2361 | 1341 | yes | 1.500 | 1.500 | 1 |  |
| paired | kchMJPK9Axs_cold33 | 1341 | 1341 | yes | 1.500 | 2361 | 1341 | yes | 1.500 | 1.500 | 1 |  |
| paired | kchMJPK9Axs_cold66 | 1341 | 1341 | yes | 1.101 | 2361 | 1341 | yes | 1.101 | 1.101 | 1 |  |
| paired | zOtIpxMT9hU | 3712 | 4892 | no | 1.500 | 3712 | 3712 | yes | 0.500 | 0.090 | 1 | FLAG |
| paired | zOtIpxMT9hU_cold33 | 3712 | 3712 | yes | 0.500 | 4892 | 3712 | yes | 0.500 | 0.500 | 1 |  |
| paired | zOtIpxMT9hU_cold66 | 3712 | 3712 | yes | 0.500 | 906 | 3712 | yes | 0.500 | 0.500 | 1 |  |
| assisted_oos | case_001 | 2333 | 2333 | yes | 1.250 | 2361 | 2333 | yes | 1.500 | 1.500 | 1 |  |
| assisted_oos | case_002 | 906 | 906 | yes | 1.500 | 2361 | 906 | yes | 1.500 | 1.500 | 1 |  |
| assisted_oos | case_003 | 2600 | 2600 | yes | 1.500 | 4892 | 2361 | no | 1.500 | 0.647 | 2 |  |
| assisted_oos | case_004 | 4892 | 4892 | yes | 1.000 | 1341 | 4892 | yes | 1.149 | 1.149 | 1 |  |
| assisted_oos | case_005 | 3297 | 3297 | yes | 0.500 | 1341 | 3297 | yes | 1.474 | 1.474 | 1 |  |

## Decision

Do not start full 300h / multi-seed training from this checkpoint.
The completed v6 warm-start improved held-out ASR slightly, but paired
and assisted-OOS frame accuracy stayed flat. The next highest-leverage
step is a generic lock/alignment change that addresses flagged
recency disagreements without weakening OOS behavior.
