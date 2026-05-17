# Shabad-lock audit

Diagnostic only. This report scores blind shabad-ID choices from cached
ASR transcripts; it does not run ASR and does not evaluate line timing.

## Inputs

- GT dir: `eval_data/oos_v1/assisted_test`
- ASR cache: `asr_cache`
- ASR tag: `medium_word`
- Corpus dir: `corpus_cache` (10 cached shabads)
- Cases: 5

## Variant summary

| Lookback | Aggregate | Correct | Missing cache |
|---:|---|---:|---:|
| 30s | `chunk_vote` | 2/5 | 0 |
| 30s | `tfidf` | 3/5 | 0 |
| 30s | `topk:3` | 3/5 | 0 |
| 30s | `tfidf_then_topk3` | 3/5 | 0 |
| 45s | `chunk_vote` | 3/5 | 0 |
| 45s | `tfidf` | 4/5 | 0 |
| 45s | `topk:3` | 4/5 | 0 |
| 45s | `tfidf_then_topk3` | 5/5 | 0 |
| 60s | `chunk_vote` | 3/5 | 0 |
| 60s | `tfidf` | 4/5 | 0 |
| 60s | `topk:3` | 4/5 | 0 |
| 60s | `tfidf_then_topk3` | 5/5 | 0 |
| 90s | `chunk_vote` | 3/5 | 0 |
| 90s | `tfidf` | 4/5 | 0 |
| 90s | `topk:3` | 4/5 | 0 |
| 90s | `tfidf_then_topk3` | 5/5 | 0 |

## Per-case details

### 30s / `chunk_vote`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| case_001 | 2333 | 1821 | chunk_vote | 85.5 | 2333 (85.5) | BAD |
| case_002 | 906 | 906 | chunk_vote | 256.5 | 1341 (0.0) | OK |
| case_003 | 2600 | 2600 | chunk_vote | 171.0 | 4892 (60.0) | OK |
| case_004 | 4892 | 906 | chunk_vote | 0.0 | — | BAD |
| case_005 | 3297 | 906 | chunk_vote | 0.0 | — | BAD |

### 30s / `tfidf`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| case_001 | 2333 | 2333 | tfidf | 29.1 | 2361 (14.7) | OK |
| case_002 | 906 | 906 | tfidf | 46.6 | 1341 (0.0) | OK |
| case_003 | 2600 | 2600 | tfidf | 29.1 | 1341 (0.0) | OK |
| case_004 | 4892 | 906 | tfidf | 0.0 | — | BAD |
| case_005 | 3297 | 906 | tfidf | 0.0 | — | BAD |

### 30s / `topk:3`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| case_001 | 2333 | 2333 | topk:3 | 256.5 | 2361 (256.5) | OK |
| case_002 | 906 | 906 | topk:3 | 256.5 | 3297 (137.6) | OK |
| case_003 | 2600 | 2600 | topk:3 | 134.3 | 3297 (131.9) | OK |
| case_004 | 4892 | 906 | topk:3 | 0.0 | — | BAD |
| case_005 | 3297 | 906 | topk:3 | 0.0 | — | BAD |

### 30s / `tfidf_then_topk3`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| case_001 | 2333 | 2333 | tfidf | 29.1 | 2361 (14.7) | OK |
| case_002 | 906 | 906 | tfidf | 46.6 | 1341 (0.0) | OK |
| case_003 | 2600 | 2600 | tfidf | 29.1 | 1341 (0.0) | OK |
| case_004 | 4892 | 906 | topk:3 | 0.0 | — | BAD |
| case_005 | 3297 | 906 | topk:3 | 0.0 | — | BAD |

### 45s / `chunk_vote`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| case_001 | 2333 | 2361 | chunk_vote | 171.0 | 1821 (85.5) | BAD |
| case_002 | 906 | 906 | chunk_vote | 430.5 | 1341 (0.0) | OK |
| case_003 | 2600 | 2600 | chunk_vote | 308.6 | 4892 (60.0) | OK |
| case_004 | 4892 | 1341 | chunk_vote | 171.0 | 1821 (0.0) | BAD |
| case_005 | 3297 | 3297 | chunk_vote | 64.9 | 1341 (0.0) | OK |

### 45s / `tfidf`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| case_001 | 2333 | 2333 | tfidf | 24.8 | 2361 (12.9) | OK |
| case_002 | 906 | 906 | tfidf | 54.1 | 2361 (2.3) | OK |
| case_003 | 2600 | 2600 | tfidf | 29.1 | 1341 (0.0) | OK |
| case_004 | 4892 | 4892 | tfidf | 21.3 | 4377 (4.3) | OK |
| case_005 | 3297 | 1341 | tfidf | 0.0 | 1821 (0.0) | BAD |

### 45s / `topk:3`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| case_001 | 2333 | 2333 | topk:3 | 256.5 | 2361 (256.5) | OK |
| case_002 | 906 | 2361 | topk:3 | 219.6 | 906 (178.9) | BAD |
| case_003 | 2600 | 2600 | topk:3 | 216.8 | 3297 (142.8) | OK |
| case_004 | 4892 | 4892 | topk:3 | 211.9 | 1821 (168.7) | OK |
| case_005 | 3297 | 3297 | topk:3 | 160.5 | 3712 (151.3) | OK |

### 45s / `tfidf_then_topk3`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| case_001 | 2333 | 2333 | tfidf | 24.8 | 2361 (12.9) | OK |
| case_002 | 906 | 906 | tfidf | 54.1 | 2361 (2.3) | OK |
| case_003 | 2600 | 2600 | tfidf | 29.1 | 1341 (0.0) | OK |
| case_004 | 4892 | 4892 | tfidf | 21.3 | 4377 (4.3) | OK |
| case_005 | 3297 | 3297 | topk:3 | 160.5 | 3712 (151.3) | OK |

### 60s / `chunk_vote`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| case_001 | 2333 | 2361 | chunk_vote | 256.5 | 1821 (85.5) | BAD |
| case_002 | 906 | 906 | chunk_vote | 430.5 | 1341 (0.0) | OK |
| case_003 | 2600 | 2600 | chunk_vote | 377.5 | 4892 (120.0) | OK |
| case_004 | 4892 | 1341 | chunk_vote | 256.5 | 1821 (0.0) | BAD |
| case_005 | 3297 | 3297 | chunk_vote | 196.4 | 1341 (0.0) | OK |

### 60s / `tfidf`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| case_001 | 2333 | 2333 | tfidf | 25.0 | 2361 (14.8) | OK |
| case_002 | 906 | 906 | tfidf | 54.1 | 2361 (2.3) | OK |
| case_003 | 2600 | 2600 | tfidf | 33.6 | 906 (5.5) | OK |
| case_004 | 4892 | 4892 | tfidf | 21.9 | 4377 (4.8) | OK |
| case_005 | 3297 | 1341 | tfidf | 0.0 | 1821 (0.0) | BAD |

### 60s / `topk:3`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| case_001 | 2333 | 2333 | topk:3 | 256.5 | 2361 (256.5) | OK |
| case_002 | 906 | 2361 | topk:3 | 219.6 | 906 (178.9) | BAD |
| case_003 | 2600 | 2600 | topk:3 | 216.8 | 906 (216.4) | OK |
| case_004 | 4892 | 4892 | topk:3 | 212.1 | 1341 (172.0) | OK |
| case_005 | 3297 | 3297 | topk:3 | 176.6 | 906 (140.0) | OK |

### 60s / `tfidf_then_topk3`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| case_001 | 2333 | 2333 | tfidf | 25.0 | 2361 (14.8) | OK |
| case_002 | 906 | 906 | tfidf | 54.1 | 2361 (2.3) | OK |
| case_003 | 2600 | 2600 | tfidf | 33.6 | 906 (5.5) | OK |
| case_004 | 4892 | 4892 | tfidf | 21.9 | 4377 (4.8) | OK |
| case_005 | 3297 | 3297 | topk:3 | 176.6 | 906 (140.0) | OK |

### 90s / `chunk_vote`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| case_001 | 2333 | 2361 | chunk_vote | 342.0 | 2333 (171.0) | BAD |
| case_002 | 906 | 906 | chunk_vote | 516.0 | 1341 (0.0) | OK |
| case_003 | 2600 | 2600 | chunk_vote | 449.5 | 4892 (120.0) | OK |
| case_004 | 4892 | 1341 | chunk_vote | 256.5 | 1821 (0.0) | BAD |
| case_005 | 3297 | 3297 | chunk_vote | 263.1 | 1341 (85.5) | OK |

### 90s / `tfidf`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| case_001 | 2333 | 2333 | tfidf | 28.0 | 2361 (13.2) | OK |
| case_002 | 906 | 906 | tfidf | 53.3 | 2361 (1.9) | OK |
| case_003 | 2600 | 2600 | tfidf | 33.6 | 906 (5.5) | OK |
| case_004 | 4892 | 4892 | tfidf | 21.9 | 4377 (4.8) | OK |
| case_005 | 3297 | 3712 | tfidf | 8.8 | 3297 (8.6) | BAD |

### 90s / `topk:3`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| case_001 | 2333 | 2333 | topk:3 | 256.5 | 2361 (256.5) | OK |
| case_002 | 906 | 2361 | topk:3 | 219.6 | 906 (172.0) | BAD |
| case_003 | 2600 | 2600 | topk:3 | 220.1 | 906 (216.4) | OK |
| case_004 | 4892 | 4892 | topk:3 | 212.1 | 1341 (172.0) | OK |
| case_005 | 3297 | 3297 | topk:3 | 213.7 | 3712 (177.9) | OK |

### 90s / `tfidf_then_topk3`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| case_001 | 2333 | 2333 | tfidf | 28.0 | 2361 (13.2) | OK |
| case_002 | 906 | 906 | tfidf | 53.3 | 2361 (1.9) | OK |
| case_003 | 2600 | 2600 | tfidf | 33.6 | 906 (5.5) | OK |
| case_004 | 4892 | 4892 | tfidf | 21.9 | 4377 (4.8) | OK |
| case_005 | 3297 | 3297 | topk:3 | 213.7 | 3712 (177.9) | OK |
