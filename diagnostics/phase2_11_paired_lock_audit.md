# Shabad-lock audit

Diagnostic only. This report scores blind shabad-ID choices from cached
ASR transcripts; it does not run ASR and does not evaluate line timing.

## Inputs

- GT dir: `../live-gurbani-captioning-benchmark-v1/test`
- ASR cache: `asr_cache`
- ASR tag: `medium_word`
- Corpus dir: `corpus_cache` (10 cached shabads)
- Cases: 12

## Variant summary

| Lookback | Aggregate | Correct | Missing cache |
|---:|---|---:|---:|
| 30s | `chunk_vote` | 9/12 | 0 |
| 30s | `tfidf` | 3/12 | 0 |
| 30s | `topk:3` | 3/12 | 0 |
| 30s | `tfidf_then_topk3` | 3/12 | 0 |
| 45s | `chunk_vote` | 9/12 | 0 |
| 45s | `tfidf` | 3/12 | 0 |
| 45s | `topk:3` | 3/12 | 0 |
| 45s | `tfidf_then_topk3` | 3/12 | 0 |
| 60s | `chunk_vote` | 6/12 | 0 |
| 60s | `tfidf` | 3/12 | 0 |
| 60s | `topk:3` | 3/12 | 0 |
| 60s | `tfidf_then_topk3` | 3/12 | 0 |
| 90s | `chunk_vote` | 9/12 | 0 |
| 90s | `tfidf` | 3/12 | 0 |
| 90s | `topk:3` | 6/12 | 0 |
| 90s | `tfidf_then_topk3` | 3/12 | 0 |

## Per-case details

### 30s / `chunk_vote`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| IZOsmkdmmcg | 4377 | 4377 | chunk_vote | 171.0 | 1341 (0.0) | OK |
| IZOsmkdmmcg_cold33 | 4377 | 4377 | chunk_vote | 171.0 | 1341 (0.0) | OK |
| IZOsmkdmmcg_cold66 | 4377 | 4377 | chunk_vote | 171.0 | 1341 (0.0) | OK |
| kZhIA8P6xWI | 1821 | 1821 | chunk_vote | 85.5 | 4892 (50.0) | OK |
| kZhIA8P6xWI_cold33 | 1821 | 1821 | chunk_vote | 85.5 | 4892 (50.0) | OK |
| kZhIA8P6xWI_cold66 | 1821 | 1821 | chunk_vote | 85.5 | 4892 (50.0) | OK |
| kchMJPK9Axs | 1341 | 1341 | chunk_vote | 177.8 | 1821 (0.0) | OK |
| kchMJPK9Axs_cold33 | 1341 | 1341 | chunk_vote | 177.8 | 1821 (0.0) | OK |
| kchMJPK9Axs_cold66 | 1341 | 1341 | chunk_vote | 177.8 | 1821 (0.0) | OK |
| zOtIpxMT9hU | 3712 | 4892 | chunk_vote | 256.5 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold33 | 3712 | 4892 | chunk_vote | 256.5 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold66 | 3712 | 4892 | chunk_vote | 256.5 | 1341 (0.0) | BAD |

### 30s / `tfidf`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| IZOsmkdmmcg | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| IZOsmkdmmcg_cold33 | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| IZOsmkdmmcg_cold66 | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| kZhIA8P6xWI | 1821 | 4377 | tfidf | 12.9 | 1821 (9.1) | BAD |
| kZhIA8P6xWI_cold33 | 1821 | 4377 | tfidf | 12.9 | 1821 (9.1) | BAD |
| kZhIA8P6xWI_cold66 | 1821 | 4377 | tfidf | 12.9 | 1821 (9.1) | BAD |
| kchMJPK9Axs | 1341 | 1341 | tfidf | 62.8 | 2361 (5.2) | OK |
| kchMJPK9Axs_cold33 | 1341 | 1341 | tfidf | 62.8 | 2361 (5.2) | OK |
| kchMJPK9Axs_cold66 | 1341 | 1341 | tfidf | 62.8 | 2361 (5.2) | OK |
| zOtIpxMT9hU | 3712 | 4892 | tfidf | 26.0 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold33 | 3712 | 4892 | tfidf | 26.0 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold66 | 3712 | 4892 | tfidf | 26.0 | 1341 (0.0) | BAD |

### 30s / `topk:3`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| IZOsmkdmmcg | 4377 | 3712 | topk:3 | 256.5 | 3297 (190.1) | BAD |
| IZOsmkdmmcg_cold33 | 4377 | 3712 | topk:3 | 256.5 | 3297 (190.1) | BAD |
| IZOsmkdmmcg_cold66 | 4377 | 3712 | topk:3 | 256.5 | 3297 (190.1) | BAD |
| kZhIA8P6xWI | 1821 | 4377 | topk:3 | 180.0 | 1821 (175.6) | BAD |
| kZhIA8P6xWI_cold33 | 1821 | 4377 | topk:3 | 180.0 | 1821 (175.6) | BAD |
| kZhIA8P6xWI_cold66 | 1821 | 4377 | topk:3 | 180.0 | 1821 (175.6) | BAD |
| kchMJPK9Axs | 1341 | 1341 | topk:3 | 198.4 | 2361 (182.6) | OK |
| kchMJPK9Axs_cold33 | 1341 | 1341 | topk:3 | 198.4 | 2361 (182.6) | OK |
| kchMJPK9Axs_cold66 | 1341 | 1341 | topk:3 | 198.4 | 2361 (182.6) | OK |
| zOtIpxMT9hU | 3712 | 4892 | topk:3 | 221.1 | 3712 (154.2) | BAD |
| zOtIpxMT9hU_cold33 | 3712 | 4892 | topk:3 | 221.1 | 3712 (154.2) | BAD |
| zOtIpxMT9hU_cold66 | 3712 | 4892 | topk:3 | 221.1 | 3712 (154.2) | BAD |

### 30s / `tfidf_then_topk3`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| IZOsmkdmmcg | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| IZOsmkdmmcg_cold33 | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| IZOsmkdmmcg_cold66 | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| kZhIA8P6xWI | 1821 | 4377 | tfidf | 12.9 | 1821 (9.1) | BAD |
| kZhIA8P6xWI_cold33 | 1821 | 4377 | tfidf | 12.9 | 1821 (9.1) | BAD |
| kZhIA8P6xWI_cold66 | 1821 | 4377 | tfidf | 12.9 | 1821 (9.1) | BAD |
| kchMJPK9Axs | 1341 | 1341 | tfidf | 62.8 | 2361 (5.2) | OK |
| kchMJPK9Axs_cold33 | 1341 | 1341 | tfidf | 62.8 | 2361 (5.2) | OK |
| kchMJPK9Axs_cold66 | 1341 | 1341 | tfidf | 62.8 | 2361 (5.2) | OK |
| zOtIpxMT9hU | 3712 | 4892 | tfidf | 26.0 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold33 | 3712 | 4892 | tfidf | 26.0 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold66 | 3712 | 4892 | tfidf | 26.0 | 1341 (0.0) | BAD |

### 45s / `chunk_vote`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| IZOsmkdmmcg | 4377 | 4377 | chunk_vote | 332.7 | 1341 (0.0) | OK |
| IZOsmkdmmcg_cold33 | 4377 | 4377 | chunk_vote | 332.7 | 1341 (0.0) | OK |
| IZOsmkdmmcg_cold66 | 4377 | 4377 | chunk_vote | 332.7 | 1341 (0.0) | OK |
| kZhIA8P6xWI | 1821 | 1821 | chunk_vote | 85.5 | 4892 (50.0) | OK |
| kZhIA8P6xWI_cold33 | 1821 | 1821 | chunk_vote | 85.5 | 4892 (50.0) | OK |
| kZhIA8P6xWI_cold66 | 1821 | 1821 | chunk_vote | 85.5 | 4892 (50.0) | OK |
| kchMJPK9Axs | 1341 | 1341 | chunk_vote | 177.8 | 1821 (0.0) | OK |
| kchMJPK9Axs_cold33 | 1341 | 1341 | chunk_vote | 177.8 | 1821 (0.0) | OK |
| kchMJPK9Axs_cold66 | 1341 | 1341 | chunk_vote | 177.8 | 1821 (0.0) | OK |
| zOtIpxMT9hU | 3712 | 4892 | chunk_vote | 342.0 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold33 | 3712 | 4892 | chunk_vote | 342.0 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold66 | 3712 | 4892 | chunk_vote | 342.0 | 1341 (0.0) | BAD |

### 45s / `tfidf`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| IZOsmkdmmcg | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| IZOsmkdmmcg_cold33 | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| IZOsmkdmmcg_cold66 | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| kZhIA8P6xWI | 1821 | 4377 | tfidf | 12.9 | 1821 (9.1) | BAD |
| kZhIA8P6xWI_cold33 | 1821 | 4377 | tfidf | 12.9 | 1821 (9.1) | BAD |
| kZhIA8P6xWI_cold66 | 1821 | 4377 | tfidf | 12.9 | 1821 (9.1) | BAD |
| kchMJPK9Axs | 1341 | 1341 | tfidf | 62.8 | 2361 (5.2) | OK |
| kchMJPK9Axs_cold33 | 1341 | 1341 | tfidf | 62.8 | 2361 (5.2) | OK |
| kchMJPK9Axs_cold66 | 1341 | 1341 | tfidf | 62.8 | 2361 (5.2) | OK |
| zOtIpxMT9hU | 3712 | 4892 | tfidf | 24.5 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold33 | 3712 | 4892 | tfidf | 24.5 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold66 | 3712 | 4892 | tfidf | 24.5 | 1341 (0.0) | BAD |

### 45s / `topk:3`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| IZOsmkdmmcg | 4377 | 3712 | topk:3 | 256.5 | 4377 (188.1) | BAD |
| IZOsmkdmmcg_cold33 | 4377 | 3712 | topk:3 | 256.5 | 4377 (188.1) | BAD |
| IZOsmkdmmcg_cold66 | 4377 | 3712 | topk:3 | 256.5 | 4377 (188.1) | BAD |
| kZhIA8P6xWI | 1821 | 4377 | topk:3 | 182.4 | 1821 (173.7) | BAD |
| kZhIA8P6xWI_cold33 | 1821 | 4377 | topk:3 | 182.4 | 1821 (173.7) | BAD |
| kZhIA8P6xWI_cold66 | 1821 | 4377 | topk:3 | 182.4 | 1821 (173.7) | BAD |
| kchMJPK9Axs | 1341 | 1341 | topk:3 | 198.4 | 2361 (182.6) | OK |
| kchMJPK9Axs_cold33 | 1341 | 1341 | topk:3 | 198.4 | 2361 (182.6) | OK |
| kchMJPK9Axs_cold66 | 1341 | 1341 | topk:3 | 198.4 | 2361 (182.6) | OK |
| zOtIpxMT9hU | 3712 | 4892 | topk:3 | 256.5 | 3712 (154.6) | BAD |
| zOtIpxMT9hU_cold33 | 3712 | 4892 | topk:3 | 256.5 | 3712 (154.6) | BAD |
| zOtIpxMT9hU_cold66 | 3712 | 4892 | topk:3 | 256.5 | 3712 (154.6) | BAD |

### 45s / `tfidf_then_topk3`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| IZOsmkdmmcg | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| IZOsmkdmmcg_cold33 | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| IZOsmkdmmcg_cold66 | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| kZhIA8P6xWI | 1821 | 4377 | tfidf | 12.9 | 1821 (9.1) | BAD |
| kZhIA8P6xWI_cold33 | 1821 | 4377 | tfidf | 12.9 | 1821 (9.1) | BAD |
| kZhIA8P6xWI_cold66 | 1821 | 4377 | tfidf | 12.9 | 1821 (9.1) | BAD |
| kchMJPK9Axs | 1341 | 1341 | tfidf | 62.8 | 2361 (5.2) | OK |
| kchMJPK9Axs_cold33 | 1341 | 1341 | tfidf | 62.8 | 2361 (5.2) | OK |
| kchMJPK9Axs_cold66 | 1341 | 1341 | tfidf | 62.8 | 2361 (5.2) | OK |
| zOtIpxMT9hU | 3712 | 4892 | tfidf | 24.5 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold33 | 3712 | 4892 | tfidf | 24.5 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold66 | 3712 | 4892 | tfidf | 24.5 | 1341 (0.0) | BAD |

### 60s / `chunk_vote`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| IZOsmkdmmcg | 4377 | 4377 | chunk_vote | 394.7 | 1341 (0.0) | OK |
| IZOsmkdmmcg_cold33 | 4377 | 4377 | chunk_vote | 394.7 | 1341 (0.0) | OK |
| IZOsmkdmmcg_cold66 | 4377 | 4377 | chunk_vote | 394.7 | 1341 (0.0) | OK |
| kZhIA8P6xWI | 1821 | 1341 | chunk_vote | 132.8 | 1821 (85.5) | BAD |
| kZhIA8P6xWI_cold33 | 1821 | 1341 | chunk_vote | 132.8 | 1821 (85.5) | BAD |
| kZhIA8P6xWI_cold66 | 1821 | 1341 | chunk_vote | 132.8 | 1821 (85.5) | BAD |
| kchMJPK9Axs | 1341 | 1341 | chunk_vote | 263.3 | 1821 (0.0) | OK |
| kchMJPK9Axs_cold33 | 1341 | 1341 | chunk_vote | 263.3 | 1821 (0.0) | OK |
| kchMJPK9Axs_cold66 | 1341 | 1341 | chunk_vote | 263.3 | 1821 (0.0) | OK |
| zOtIpxMT9hU | 3712 | 4892 | chunk_vote | 598.5 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold33 | 3712 | 4892 | chunk_vote | 598.5 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold66 | 3712 | 4892 | chunk_vote | 598.5 | 1341 (0.0) | BAD |

### 60s / `tfidf`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| IZOsmkdmmcg | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| IZOsmkdmmcg_cold33 | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| IZOsmkdmmcg_cold66 | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| kZhIA8P6xWI | 1821 | 4377 | tfidf | 14.5 | 1821 (12.9) | BAD |
| kZhIA8P6xWI_cold33 | 1821 | 4377 | tfidf | 14.5 | 1821 (12.9) | BAD |
| kZhIA8P6xWI_cold66 | 1821 | 4377 | tfidf | 14.5 | 1821 (12.9) | BAD |
| kchMJPK9Axs | 1341 | 1341 | tfidf | 68.0 | 4892 (4.0) | OK |
| kchMJPK9Axs_cold33 | 1341 | 1341 | tfidf | 68.0 | 4892 (4.0) | OK |
| kchMJPK9Axs_cold66 | 1341 | 1341 | tfidf | 68.0 | 4892 (4.0) | OK |
| zOtIpxMT9hU | 3712 | 4892 | tfidf | 26.3 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold33 | 3712 | 4892 | tfidf | 26.3 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold66 | 3712 | 4892 | tfidf | 26.3 | 1341 (0.0) | BAD |

### 60s / `topk:3`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| IZOsmkdmmcg | 4377 | 3712 | topk:3 | 256.5 | 4377 (204.7) | BAD |
| IZOsmkdmmcg_cold33 | 4377 | 3712 | topk:3 | 256.5 | 4377 (204.7) | BAD |
| IZOsmkdmmcg_cold66 | 4377 | 3712 | topk:3 | 256.5 | 4377 (204.7) | BAD |
| kZhIA8P6xWI | 1821 | 4377 | topk:3 | 222.9 | 1821 (217.0) | BAD |
| kZhIA8P6xWI_cold33 | 1821 | 4377 | topk:3 | 222.9 | 1821 (217.0) | BAD |
| kZhIA8P6xWI_cold66 | 1821 | 4377 | topk:3 | 222.9 | 1821 (217.0) | BAD |
| kchMJPK9Axs | 1341 | 1341 | topk:3 | 256.5 | 4892 (181.4) | OK |
| kchMJPK9Axs_cold33 | 1341 | 1341 | topk:3 | 256.5 | 4892 (181.4) | OK |
| kchMJPK9Axs_cold66 | 1341 | 1341 | topk:3 | 256.5 | 4892 (181.4) | OK |
| zOtIpxMT9hU | 3712 | 4892 | topk:3 | 256.5 | 3712 (154.5) | BAD |
| zOtIpxMT9hU_cold33 | 3712 | 4892 | topk:3 | 256.5 | 3712 (154.5) | BAD |
| zOtIpxMT9hU_cold66 | 3712 | 4892 | topk:3 | 256.5 | 3712 (154.5) | BAD |

### 60s / `tfidf_then_topk3`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| IZOsmkdmmcg | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| IZOsmkdmmcg_cold33 | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| IZOsmkdmmcg_cold66 | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| kZhIA8P6xWI | 1821 | 4377 | tfidf | 14.5 | 1821 (12.9) | BAD |
| kZhIA8P6xWI_cold33 | 1821 | 4377 | tfidf | 14.5 | 1821 (12.9) | BAD |
| kZhIA8P6xWI_cold66 | 1821 | 4377 | tfidf | 14.5 | 1821 (12.9) | BAD |
| kchMJPK9Axs | 1341 | 1341 | tfidf | 68.0 | 4892 (4.0) | OK |
| kchMJPK9Axs_cold33 | 1341 | 1341 | tfidf | 68.0 | 4892 (4.0) | OK |
| kchMJPK9Axs_cold66 | 1341 | 1341 | tfidf | 68.0 | 4892 (4.0) | OK |
| zOtIpxMT9hU | 3712 | 4892 | tfidf | 26.3 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold33 | 3712 | 4892 | tfidf | 26.3 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold66 | 3712 | 4892 | tfidf | 26.3 | 1341 (0.0) | BAD |

### 90s / `chunk_vote`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| IZOsmkdmmcg | 4377 | 4377 | chunk_vote | 480.2 | 1341 (0.0) | OK |
| IZOsmkdmmcg_cold33 | 4377 | 4377 | chunk_vote | 480.2 | 1341 (0.0) | OK |
| IZOsmkdmmcg_cold66 | 4377 | 4377 | chunk_vote | 480.2 | 1341 (0.0) | OK |
| kZhIA8P6xWI | 1821 | 1821 | chunk_vote | 277.8 | 1341 (132.8) | OK |
| kZhIA8P6xWI_cold33 | 1821 | 1821 | chunk_vote | 277.8 | 1341 (132.8) | OK |
| kZhIA8P6xWI_cold66 | 1821 | 1821 | chunk_vote | 277.8 | 1341 (132.8) | OK |
| kchMJPK9Axs | 1341 | 1341 | chunk_vote | 434.3 | 1821 (0.0) | OK |
| kchMJPK9Axs_cold33 | 1341 | 1341 | chunk_vote | 434.3 | 1821 (0.0) | OK |
| kchMJPK9Axs_cold66 | 1341 | 1341 | chunk_vote | 434.3 | 1821 (0.0) | OK |
| zOtIpxMT9hU | 3712 | 4892 | chunk_vote | 598.5 | 3712 (120.2) | BAD |
| zOtIpxMT9hU_cold33 | 3712 | 4892 | chunk_vote | 598.5 | 3712 (120.2) | BAD |
| zOtIpxMT9hU_cold66 | 3712 | 4892 | chunk_vote | 598.5 | 3712 (120.2) | BAD |

### 90s / `tfidf`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| IZOsmkdmmcg | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| IZOsmkdmmcg_cold33 | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| IZOsmkdmmcg_cold66 | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| kZhIA8P6xWI | 1821 | 4377 | tfidf | 14.8 | 1821 (12.5) | BAD |
| kZhIA8P6xWI_cold33 | 1821 | 4377 | tfidf | 14.8 | 1821 (12.5) | BAD |
| kZhIA8P6xWI_cold66 | 1821 | 4377 | tfidf | 14.8 | 1821 (12.5) | BAD |
| kchMJPK9Axs | 1341 | 1341 | tfidf | 66.0 | 4892 (7.0) | OK |
| kchMJPK9Axs_cold33 | 1341 | 1341 | tfidf | 66.0 | 4892 (7.0) | OK |
| kchMJPK9Axs_cold66 | 1341 | 1341 | tfidf | 66.0 | 4892 (7.0) | OK |
| zOtIpxMT9hU | 3712 | 4892 | tfidf | 26.3 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold33 | 3712 | 4892 | tfidf | 26.3 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold66 | 3712 | 4892 | tfidf | 26.3 | 1341 (0.0) | BAD |

### 90s / `topk:3`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| IZOsmkdmmcg | 4377 | 3712 | topk:3 | 256.5 | 4892 (184.1) | BAD |
| IZOsmkdmmcg_cold33 | 4377 | 3712 | topk:3 | 256.5 | 4892 (184.1) | BAD |
| IZOsmkdmmcg_cold66 | 4377 | 3712 | topk:3 | 256.5 | 4892 (184.1) | BAD |
| kZhIA8P6xWI | 1821 | 1821 | topk:3 | 237.2 | 4377 (216.0) | OK |
| kZhIA8P6xWI_cold33 | 1821 | 1821 | topk:3 | 237.2 | 4377 (216.0) | OK |
| kZhIA8P6xWI_cold66 | 1821 | 1821 | topk:3 | 237.2 | 4377 (216.0) | OK |
| kchMJPK9Axs | 1341 | 1341 | topk:3 | 256.5 | 3712 (183.5) | OK |
| kchMJPK9Axs_cold33 | 1341 | 1341 | topk:3 | 256.5 | 3712 (183.5) | OK |
| kchMJPK9Axs_cold66 | 1341 | 1341 | topk:3 | 256.5 | 3712 (183.5) | OK |
| zOtIpxMT9hU | 3712 | 4892 | topk:3 | 256.5 | 3712 (165.6) | BAD |
| zOtIpxMT9hU_cold33 | 3712 | 4892 | topk:3 | 256.5 | 3712 (165.6) | BAD |
| zOtIpxMT9hU_cold66 | 3712 | 4892 | topk:3 | 256.5 | 3712 (165.6) | BAD |

### 90s / `tfidf_then_topk3`

| Case | GT | Pred | Mode | Score | Runner-up | Result |
|---|---:|---:|---|---:|---|---|
| IZOsmkdmmcg | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| IZOsmkdmmcg_cold33 | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| IZOsmkdmmcg_cold66 | 4377 | 3712 | tfidf | 16.2 | 4377 (14.6) | BAD |
| kZhIA8P6xWI | 1821 | 4377 | tfidf | 14.8 | 1821 (12.5) | BAD |
| kZhIA8P6xWI_cold33 | 1821 | 4377 | tfidf | 14.8 | 1821 (12.5) | BAD |
| kZhIA8P6xWI_cold66 | 1821 | 4377 | tfidf | 14.8 | 1821 (12.5) | BAD |
| kchMJPK9Axs | 1341 | 1341 | tfidf | 66.0 | 4892 (7.0) | OK |
| kchMJPK9Axs_cold33 | 1341 | 1341 | tfidf | 66.0 | 4892 (7.0) | OK |
| kchMJPK9Axs_cold66 | 1341 | 1341 | tfidf | 66.0 | 4892 (7.0) | OK |
| zOtIpxMT9hU | 3712 | 4892 | tfidf | 26.3 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold33 | 3712 | 4892 | tfidf | 26.3 | 1341 (0.0) | BAD |
| zOtIpxMT9hU_cold66 | 3712 | 4892 | tfidf | 26.3 | 1341 (0.0) | BAD |
