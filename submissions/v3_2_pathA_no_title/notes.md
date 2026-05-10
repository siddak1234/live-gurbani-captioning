# v3_2_pathA_no_title

v3.1 + line-0 exclusion. The shabad's heading line (line 0, e.g. "saarag mahalaa
panjavaa") is now skipped at scoring time — it's never sung and never appears in
GT, so allowing it to win on noisy chunks just costs real-segment frames.

## Config

- Identical to v3.1 (BLIND + LIVE + tentative emit, lookback=30s).
- One-line change: `score_chunk()` and `match_chunk()` in `src/matcher.py` return
  0 / skip when `line_idx == 0`.
- **Run**: `python scripts/run_path_a.py --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 --stay-bias 6 --blind --blind-aggregate chunk_vote --blind-lookback 30 --live --tentative-emit --out-dir submissions/v3_2_pathA_no_title`

## Score

**Overall: 86.5%** (2962/3425 frames) — up +0.5 from v3.1.

## Per-shabad: v3.1 → v3.2

| Shabad | Cold0 | Cold33 | Cold66 |
|---|---|---|---|
| IZOsmkdmmcg | 98 → 98 | 96 → 96 | 94 → 94 |
| kZhIA8P6xWI | 87 → 87 | 81 → 81 | 88 → 88 |
| kchMJPK9Axs | 83 → 83 | 76 → 77 | 74 → 76 |
| zOtIpxMT9hU | 93 → 93 | 79 → 79 | 83 → 83 |

Modest but free. The gain is concentrated on kchMJPK9Axs cold66 (+2.2), which is
where the audit found the title-wins frames.

## What's still missing

The audit (kchMJPK9Axs cold33) showed three failure modes; this fix only addresses
one. Remaining:

1. **Long null gaps where Whisper's VAD rejects audio as silence/instrumental.**
   ~50% of remaining failures. Requires re-transcription with `vad_filter=False`
   and/or `word_timestamps=True`. Out of scope for this single-line change.
2. **Wrong-line predictions on clear ASR**: e.g. ASR `"kahu kabeer chhootan nahee"`
   matched to line 9 instead of line 10 despite literal phonetic agreement with
   line 10. Length-mismatch effect in the scorers. Path B territory.

## Artifacts

- 12 submission JSONs
- `tiles.html`
