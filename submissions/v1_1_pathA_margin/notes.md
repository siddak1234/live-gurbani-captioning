# v1_1_pathA_margin

**Regression. Don't re-run this approach without new ideas.**

After v1.0 (68.4%), I tried adding a "confidence margin gate" to the matcher:
if the top-1 fuzzy score is within `margin_threshold` of top-2, treat the
chunk as ambiguous and emit `null` instead of the top match. Hypothesis:
ambiguous chunks were the source of v1.0's mismatches.

## Score: 67.3% (regressed -1.1 from v1.0)

## Why it regressed

The audit on zOtIpxMT9hU revealed: **margin is correlated with confidence but
not causally**. Some honest-correct matches naturally have low margins (e.g.
when two lines of a shabad share end-rhyme structure, the right line may only
beat the runner-up by ~5 points). The gate killed those correct matches and
turned them into nulls inside GT segments — counted wrong.

So gating on small top1-top2 margin sometimes filters out wrong answers but
also filters out correct answers that just happen to have nearby competitors.

## Lesson

Confidence gates need to be calibrated to the actual error distribution, not
the intuitive prior. We dropped this idea entirely; later wins came from
changing the scorer (`token_sort_ratio`, then a blend) and the smoother
(stay-bias). See v1.3, v1.4, v1.5 notes.

## Run command (for reference)

```bash
python scripts/run_path_a.py --margin 10 --out-dir submissions/v1_1_pathA_margin
```
