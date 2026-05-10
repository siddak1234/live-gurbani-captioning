# v0_empty

The 26% baseline. Every submission file has `"segments": []` — no predictions
at all. Verifies our scoring loop talks to the benchmark correctly.

The benchmark scores empty submissions at ~26% because gaps between GT
segments accept `null` as a correct prediction. Anything below this is a bug
in the scorer or our submission format; anything above is real engine value.

## Score: 26.0% (889/3425 frames)

## Reproduce

```bash
python scripts/run_benchmark.py
python ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/v0_empty/ \
  --gt   ../live-gurbani-captioning-benchmark-v1/test/
```
