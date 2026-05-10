# v4_mlx_large_v3

Exploratory: Path A pipeline + mlx-whisper backend + large-v3 model + lookback=45s.

## Score: 83.8% (still -2.7 below v3.2's 86.5%)

The model upgrade (medium → large-v3) recovered most of the regression seen in
v4_mlx_medium (which scored 70.6% at lookback=30). Increasing
`--blind-lookback` from 30s to 45s was needed because mlx's longer chunks meant
fewer votes-per-buffer for chunk_vote; 30s was producing literal ties on some
cases. At lookback=45 all 12 blind IDs are correct.

## Mixed signal vs faster-whisper baseline

mlx-large-v3 wins on cases where fw-medium was moderate, loses on cases where
fw-medium was already strong:

| Shabad | fw-medium v3.2 | mlx-large-v3 | Δ |
|---|---|---|---|
| IZOsmkdmmcg | 98 / 96 / 94 | 84 / 84 / 83 | -10 to -14 ✗ |
| kZhIA8P6xWI | 87 / 81 / 88 | 97 / 93 / 89 | +1 to +12 ✓ |
| kchMJPK9Axs | 83 / 77 / 76 | 83 / 78 / 82 | flat to +6 |
| zOtIpxMT9hU | 93 / 79 / 83 | 75 / 80 / 84 | -18 to flat |

Pattern is "differently better, not strictly better." Best-of-both ensembling
might land 89-91% but not committed.

## Run command

```bash
python scripts/run_path_a.py --backend mlx_whisper --model large-v3 \
  --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 --stay-bias 6 \
  --blind --blind-aggregate chunk_vote --blind-lookback 45 \
  --live --tentative-emit \
  --out-dir submissions/v4_mlx_large_v3
```

## Conclusion

mlx-whisper is viable as a fast backend but needs separate tuning; the matcher
and stay-bias parameters Path A uses are calibrated for fw chunk granularity.
For Path A's canonical 86.5%, stick with `--backend faster_whisper`. For Path B
or future fast-iteration experiments, mlx is fine — just don't expect drop-in
parity.
