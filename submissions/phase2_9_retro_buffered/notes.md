# phase2_9_retro_buffered — ID-lock with retro-buffered finalization

**Decision:** positive architecture signal; not production-promoted yet.

Overall: **88.7%** frame accuracy (`3038/3425`), collar `1s`.

This is the first Phase 2.9 result to beat the current-runtime `86.6%` best path
without a shabad route table.

## What changed

The model stack is unchanged:

- pre-lock shabad ID: faster-whisper `medium`, `word_timestamps=True`, 30s
  lookback;
- post-lock captioning: `surindersinghssj/surt-small-v3` +
  `v5b_mac_diverse`;
- matcher/smoother: `0.5*token_sort_ratio + 0.5*WRatio`, stay-bias=6.

Only the merge policy changed:

- `phase2_8_idlock_preword`: **commit cutover** — pre-lock tentative captions
  are final before `uem.start + 30s`; post-lock captions only appear after
  commit.
- `phase2_9_retro_buffered`: **retro-buffered finalization** — pre-lock output
  still decides shabad ID, but after lock the final transcript is revised from
  `uem.start` using the locked-shabad post engine.

This is state/time based, not benchmark-shabad based. It matches a real UI model
where captions before lock are explicitly tentative and can be corrected once
the shabad is confirmed.

## Command

```bash
HF_WINDOW_SECONDS=10 python3 scripts/run_idlock_path.py \
  --out-dir submissions/phase2_9_retro_buffered \
  --post-adapter-dir lora_adapters/v5b_mac_diverse \
  --post-context buffered \
  --merge-policy retro-buffered \
  --pre-word-timestamps

python3 ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/phase2_9_retro_buffered \
  --gt ../live-gurbani-captioning-benchmark-v1/test
```

## Scores

| Case | Accuracy | Lock |
|---|---:|---|
| IZOsmkdmmcg | 90.3% | 4377 -> 4377 |
| IZOsmkdmmcg_cold33 | 89.6% | 4377 -> 4377 |
| IZOsmkdmmcg_cold66 | 81.4% | 4377 -> 4377 |
| kZhIA8P6xWI | 84.5% | 1821 -> 1821 |
| kZhIA8P6xWI_cold33 | 86.5% | 1821 -> 1821 |
| kZhIA8P6xWI_cold66 | 84.8% | 1821 -> 1821 |
| kchMJPK9Axs | 93.2% | 1341 -> 1341 |
| kchMJPK9Axs_cold33 | 95.2% | 1341 -> 1341 |
| kchMJPK9Axs_cold66 | 100.0% | 1341 -> 1341 |
| zOtIpxMT9hU | 85.4% | 3712 -> 3712 |
| zOtIpxMT9hU_cold33 | 78.6% | 3712 -> 3712 |
| zOtIpxMT9hU_cold66 | 57.6% | 3712 -> 3712 |

## Interpretation

This validates the Phase 2.9.A diagnosis: much of the missing score was caused
by the final merge policy discarding useful locked-shabad evidence in the
buffered pre-lock window.

Compared with `phase2_8_idlock_preword`:

- overall improves **+2.1 pts** (`86.6% -> 88.7%`);
- `kZhIA8P6xWI_cold66` improves **+6.7 pts** (`78.1% -> 84.8%`);
- `kchMJPK9Axs_cold66` improves **+3.6 pts** (`96.4% -> 100.0%`);
- `zOtIpxMT9hU_cold66` improves **+20.2 pts** (`37.4% -> 57.6%`).

The result clears the paired benchmark threshold (`>=87.0%`) but still fails
the "no catastrophic case below 60%" guardrail by a small margin because
`zOtIpxMT9hU_cold66` is `57.6%`. OOS v1 is also still owed.

## Next

Do not start Phase 3 yet. The next targeted step is loop/null-aware alignment
for `zOtIpxMT9hU_cold66`-style sparse cold windows, then OOS v1. This result is
the first positive Phase 2.9 architecture signal, not a deployment claim.
