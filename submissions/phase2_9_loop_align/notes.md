# phase2_9_loop_align — retro-buffered ID-lock with simran-aware null alignment

**Decision:** positive architecture checkpoint; paired benchmark gate cleared.
Not production-promoted until OOS v1 exists.

Overall: **91.2%** frame accuracy (`3125/3425`), collar `1s`.

This is the best current honest runtime result and the first non-route-table
runtime candidate above 90%.

## What changed

The model stack is unchanged from `phase2_9_retro_buffered`:

- pre-lock shabad ID: faster-whisper `medium`, `word_timestamps=True`, 30s
  lookback;
- post-lock captioning: `surindersinghssj/surt-small-v3` +
  `v5b_mac_diverse`;
- merge policy: `retro-buffered`, so locked-shabad post output can revise the
  tentative pre-lock window.

Only the post-lock smoother changed:

- `phase2_9_retro_buffered`: stay-bias line tracker, every non-empty ASR chunk
  forced into a canonical line.
- `phase2_9_loop_align`: same stay-bias line tracker, plus a generic null state
  for chunks dominated by repeated `ਵਾਹਿਗੁਰੂ` / `waheguru` simran.

This rule is deliberately generic. It uses chunk text only; no benchmark case,
shabad ID, or timing route table enters the decision.

## Command

```bash
HF_WINDOW_SECONDS=10 python3 scripts/run_idlock_path.py \
  --out-dir submissions/phase2_9_loop_align \
  --post-adapter-dir lora_adapters/v5b_mac_diverse \
  --post-context buffered \
  --merge-policy retro-buffered \
  --pre-word-timestamps \
  --smoother loop_align

python3 ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/phase2_9_loop_align \
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
| zOtIpxMT9hU | 95.5% | 3712 -> 3712 |
| zOtIpxMT9hU_cold33 | 93.4% | 3712 -> 3712 |
| zOtIpxMT9hU_cold66 | 86.9% | 3712 -> 3712 |

## Interpretation

Compared with `phase2_9_retro_buffered`:

- overall improves **+2.5 pts** (`88.7% -> 91.2%`);
- `zOtIpxMT9hU` improves **+10.1 pts** (`85.4% -> 95.5%`);
- `zOtIpxMT9hU_cold33` improves **+14.8 pts** (`78.6% -> 93.4%`);
- `zOtIpxMT9hU_cold66` improves **+29.3 pts** (`57.6% -> 86.9%`).

The score-lattice diagnosis was valid: the failing cold case did not need more
ASR training or shabad-specific routing. It needed a null state for repeated
simran/filler chunks that are not canonical lyric lines.

This also explains why the earlier Viterbi null probe failed: it used a generic
constant-score null state and dropped weak but useful lyric evidence. The new
null rule is lexical and narrow; it removes the exact filler pattern without
lowering low-confidence lyric chunks such as `kZhIA8P6xWI`'s short cold-window
evidence.

## Validity

Strong points:

- no route table;
- 12/12 shabad locks correct;
- no catastrophic case below 60%;
- single pre-declared default probe, no penalty-grid tuning;
- 137/137 unit tests pass.

Limits:

- still benchmark-only;
- OOS v1 is owed before production promotion;
- this is not yet a 95% engine, and it does not replace the need for broader
  OOS/streaming validation.

## Next

Do not start Phase 3 solely from this paired-benchmark result. The next
protocol-correct step is OOS v1: curate the 5-case held-out pack and score
`phase2_9_loop_align` against recordings outside the paired benchmark. If OOS
is acceptable, promote this as the current v0 runtime architecture and then
resume training/forced-alignment work toward the 95% target.
