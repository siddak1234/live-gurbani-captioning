# Phase 3 warm-start plan

**Status:** data and training completed; silver held-out evaluation is the
active gate.

This is not full Phase 3 promotion. It is the first large, controlled acoustic
scaling run on the M4 Pro after the lock/alignment stack became strong enough to
make more training worth measuring.

## Why we are doing this now

The last important checkpoints are:

- `v5b_mac_diverse` proved the M4 Pro training path works, but blind/live score
  regressed to `65.6%`.
- Phase 2.9 loop-align reached `91.2%` paired using a generic runtime
  architecture, but OOS was still weak.
- Phase 2.13 evidence fusion, after fixing the diagnostic to respect
  `uem.start`, gives **11/12 paired locks** and **5/5 assisted-OOS locks**.
- Full-frame fusion scoring is **84.1% paired** and **59.9% assisted OOS**.
  The remaining paired false lock is full-start `zOtIpxMT9hU -> 4892`.
- Tail-window features can overfit that one paired miss but hurt OOS, so they
  are rejected.

Interpretation: the architecture is clean enough to test whether more diverse
acoustic training improves the evidence available to the lock/aligner. It is
not clean enough to spend the full all-300h / 3-seed budget or claim promotion.

## Experiment

Use a fresh slice of the HuggingFace canonical kirtan data that does not overlap
the prior v5b slice or the silver-eval slice.

```bash
make data-v6-scale20
```

The target is idempotent: if `training_data/v6_mac_scale20/manifest.json`
already exists, it skips. Use `make data-v6-scale20 DATA_FORCE=1` only when
intentionally replacing the slice.

Target:

- output: `training_data/v6_mac_scale20/`
- source shards: `20-49`
- clips: `10000`
- quality floor: `canonical_match_score >= 0.88`
- max scan: `80000`
- diversity gates: at least 40 source videos and 300 shabad tokens
- holdouts: benchmark videos, benchmark shabads, benchmark canonical line text,
  and OOS source videos from `configs/datasets.yaml`

If the data-card fails diversity or holdout checks, do not train. Adjust the
pull first.

Actual data checkpoint, 2026-05-17:

- output: `training_data/v6_mac_scale20/`
- clips: `12,216`
- hours: `24.593`
- unique shabad tokens: `524`
- unique source videos: `40`
- data-card diversity gate: `PASS`
- holdout rejections: `holdout_shabad=0`, `holdout_video=0`,
  `holdout_content=0`
- quality rejections: `score_low=16,956`, `dur_long=6`

Then train one adapter:

```bash
make train-v6-scale20
```

Target:

- output: `lora_adapters/v6_mac_scale20/`
- config: `configs/training/surt_lora_mac.yaml`
- expected size: roughly 15-25 h of audio depending on clip length
- expected wall-clock: several hours on M4 Pro fp32 MPS

Actual training checkpoint, 2026-05-17:

- status: `completed`
- optimizer steps: `4,581`
- epochs: `3.0`
- wall-clock: `13,568.6 s` (`3 h 46 m`)
- train throughput: `0.338 steps/s`, `2.701 samples/s`
- final logged train loss in `run_card.json`: `0.028`
- trainer mean `train_loss`: `0.1272`
- peak MPS driver memory: `27.24 GB`
- device: `mps`
- config hash:
  `4e75d4e591280bb389cd31bbc16cedbbc1b88cbc2bcee19928bae04668a7acb4`
- data hash:
  `c6cd0479b65f58161b8cce2f7ded444e30e0dba59887448b819a179347850104`

Interpretation: the Mac training stack and the 24.6 h data pull are healthy.
The very low train loss is not a promotion signal by itself; it can indicate
strong fitting or memorization. The next valid test is held-out silver ASR.

## Validation gates

Training success is not accuracy success. The run must pass the following gates
before it can justify a bigger Phase 3 run.

1. **Run-card gate.** `lora_adapters/v6_mac_scale20/run_card.json` has
   `status=completed`, `device=mps`, stable config/data hashes, and no memory
   pressure. **Status: PASS** for the 2026-05-17 run.
2. **Data-card gate.** `training_data/v6_mac_scale20/data_card.md` confirms
   no benchmark/OOS leakage and sufficient video/shabad diversity. **Status:
   PASS** for the 2026-05-17 pull.
3. **Silver ASR gate.** Evaluate on held-out silver shards `10-19`. The v6
   adapter should beat or at least not regress from base `surt-small-v3` and
   `v5b_mac_diverse` on the same silver slice. **Status: PASS, modest.**
   On the deterministic 100-row silver slice, `v6_mac_scale20` reached
   `96.55` mean WRatio and `78.0%` exact normalized match, versus base
   `96.29` / `75.0%` and `v5b_mac_diverse` `96.33` / `73.0%`.
4. **Paired runtime gate.** Evaluate under the current best generic runtime
   stack. A paired gain that breaks lock behavior is not a win. **Status:
   next.**
5. **OOS diagnostic gate.** Assisted OOS is still silver, not gold, but it must
   not regress catastrophically. Gold OOS remains the promotion gate.

## Decision table

| Outcome | Decision |
|---|---|
| Silver improves and paired/OOS do not regress | Continue toward full Phase 3: larger slice, rank/modules/augmentation, then 3 seeds. |
| Silver improves but paired/OOS regress | Acoustic adapter is useful but integration is brittle. Return to lock/alignment before scaling. |
| Silver does not improve | Stop large training. The next bottleneck is architecture/data labels, not data volume. |
| Data-card diversity/holdout fails | Fix data pull. Do not train. |

## Architecture rule

No route tables, no benchmark-specific shabad IDs, and no case-specific lock
overrides. The runtime remains:

```text
audio -> ASR evidence -> generic shabad lock -> locked-shabad aligner
```

Large training is allowed only because it is now tied to this generic runtime
stack and the validation gates above.

## Silver result checkpoint

Command:

```bash
make eval-silver-300h \
  SILVER_ADAPTER_DIR=lora_adapters/v6_mac_scale20 \
  SILVER_OUT=submissions/silver_300h_v6_mac_scale20.json
```

Result, 2026-05-17:

| Run | n | videos | shabads | mean WRatio | median WRatio | exact normalized |
|---|---:|---:|---:|---:|---:|---:|
| `surt-small-v3` base | 100 | 19 | 28 | 96.29 | 100.00 | 75.0% |
| `surt-small-v3 + v5b_mac_diverse` | 100 | 19 | 28 | 96.33 | 100.00 | 73.0% |
| `surt-small-v3 + v6_mac_scale20` | 100 | 19 | 28 | 96.55 | 100.00 | 78.0% |

Pairwise against the exact same rows:

- `v6` vs base: `+0.264` mean WRatio, 7 rows improved, 2 rows regressed.
- `v6` vs `v5b`: `+0.218` mean WRatio, 7 rows improved, 2 rows regressed.

Interpretation: the 24.6 h warm-start produced a small but real held-out ASR
gain. It is not the large jump needed for a direct path to 95%+, but it clears
the "do not regress on silver" gate. The next experiment should test whether
the stronger adapter helps or hurts the current generic lock/alignment runtime.
