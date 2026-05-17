# Phase 3 warm-start plan

**Status:** v6 warm-start completed, and the first targeted architecture fix
after that run is validated. Silver ASR improved modestly; the original paired
and assisted-OOS frame gates were flat; a generic recency-consistency lock guard
then raised paired frame accuracy from `84.0%` to `91.0%` without assisted-OOS
regression.

This is not full Phase 3 promotion. It is the first large, controlled acoustic
scaling run on the M4 Pro after the lock/alignment stack became strong enough to
make more training worth measuring. It answered a specific question: does a
24.6 h fresh-slice LoRA adapter improve the evidence available to the current
generic lock/alignment runtime?

Answer: **slightly on held-out ASR, not on frame accuracy yet.**

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
   PASS/FLAT.** `v6_mac_scale20` scored `84.0%` overall under the Phase 2.13
   generic fusion-lock runtime. Lock accuracy stayed `11/12`; the same
   full-start `zOtIpxMT9hU -> 4892` false lock remains.
5. **OOS diagnostic gate.** Assisted OOS is still silver, not gold, but it must
   not regress catastrophically. Gold OOS remains the promotion gate. **Status:
   PASS/FLAT.** `v6_mac_scale20` kept assisted-OOS lock accuracy at `5/5`
   and frame accuracy at `59.9%`, matching the Phase 2.13 diagnostic.

## Decision table

| Outcome | Decision |
|---|---|
| Silver improves and paired/OOS do not regress | Continue only if frame accuracy improves or a clear architecture bottleneck is isolated. |
| Silver improves but paired/OOS regress | Acoustic adapter is useful but integration is brittle. Return to lock/alignment before scaling. |
| Silver does not improve | Stop large training. The next bottleneck is architecture/data labels, not data volume. |
| Data-card diversity/holdout fails | Fix data pull. Do not train. |

Current outcome: silver improved, paired/OOS did not regress, but the original
frame-accuracy gates were flat. The recency-consistency diagnostic then
identified one safe-looking generic veto candidate: the full-start
`zOtIpxMT9hU -> 4892` false lock. The opt-in guarded fusion runtime fixed that
specific failure generically and moved paired accuracy to `91.0%`.

This is a meaningful architecture gain. It is still not enough to launch the
full 300h / 3-seed plan, because assisted-OOS frame accuracy remains `59.9%`
even with all OOS locks correct. The next bottleneck is line timing/alignment,
not shabad ID or raw acoustic scale alone.

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

## Paired runtime checkpoint

Command:

```bash
HF_WINDOW_SECONDS=10 python3 scripts/run_idlock_path.py \
  --gt-dir ../live-gurbani-captioning-benchmark-v1/test \
  --audio-dir audio \
  --out-dir submissions/phase3_v6_lock_fusion_paired \
  --post-adapter-dir lora_adapters/v6_mac_scale20 \
  --post-context buffered \
  --merge-policy retro-buffered \
  --pre-word-timestamps \
  --smoother loop_align \
  --blind-lookback 90 \
  --blind-aggregate "fusion:tfidf_45+0.5*chunk_vote_90"
python3 ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/phase3_v6_lock_fusion_paired \
  --gt ../live-gurbani-captioning-benchmark-v1/test
```

Result, 2026-05-17:

- overall paired frame accuracy: `84.0%`
- lock accuracy: `11/12`
- correct full/cold locks: all except full-start `zOtIpxMT9hU`
- remaining false lock: `zOtIpxMT9hU -> 4892` (GT `3712`)
- comparison to Phase 2.13 paired diagnostic: effectively flat (`84.1%` -> `84.0%`)

Interpretation: v6 does not fix the known paired lock failure and does not
create a new paired regression. The next valid gate is assisted OOS diagnostic
under the same runtime. If OOS is flat/improved, v6 can be treated as a small
ASR improvement but not a route to 95% by itself. If OOS regresses, stop
adapter scaling and return to lock/alignment or gold-label quality.

## Assisted-OOS diagnostic checkpoint

Command:

```bash
HF_WINDOW_SECONDS=10 python3 scripts/run_idlock_path.py \
  --gt-dir eval_data/oos_v1/assisted_test \
  --audio-dir eval_data/oos_v1/audio \
  --out-dir submissions/oos_v1_assisted_phase3_v6_lock_fusion \
  --post-adapter-dir lora_adapters/v6_mac_scale20 \
  --post-context buffered \
  --merge-policy retro-buffered \
  --pre-word-timestamps \
  --smoother loop_align \
  --blind-lookback 90 \
  --blind-aggregate "fusion:tfidf_45+0.5*chunk_vote_90"
python3 ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/oos_v1_assisted_phase3_v6_lock_fusion \
  --gt eval_data/oos_v1/assisted_test
```

Result, 2026-05-17:

- overall assisted-OOS frame accuracy: `59.9%`
- lock accuracy: `5/5`
- case scores: `case_001=47.2%`, `case_002=60.0%`, `case_003=75.0%`,
  `case_004=58.3%`, `case_005=58.9%`
- comparison to Phase 2.13 assisted-OOS diagnostic: flat (`59.9%`)

Interpretation: the v6 acoustic adapter does not improve OOS frame alignment,
but it also does not break OOS shabad locking. The limiting factor for the
95%+ goal is now the generic runtime architecture around shabad lock
confidence, delayed evidence, and line-level timing/loop alignment, not simply
more broad acoustic training.

## Current next step

Run and maintain the Phase 3 recency-consistency diagnostic:

```bash
make audit-lock-recency-consistency
```

This compares the current early lock policy (`tfidf_45 + 0.5*chunk_vote_90`)
with a later validation window. A useful next runtime change would be a generic
delay/veto rule only if it catches the known false lock while preserving paired
and assisted-OOS behavior. If the diagnostic shows that any such rule would
hurt OOS or other paired starts, do not implement it; move to line-alignment
error analysis instead.

The next large training run should be launched only after one of these is true:

1. a generic lock/validation change improves paired/OOS frame accuracy, or
2. diagnostics show the remaining errors are true held-out ASR misses that
   larger acoustic training is expected to fix.

## Recency-guard runtime checkpoint

The recency-consistency audit produced one flagged disagreement:

| Dataset | Case | GT | Prefix pred | Late pred | Prefix late support |
|---|---|---:|---:|---:|---:|
| paired | `zOtIpxMT9hU` | 3712 | 4892 | 3712 | 0.090 |

The opt-in runtime aggregate is:

```text
guarded_fusion:tfidf_45+0.5*chunk_vote_90|offset=90|low=0.15|min=0.5
```

Runtime command:

```bash
make eval-paired-recency-guard-v6
make eval-oos-recency-guard-v6-assisted
```

Result, 2026-05-17:

| Runtime | Paired frames | Paired locks | Assisted-OOS frames | Assisted-OOS locks |
|---|---:|---:|---:|---:|
| Phase 2.13 fusion + v6 adapter | 84.0% | 11/12 | 59.9% | 5/5 |
| Phase 3 guarded fusion + v6 adapter | 91.0% | 12/12 | 59.9% | 5/5 |

Interpretation: the guarded fusion rule is a legitimate next runtime candidate.
It is generic, evidence-based, and it fixes the known full-start false lock
without OOS lock regression. It does not solve OOS frame timing, so the next
architecture target is the locked-shabad aligner: boundary offsets, repeated
rahao/loop behavior, and low-confidence no-line spans under correct shabad ID.

## Alignment-error checkpoint

Command:

```bash
make report-paired-recency-guard-alignment
make report-oos-recency-guard-alignment
```

Result, 2026-05-17:

| Set | Accuracy | Error frames | Dominant remaining errors |
|---|---:|---:|---|
| Paired recency guard | 91.0% | 307 | wrong_line 63.2%, boundary_wrong 31.9%, missing_pred 4.9% |
| Assisted OOS recency guard | 59.9% | 353 | wrong_line 56.4%, unresolved_pred 37.1%, boundary_wrong 6.5% |

Interpretation:

- Paired remaining errors are mostly wrong-line and boundary/timing issues, not
  shabad-ID errors.
- Assisted OOS remains weak despite 5/5 correct locks. The two biggest OOS
  buckets are `wrong_line` and `unresolved_pred`, which points to locked-shabad
  line resolution/alignment and assisted-GT/corpus canonical-ID consistency.
- This is the evidence against launching the all-300h / 3-seed training run
  immediately. A larger acoustic adapter may help some ASR text, but it will
  not fix unresolved canonical IDs or loop-align line choices under the correct
  shabad by itself.

Next recommended implementation step: add a locked-shabad alignment diagnostic
that compares predicted `verse_id`/`banidb_gurmukhi` against the case corpus and
reports whether `unresolved_pred` is a corpus/GT ID mismatch or a true wrong
line. If it is a resolution mismatch, fix canonical resolution first. If it is
true wrong-line behavior, tune loop-align/Viterbi scoring on paired + assisted
OOS before spending the next large-training budget.
