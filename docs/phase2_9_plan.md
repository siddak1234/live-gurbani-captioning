# Phase 2.9 — full-shabad alignment prototype

**Status:** next active phase.

Phase 2.8 closed the cheap-probe branch:

- word timestamps are useful for shabad ID (`12/12` locks in
  `phase2_8_idlock_preword`);
- VAD-on is not viable for sung kirtan;
- generic post-lock Viterbi smoothers regress badly (`77.2%` / `77.1%`);
- current best honest runtime is still `86.6%`, below the `>=87.0%` Phase 2.8
  gate and far below the 95% production target.

The next move is not more ASR knob sweeping and not larger LoRA training. It is
a real alignment layer after shabad lock.

## Hypothesis

Once the shabad is correctly locked, the remaining error is a sequence-alignment
problem:

- ASR chunks contain useful but noisy evidence;
- short cold-start windows lose line-transition context;
- refrain/rahao loops create legal non-monotonic returns;
- per-chunk argmax and generic line-distance Viterbi cannot model those loops.

A full-shabad aligner that sees the whole locked shabad sequence and encodes
legal repeat structure should beat `phase2_8_idlock_preword` without a
benchmark shabad route table.

## Architecture

Keep the existing clean layer split:

1. **Pre-lock shabad ID** stays as Phase 2.8's best path:
   faster-whisper `medium`, `word_timestamps=True`, 30s lookback.
2. **Post-lock evidence** comes from the v5b/surt ASR chunks and matcher scores.
3. **New aligner** consumes `(time_start, time_end, score_vector)` plus canonical
   line metadata and emits line segments.
4. `src/engine.py` owns only the switch between smoothers/aligners. CLI runners
   remain thin I/O shims.

## Prototype sequence

### 2.9.A — score-lattice inspection

Archive a machine-readable diagnostic for each benchmark case:

- ASR chunks after lock;
- top-5 line scores per chunk;
- current `stay_bias` chosen line;
- Viterbi chosen line;
- GT line at chunk midpoint.

Purpose: confirm whether failures are caused by missing evidence, legal loops,
or timing gaps before building a larger decoder.

**Initial result:** completed in
[`diagnostics/phase2_9_score_lattice`](../diagnostics/phase2_9_score_lattice).
Across 268 post-lock chunks whose midpoint overlaps a GT lyric, local best
matches GT on 235 and stay-bias matches GT on 244 (**91.0%**). The worst
`zOtIpxMT9hU_cold66` case has only 3 GT-overlapping post-lock chunks and local
best is correct on all 3. This points away from "recognition is impossible" and
toward boundary/null handling plus loop-aware alignment.

### 2.9.B — retro-buffered finalization policy

Before building a heavier aligner, test the smallest state-policy change implied
by the score-lattice evidence.

Current `idlock` behavior is **commit cutover**:

- pre-lock engine emits tentative captions before `uem.start + 30s`;
- post-lock engine emits only after commit time;
- buffered post-lock evidence before commit is used as smoother context but is
  not allowed to revise the final transcript.

That is conservative for a UI, but it may be unnecessarily conservative for the
final transcript. In a real live product, pre-lock captions are tentative; after
the shabad is confirmed, the app can revise the buffered window. The new
diagnostic policy is therefore **retro-buffered finalization**:

- pre-lock word timestamps still decide the shabad ID;
- once locked, the post-lock v5b/surt path rewrites the final transcript from
  `uem.start`, not only from commit time;
- no shabad-specific route table and no benchmark-specific timing rule.

This is not a new model. It tests whether the remaining Phase 2.8 miss is caused
by the merge policy discarding already-good post-lock evidence.

Run exactly one default configuration:

```bash
HF_WINDOW_SECONDS=10 python3 scripts/run_idlock_path.py \
  --out-dir submissions/phase2_9_retro_buffered \
  --post-adapter-dir lora_adapters/v5b_mac_diverse \
  --post-context buffered \
  --merge-policy retro-buffered \
  --pre-word-timestamps
```

Decision:

- If this clears `>=87.0%`, archive it as the first positive Phase 2.9
  architecture signal and then OOS-test before promotion.
- If it does not, proceed to 2.9.C loop-aware alignment.

**Result:** `phase2_9_retro_buffered` scored **88.7%** (`3038/3425`) with 12/12
correct locks. This is a positive architecture signal and beats
`phase2_8_idlock_preword` by +2.1 pts. It is not production-promoted yet:
`zOtIpxMT9hU_cold66` remains **57.6%**, just below the no-catastrophic-case
guardrail, and OOS v1 is still owed.

### 2.9.C — loop-aware text-score aligner

Build a first aligner over matcher score vectors, not frame-level CTC. The
2.9.B result narrows the first implementation: the biggest remaining miss is
not another blind-ID failure and not weak line recognition; it is that sparse
cold windows still force simran/filler audio into canonical lyric lines.

The first 2.9.C probe is therefore intentionally small:

- states are canonical line indices plus optional null;
- keep the existing stay-bias line path as the lyric-state baseline;
- add a generic simran/null detector that suppresses chunks dominated by
  repeated `ਵਾਹਿਗੁਰੂ` / `waheguru` tokens instead of forcing them into the
  nearest pangti;
- keep this rule shabad-agnostic and benchmark-agnostic;
- do **not** introduce tuned penalty grids or case-specific timing rules;
- no benchmark-case or shabad-ID route table.

Why this is the right first move:

- `phase2_9_retro_buffered` already scores **88.7%** and 12/12 locks correct;
- `zOtIpxMT9hU_cold66` is the sole sub-60 case at **57.6%**;
- the score lattice shows local best is correct on all 3 lyric-overlapping
  chunks for that case;
- the lost frames are mostly 10-second repeated-simran chunks that the current
  smoother maps to line 3.

If this clears the catastrophic-case guardrail, the next 2.9.C increment can
add explicit refrain/rahao loop edges inferred from repeated line text and
nearby high-similarity lines. If it does not, inspect the residual errors before
adding complexity.

This is deliberately less ambitious than Path B's CTC HMM. Past `pb1_hmm`
showed MMS CTC frames are blank-dominated on slow kirtan, while Whisper chunk
text is already discriminative. Start from the evidence that works.

**Result:** `phase2_9_loop_align` scored **91.2%** (`3125/3425`) with 12/12
correct locks. The only code-path change from `phase2_9_retro_buffered` is the
simran-aware null state in `smooth_with_loop_align()`. It clears the paired
score gate and the no-catastrophic-case guardrail:
`zOtIpxMT9hU_cold66` improves from **57.6%** to **86.9%**. This validates the
score-lattice diagnosis. It is still not production-promoted because OOS v1 is
owed.

### 2.9.D — paired benchmark gate

Run exactly one default configuration:

```bash
HF_WINDOW_SECONDS=10 python3 scripts/run_idlock_path.py \
  --out-dir submissions/phase2_9_loop_align \
  --post-adapter-dir lora_adapters/v5b_mac_diverse \
  --post-context buffered \
  --merge-policy retro-buffered \
  --pre-word-timestamps \
  --smoother loop_align
```

Promotion threshold for this phase:

- paired benchmark `>= 87.0%`;
- no catastrophic case below `60%`;
- OOS v1 owed before production promotion.

### 2.9.E — decision

`phase2_9_loop_align` beats `86.6%` and clears the paired guardrails. Proceed
to OOS v1 now. Do **not** start Phase 3 scale-up until OOS v1 exists and this
architecture is evaluated on held-out shabads outside the paired benchmark.

If it fails, do not keep tuning per-chunk penalties. The next branch is a
stronger acoustic timestamp/alignment model:

- IndicConformer kirtan model;
- Whisper word-level timing as segmentation only, v5b text as recognition;
- or a full forced-alignment lattice over canonical text.

## Stop rules

- Do not add shabad-specific route tables.
- Do not count paired-benchmark-only wins as production progress.
- Do not start the 50h/3-seed Phase 3 training budget until an alignment
  prototype clears the paired gate and OOS v1 exists.
- Do not tune dozens of penalty values on the paired benchmark. One default,
  one clear result.
