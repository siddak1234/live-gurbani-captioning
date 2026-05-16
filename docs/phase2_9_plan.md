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

### 2.9.B — loop-aware text-score aligner

Build a first aligner over matcher score vectors, not frame-level CTC:

- states are canonical line indices plus optional null;
- default legal transitions: stay, next line, previous line;
- add explicit refrain/rahao loop edges inferred from repeated line text and
  nearby high-similarity lines;
- allow rare larger jumps only with strong local evidence;
- no benchmark-case or shabad-ID route table.

This is deliberately less ambitious than Path B's CTC HMM. Past `pb1_hmm`
showed MMS CTC frames are blank-dominated on slow kirtan, while Whisper chunk
text is already discriminative. Start from the evidence that works.

### 2.9.C — paired benchmark gate

Run exactly one default configuration:

```bash
HF_WINDOW_SECONDS=10 python3 scripts/run_idlock_path.py \
  --out-dir submissions/phase2_9_loop_align \
  --post-adapter-dir lora_adapters/v5b_mac_diverse \
  --post-context buffered \
  --pre-word-timestamps \
  --smoother loop_align
```

Promotion threshold for this phase:

- paired benchmark `>= 87.0%`;
- no catastrophic case below `60%`;
- OOS v1 owed before production promotion.

### 2.9.D — decision

If loop-aware text-score alignment beats `86.6%`, proceed to OOS v1 and then
Phase 3 scale-up only if OOS is acceptable.

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
