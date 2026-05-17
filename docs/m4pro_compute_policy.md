# M4 Pro compute policy

## Current answer

Yes, the 48 GB M4 Pro has been used properly for the experiments we have
actually approved. The completed LoRA runs used PyTorch MPS and emitted
run-card memory telemetry:

- `v5_mac_baseline`: 200 clips, 0.418 h, 3 epochs, 232 s wall-clock, 22.34 GB
  peak MPS driver memory.
- `v5b_mac_diverse`: 2,544 clips, 4.936 h, 3 epochs, 2,870 s wall-clock,
  27.05 GB peak MPS driver memory.
- `v6_mac_scale20`: 12,216 clips, 24.593 h, 3 epochs, 13,568.6 s wall-clock,
  27.24 GB peak MPS driver memory.

That means the machine is not sitting unused because the training stack is
misconfigured. It also means there is still memory headroom, but memory headroom
alone is not a reason to run a larger training job.

## Why we are not pulling/training on all 300h right now

The current bottleneck is not compute. It is experiment validity.

Evidence:

1. `v5b_mac_diverse` scaled data from 0.418 h to 4.936 h and used the M4 Pro
   successfully, but blind/live benchmark score regressed to 65.6%.
2. Phase 2.6 showed the adapter helps under oracle-shabad conditions, so the
   problem is integration / routing / alignment, not "training cannot run."
3. Phase 2.9 produced the best honest runtime result, `phase2_9_loop_align`
   at 91.2%, by improving ID-lock, retro-buffering, and loop/null-aware
   alignment.
4. Phase 2.10 silver diagnostics found broad segment ASR is already strong and
   the v5b adapter is neutral against the base model on held-out shards.
5. The weak silver rows are mostly source-label risks, not clean ASR misses.

So the correct expert decision was:

- Do not spend the M4 Pro budget on a broad all-300h/3-seed run yet.
- Do not scale adapter training just because we have 48 GB.
- Use the machine for exactly one controlled Phase 3 warm-start after Phase 2.13
  produced a better lock policy: fresh shards, strict data-card gates, one
  adapter, then silver/paired/OOS comparison.

That warm-start completed on 2026-05-17. The silver gate is now passed, but only
modestly: `v6_mac_scale20` scored 96.55 mean WRatio / 78.0% exact normalized on
the deterministic 100-row silver slice, versus base 96.29 / 75.0%.

## How to audit the machine

Run:

```bash
make audit-m4pro
```

This writes:

```text
diagnostics/m4pro_compute_audit.md
```

The report checks:

- hardware chip and unified memory;
- PyTorch MPS visibility in the current process;
- completed run-card wall-clock and peak MPS memory;
- data/artifact footprint;
- current compute decision.

Note: Codex sandboxed shell processes may report `mps_available=False`, while a
non-sandboxed terminal or approved command reports `mps_available=True`. The
run cards are the authoritative evidence for completed training: they recorded
`device=mps` and MPS driver memory.

## When to use the 48 GB more aggressively

Use the M4 Pro harder only after one of these happens:

1. Gold OOS validates `phase2_9_loop_align` with no catastrophic case.
2. A clean silver/gold audit identifies true acoustic misses, not label-risk
   rows.
3. A deliberate pivot is documented to test a different model family or
   forced-alignment architecture.

Then the 48 GB plan is:

- verify `gradient_checkpointing=true` with PEFT+MPS in isolation;
- try `batch_size=8, grad_accum=1` only if throughput improves without memory
  pressure;
- upgrade torch to >= 2.8 and accelerate to >= 1.11 before re-enabling MPS
  fp16;
- run Phase 3 only with val/OOS gates, not paired-benchmark confidence alone.

## Current next step

Evaluate the Phase 3 warm-start adapter under the current generic paired
runtime stack:

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
```

Only if paired runtime does not regress should we spend runtime on OOS
diagnostics or a larger training slice. Promotion still requires OOS/gold
validation; paired-benchmark gains alone are not enough.
