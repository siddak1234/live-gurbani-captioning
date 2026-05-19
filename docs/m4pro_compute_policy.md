# M4 Pro compute policy

## Current answer

Yes, the 48 GB M4 Pro is being used properly. The project has moved from
"do not scale yet" to a controlled large-slice Phase 3 experiment because the
confirmed loop-align runtime lane reached a local plateau. The current run is
**v7 300h-source epoch 1**, not a blind all-300h/3-seed spend.

Completed LoRA runs used PyTorch MPS and emitted run-card memory telemetry:

- `v5_mac_baseline`: 200 clips, 0.418 h, 3 epochs, 232 s wall-clock, 22.34 GB
  peak MPS driver memory.
- `v5b_mac_diverse`: 2,544 clips, 4.936 h, 3 epochs, 2,870 s wall-clock,
  27.05 GB peak MPS driver memory.
- `v6_mac_scale20`: 12,216 clips, 24.593 h, 3 epochs, 13,568.6 s wall-clock,
  27.24 GB peak MPS driver memory.
- `v7_mac_300h_epoch1` is in progress on 93,292 train clips / 188.61 h train
  audio, with 11,541 validation clips. The first interrupted attempt recorded
  38.82 GB peak MPS driver memory at step 1000, showing this run does use most
  of the 48 GB machine.

That means the machine is not sitting unused because the training stack is
misconfigured. It also means memory headroom alone is not the planning driver:
the gate is whether acoustic scaling improves paired + assisted-OOS runtime
accuracy under the confirmed runtime.

## Why we are running one epoch, not a full 3-seed campaign

The current bottleneck is still experiment validity, not raw compute.

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

So the correct expert decision evolved in two stages:

1. Before confirmed loop-align, do not spend the M4 Pro budget on broad
   all-300h training. Runtime line-path changes were still producing generic
   gains.
2. After confirmed loop-align plateaued at `92.8%` paired / `60.8%`
   assisted-OOS, run exactly one controlled large-slice acoustic-scaling
   experiment: v7 epoch 1 over the 300h-source filtered dataset.

That warm-start completed on 2026-05-17. The silver gate passed, but only
modestly: `v6_mac_scale20` scored 96.55 mean WRatio / 78.0% exact normalized on
the deterministic 100-row silver slice, versus base 96.29 / 75.0%.

The first paired and assisted-OOS runtime gates were flat:

- paired runtime: `84.0%`, lock `11/12`, same full-start
  `zOtIpxMT9hU -> 4892` false lock as Phase 2.13;
- assisted-OOS runtime: `59.9%`, lock `5/5`, same frame accuracy as Phase 2.13.

Conclusion: the M4 Pro did useful work in v6, the runtime lane then produced a
real gain, and the project is now correctly using the 48 GB machine for the next
controlled acoustic-scaling question. The run should continue to epoch end, but
it should **not** automatically expand into 3 epochs / multiple seeds unless the
runtime metrics improve.

The next targeted architecture step did help: a generic recency-consistency
guard lifted paired frame accuracy to `91.0%` and paired locks to `12/12` while
assisted-OOS stayed flat at `59.9%` with `5/5` locks. That moves the project
forward, but it also makes the remaining bottleneck clearer: correct-shabad
line timing/alignment, especially on OOS.

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

Use the recency-guarded runtime as the current paired candidate, then audit
line timing/alignment:

```bash
make eval-paired-recency-guard-v6
make eval-oos-recency-guard-v6-assisted
```

The paired command currently scores `91.0%`; the assisted-OOS command currently
scores `59.9%`. The next implementation should analyze why OOS remains weak
despite `5/5` correct shabad locks.

Current alignment-error reports show:

- paired recency guard: wrong_line `63.2%` of remaining errors,
  boundary_wrong `31.9%`, missing_pred `4.9%`;
- assisted-OOS recency guard: wrong_line `56.4%`, outside_gt_line `37.1%`,
  boundary_wrong `6.5%`.

That makes the immediate next step a locked-shabad aligner/canonical-resolution
diagnostic, not another broad data pull. The OOS errors mostly stay inside the
correct shabad; the problem is the line path chosen inside that shabad.

Follow-up line-path audit:

- paired recency guard: adjacent_backtrack `31.9%`,
  predicted_during_unlabeled_gt `30.0%`, future_jump `17.6%`;
- assisted-OOS recency guard: outside_gt_line_set `30.6%`,
  future_jump `25.8%`, adjacent/backtrack jumps `24.4%` combined.

So the next use of engineering time should be a constrained line-state smoother
and no-line/end-of-clip guard, evaluated from cached predictions first. The next
large 300h run is justified only after this runtime path either improves OOS or
proves the remaining misses are genuinely acoustic.

Two cached probes also narrow the path:

- existing Viterbi with tighter penalties + null state: paired `79.5%`;
- loop-align with stay-bias `10`: paired `89.6%`.
- global threshold `50`: paired `79.9%`;
- confirmed non-adjacent loop-align: paired `92.8%`, assisted-OOS `60.8%`.

The confirmed loop-align smoother is now the active runtime checkpoint because
it clears the `>= 91.0%` paired non-regression gate and improves assisted-OOS
without more training. The next large 300h run should still wait: the remaining
errors are line-path buckets (`adjacent_backtrack`, `outside_gt_line_set`), not
M4 Pro underuse.

Follow-up parameter sweep found a local confirmed-loop plateau:

- `confirm_chunks=2`, `hard_jump_margin=12/15/18/20`: paired `92.8%`;
- assisted-OOS at margins `12/15/20`: `60.8%`;
- stricter guards (`margin=8`, `confirm_chunks=3`) regress paired.

This is the exact condition that justified starting the current v7 epoch-1 run.
It is framed as acoustic-scaling under the confirmed runtime, not proof that the
line-path problem is solved.

## Current next step

The v7 300h-source epoch-1 run completed on 2026-05-18. Held-out validation
loss improved monotonically but only slightly:

```text
0.03620 -> 0.03565 -> 0.03543 -> 0.03532 -> 0.03524 -> 0.03516 -> 0.03514 -> 0.03512 -> 0.03511 -> 0.035107
```

`checkpoint-11000` is the best validated checkpoint, and `run_card.json` reports
`status=completed`, `train_n_clips=93292`, `eval_n_clips=11541`, and peak MPS
memory `35.53 GB`. The M4 Pro was used appropriately for the intended acoustic
scaling experiment; the next question is not "use more machine" but "did this
lower acoustic loss improve runtime captions?"

Therefore the expert move is now:

1. evaluate the best v7 adapter through the confirmed runtime:
   - paired benchmark gate: beat `92.8%`;
   - assisted-OOS gate: beat `60.8%`;
   - locks remain `12/12` paired and `5/5` assisted-OOS;
2. only then decide whether to promote v7, continue to multi-epoch/seed
   training, or pivot back to runtime/architecture.

If v7 improves validation loss but not paired/OOS runtime accuracy, the
architecture implication is clear: the adapter is learning acoustic evidence,
but the remaining blocker is line alignment/candidate resolution, not machine
underuse.
