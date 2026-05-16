# Phase 2.7 — runtime ID-lock integration

**Status:** complete — benchmark gate failed.

Phase 2.7 is the checkpoint after the Phase 2.6 diagnostic. The important lesson from Phase 2.6 is not "train larger." The `v5b_mac_diverse` adapter dropped the blind/live benchmark score to `65.6%`, but oracle-shabad alignment improved to `87.4%` and the v3.2-ID-lock proxy scored `87.1%`. That pattern says the model is useful after shabad lock and brittle before it.

The next valid experiment is therefore a runtime architecture experiment, not a training-scale experiment.

## Hypothesis

Use the frozen v3.2 faster-whisper path for the first 30 seconds, where generic blind shabad ID and tentative captions matter most. Once a shabad is committed, switch to the `v5b_mac_diverse` LoRA adapter and match only inside the locked shabad's line set.

If Phase 2.6 was right, this should recover most of the proxy score without benchmark-specific route tables.

## Architecture

The implementation is split by layer:

| Layer | File | Responsibility |
|---|---|---|
| Layer 2 library | `src/idlock_engine.py` | Compose two `engine.predict()` calls by state and time. No argparse, benchmark paths, or filesystem output. |
| Layer 3 runner | `scripts/run_idlock_path.py` | Benchmark-only I/O: load GT/audio/corpus, build configs, serialize submission JSON. |
| Diagnostic-only helper | `scripts/merge_idlock_submissions.py` | Historical Phase 2.6 proxy. Must not be treated as the runtime path. |

Runtime state:

```text
before commit_time = uem.start + 30s:
    pre-lock engine = v3.2/faster-whisper, blind/live, tentative captions

after commit_time:
    post-lock engine = surt-small-v3 + v5b_mac_diverse LoRA, locked shabad_id
```

The default post-lock context mode is **buffered**. In benchmark batch simulation, the post-lock smoother may use the pre-lock transcript as causal context, but emitted segments are clipped to after commit time. This mirrors a streaming system that buffers transcript state during the ID window and switches displayed output only after lock.

There is also a stricter `strict-live` mode where the post-lock engine ignores pre-commit chunks. Use it as an ablation if the buffered runtime result is suspiciously close to the Phase 2.6 proxy.

## Validity Boundaries

This experiment is valid only if:

- The switch is based on time and committed shabad state, never benchmark shabad IDs.
- The pre-lock engine predicts the shabad; the runner must not pass GT shabad ID to the pre-lock path.
- The post-lock engine receives only the committed shabad ID from the pre-lock path.
- Matcher weights are not tuned on the paired benchmark during this phase.
- Results are labeled separately from OOS. Paired benchmark success alone is not promotion to production.

This experiment is not valid if it becomes a hand-written route table, a submission-level merge, or a benchmark-only exception path.

## Success Criteria

Primary paired benchmark gate:

- `v5b_idlock_runtime >= 87.0%` frame accuracy.

Secondary OOS gate:

- `oos_v1` non-regressing vs the frozen v3.2/x4 baselines once the OOS pack is populated.

Interpretation:

| Outcome | Decision |
|---|---|
| `>= 87.0%` paired and OOS non-regressing | Continue adapter path; build streaming/iOS shape and then consider Phase 3 scale. |
| `>= 86.5%` but `< 87.0%` paired | Architecture is competitive with frozen v3.2 but not enough. Run `strict-live` and timing diagnostics before training more. |
| `< 86.5%` paired | Phase 2.6 proxy did not transfer to runtime. Pivot to word timestamps, hybrid timing, or full-shabad forced alignment before more LoRA scale. |

## Commands

```bash
HF_WINDOW_SECONDS=10 python3 scripts/run_idlock_path.py \
  --out-dir submissions/v5b_idlock_runtime \
  --post-adapter-dir lora_adapters/v5b_mac_diverse \
  --post-context buffered

python3 ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/v5b_idlock_runtime \
  --gt ../live-gurbani-captioning-benchmark-v1/test
```

If buffered passes but looks too close to the proxy, run:

```bash
HF_WINDOW_SECONDS=10 python3 scripts/run_idlock_path.py \
  --out-dir submissions/v5b_idlock_runtime_strict \
  --post-adapter-dir lora_adapters/v5b_mac_diverse \
  --post-context strict-live
```

## Result Checkpoint

2026-05-16:

- Phase 2.6 archive landed on `main`.
- Phase 2.7 runtime implementation landed as a clean library/runner split:
  - `src/idlock_engine.py`
  - `scripts/run_idlock_path.py`
  - `tests/test_idlock_engine.py`
- Unit tests: `116/116` pass.
- Runtime paired benchmark result: `submissions/v5b_idlock_runtime` = **75.6%** (`2589/3425`), below the `87.0%` gate.

The failure is localized, not diffuse: the runtime pre-lock shabad ID commits `kZhIA8P6xWI` and `kZhIA8P6xWI_cold33` to shabad `4377` instead of `1821`, collapsing those cases to `5.0%` and `10.6%`.

Diagnostic blind-ID probes did not rescue the runtime:

| Variant | Overall | Interpretation |
|---|---:|---|
| `chunk_vote`, 30s | **75.6%** | Best checked runtime mode, but wrong on 2/12 starts. |
| `tfidf`, 30s | 65.5% | Fixes `kZhIA8P6xWI`, breaks `IZOsmkdmmcg` and `zOtIpxMT9hU`. |
| `topk:3`, 30s | 58.7% | Worse. |

Additional checkpoint: re-running the documented v3.2 command under the current environment produces `submissions/v3_2_repro_current` = **73.5%**, not the archived **86.5%** from `submissions/v3_2_pathA_no_title`.

That changes the scientific interpretation of Phase 2.6. The `87.1%` ID-lock proxy was valid as a submission-level diagnostic, but it depended on a historical submission artifact that is not currently reproducible as a runtime. Phase 2.7 therefore does **not** justify Phase 3 training scale-up.

## Decision

Do not promote `v5b_idlock_runtime`. Do not start Phase 3. The next phase is Phase 2.8: recover/pin the ASR baseline and attack the timing/alignment boundary directly.

Recommended Phase 2.8 scope:

1. **Reproducibility recovery.** Pin faster-whisper, CTranslate2, model snapshot, VAD flags, and ASR transcript cache checksums. A current runtime must reproduce either the archived `86.5%` v3.2 result or explain the drift with transcript diffs.
2. **Timing/alignment prototypes.** Test `vad_filter=False`, `word_timestamps=True`, and hybrid timing (`surt`/v5b text with faster-whisper timestamps). Past notes repeatedly identify long VAD null gaps and chunk-boundary mismatch as the remaining lift.
3. **Forced-alignment path.** If timestamp prototypes do not recover the 86-90% range, move to full-shabad forced alignment rather than more per-chunk classification.
4. **OOS remains required.** Paired benchmark recovery alone is not production promotion.
