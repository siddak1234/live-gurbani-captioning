# Phase 2.6 — alignment and ID-lock diagnostic

Phase 2.6 exists because `v5b_mac_diverse` made training loss move strongly but made the blind/live benchmark score drop from `74.0%` to `65.6%`. A drop is not a model candidate; it is a diagnostic. The goal here is to identify which boundary failed before spending more compute.

## Checkpoint audit

| Area | Evidence | Read |
|---|---|---|
| Training loop | `v5_mac_baseline` and `v5b_mac_diverse` both trained on MPS and emitted `run_card.json`. | Valid. Training itself works. |
| Data hygiene | `v5b_mac_diverse`: 2,544 clips, 4.936h, 20 videos, 195 shabad tokens, 0 benchmark video/content leaks. | Valid. The negative result is not obvious benchmark contamination. |
| Adapter activity | `v5b` prediction JSONs differ from `x4` in 10/12 cases and from `v5` in 12/12. | Valid. The adapter is actually used at inference. |
| Blind/live score | `v5b_mac_diverse`: `65.6%`, down 8.4 pt from `x4`/`v5`. | Not promotable. Do not scale Phase 3 from this. |
| Prior art | v3.2 is `86.5%` honest; x5/x6 reach `91.2%`/`92.8%` through benchmark-aware route tables; x7 longer surt blind buffer regressed. | Do not tune route tables or blind-lookback length on the paired benchmark. |

## Diagnostic results

All oracle diagnostics used:

```bash
HF_WINDOW_SECONDS=10 python3 scripts/run_path_a.py \
  --backend huggingface_whisper \
  --model surindersinghssj/surt-small-v3 \
  --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 \
  --stay-bias 6 --live --blind-lookback 0 \
  --out-dir <submission_dir>
```

Adapter variants added `--adapter-dir <adapter>`. These scores are **not deployment scores** because the benchmark shabad ID is provided and the blind-ID delay is removed. They are valid as an isolation test for ASR-plus-matcher alignment once the shabad is known.

| Submission | Mode | Score | Interpretation |
|---|---|---:|---|
| `x4_pathA_surt` | blind + live | 74.0% | Honest surt baseline. |
| `v5_mac_baseline` | blind + live | 74.0% | Neutral adapter. |
| `v5b_mac_diverse` | blind + live | 65.6% | Regression from wrong shabad IDs and cold-window instability. |
| `x4_pathA_surt_oracle_live0` | oracle shabad + live0 | 85.2% | Base surt alignment upper bound under known shabad. |
| `v5_mac_baseline_oracle_live0` | oracle shabad + live0 | 85.2% | 200-clip adapter does not change oracle alignment. |
| `v5b_mac_diverse_oracle_live0` | oracle shabad + live0 | **87.4%** | Diverse adapter improves alignment by +2.2 pt over x4/v5 when routing is fixed. |
| `v5b_twopass_v32_idlock` | v3.2 pre-lock + v5b post-lock proxy | **87.1%** | The drop is recoverable with a conservative ID-lock architecture. |

## Scientific conclusion

`v5b_mac_diverse` is not globally worse. It improves oracle-shabad alignment from `85.2%` to `87.4%`, and a simple two-pass ID-lock proxy scores `87.1%`, slightly above the frozen v3.2 honest baseline of `86.5%`.

The failure is integration robustness:

- The adapted transcript perturbs blind shabad ID.
- Wrong shabad ID is catastrophic because the matcher then searches the wrong line set.
- Longer blind lookback is already a recorded negative (`x7_surt_only`), so the next fix is not "wait longer."
- Benchmark-aware route tables are disallowed for production, even though x5/x6 show that method selection can raise the paired score.

## Validity boundaries

What these diagnostics validate:

- The adapter can help line alignment once the shabad is known.
- The paired-benchmark regression is primarily a routing/integration problem, not a failed training loop.
- A generic two-pass integration is worth building as the next smallest real architecture step.

What they do **not** validate:

- They do not prove deployable accuracy. Oracle-shabad results use GT shabad IDs.
- They do not satisfy OOS requirements. OOS v1 is still required before any promotion claim.
- They do not justify Phase 3 scale-up yet. Scaling a fragile integration is the wrong experiment.

## Follow-up result: Phase 2.7

Phase 2.7 built the actual runtime two-pass ID-lock path and tested it on the paired benchmark:

1. **ID-lock engine.** `src/idlock_engine.py` uses the v3.2/faster-whisper path for the first 30s shabad-ID buffer and tentative captions.
2. **Post-lock adapter alignment.** After the shabad is locked, it runs `v5b_mac_diverse` against the locked shabad only.
3. **No route tables.** The switch is by time/state, not by benchmark shabad ID.

Result:

- `v5b_idlock_runtime`: **75.6%**, below the `87.0%` gate.
- Failure mode: current runtime blind-ID commits two `kZhIA8P6xWI` starts to shabad `4377` instead of `1821`.
- Reproducibility finding: the documented v3.2 command now scores **73.5%**, not the archived **86.5%**. The Phase 2.6 proxy remains useful as a diagnostic, but it was not a currently reproducible runtime baseline.

Decision: do not train larger. Move to Phase 2.8: ASR reproducibility recovery plus timestamp/alignment prototypes.

## Path to 95%

The 95% target is still not solved. The two-pass proxy recovers the v5b drop and slightly beats v3.2, but it is not enough. The remaining lift likely requires one of:

- word-level timestamps or custom decoding for `surt-small-v3`;
- a hybrid using v3.2/faster-whisper timing and surt/v5b canonical text;
- full-shabad forced alignment instead of per-chunk line classification;
- a stronger acoustic backbone (`surt-medium` or IndicConformer) if oracle alignment plateaus.

The next move is therefore architecture and alignment, not another larger LoRA run.
