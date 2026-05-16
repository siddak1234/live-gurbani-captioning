# Phase 2.8 — ASR reproducibility recovery + timing/alignment pivot

**Status:** complete as a diagnostic phase — ASR timestamp probes and
post-lock smoother probes are archived. Phase 2.9 is the next step.

Phase 2.8 exists because Phase 2.7 exposed a more basic problem than the v5b adapter: the archived Path A v3.2 result is no longer reproducible as a runtime.

Current runtime snapshot: [`docs/phase2_8_runtime_snapshot.md`](phase2_8_runtime_snapshot.md).

Key facts:

| Artifact | Score | Meaning |
|---|---:|---|
| `submissions/v3_2_pathA_no_title` | 86.5% | Historical frozen submission artifact. |
| `submissions/v3_2_repro_current` | 73.5% | Same documented command, current environment/cache. |
| `submissions/v5b_twopass_v32_idlock` | 87.1% | Phase 2.6 proxy using historical v3.2 pre-lock segments. |
| `submissions/v5b_idlock_runtime` | 75.6% | Real Phase 2.7 runtime path; fails on current blind-ID. |

That gap means we cannot honestly claim the old `86.5%` baseline as a currently reproducible production path. Before we spend more compute on LoRA scale, we need to pin the ASR runtime and move the next architecture bet toward timing/alignment, where the old notes already point.

## Hypotheses

1. **Reproducibility drift hypothesis.** The v3.2 drop is caused by an unpinned ASR variable: faster-whisper/CTranslate2 version, model snapshot, VAD behavior, or transcript cache contents.
2. **Timing hypothesis.** The remaining path to 95% is not another per-chunk scorer; it is better timestamps and full-shabad alignment. Past notes repeatedly mention VAD null gaps, chunk-boundary mismatch, word timestamps, and forced alignment.

## Workstream A — Recover the baseline

Goal: explain or eliminate the `86.5% -> 73.5%` drift.

Tasks:

1. Record current runtime versions:
   - Python
   - faster-whisper
   - CTranslate2
   - tokenizers / transformers (if applicable)
   - model path / snapshot if exposed
2. Preserve ASR transcript checksums for the four benchmark WAVs.
3. Diff current faster-whisper transcripts against the historical v3.2 submission behavior:
   - start/end chunk boundaries;
   - first 30s text used for blind ID;
   - cases where ID flips (`kZhIA8P6xWI`, `kZhIA8P6xWI_cold33`).
4. Add a small notes/run-card convention for future submissions: ASR cache keys + hashes must be recorded next to the score.

Success:

- Either reproduce archived v3.2 at `>= 86.0%`, or document the exact transcript/runtime cause of the drift.

## Workstream B — Timestamp prototypes

Goal: test the lowest-cost timing fixes before rebuilding Path B.

Candidates:

1. `faster-whisper` with `vad_filter=False`.
2. `faster-whisper` with `word_timestamps=True`.
3. `faster-whisper` with both `vad_filter=False` and `word_timestamps=True`.
4. Hybrid: `surt-small-v3` / `v5b` text for recognition, faster-whisper timestamps for segmentation.

Evaluation:

- Run paired benchmark only as a smoke gate.
- Archive each prototype under `submissions/phase2_8_*`.
- Do not tune per-shabad route tables.

Success:

- One timestamp/alignment prototype scores `>= 87.0%` paired benchmark without a shabad route table, or yields a clear reason to move to full-shabad forced alignment.

### Initial probe results

2026-05-16:

| Prototype | Score | Result |
|---|---:|---|
| `v3_2_repro_current` | 73.5% | Current default (`vad_filter=False`, no word timestamps); reproduces the drift, not archived v3.2. |
| `phase2_8_fw_word` | 72.0% | Word timestamps fix blind ID (12/12 locks) but hurt full-run line tracking/timing. |
| `phase2_8_fw_vad` | 25.4% | VAD filtering deletes too much sung kirtan; dead path. |
| `phase2_8_idlock_preword` | **86.6%** | Best current runtime: word timestamps for pre-lock ID, v5b for post-lock alignment. Misses gate by 0.4 pts; OOS still owed. |
| `phase2_8_idlock_preword_viterbi` | 77.2% | Generic line-distance Viterbi smoother improves `IZOsmkdmmcg` but collapses `kZhIA8P6xWI` / `kchMJPK9Axs`; over-regularizes refrain/loop structure. |
| `phase2_8_idlock_preword_viterbi_null45` | 77.1% | Null-state Viterbi suppresses some filler but removes useful weak evidence elsewhere. |

Shorter pre-word ID-lock windows were worse: 15s = `79.7%`, 20s = `79.8%`, both because the open `kZhIA8P6xWI` case mis-locks. Keep 30s as the current conservative runtime choice.

Interpretation: word timestamps are useful for shabad ID, but not as the final
caption timing layer. A generic post-lock Viterbi smoother is also not enough:
it is too blunt for shabads with refrain loops and non-local returns. The next
lift is a real full-shabad alignment layer with explicit loop/refrain handling,
especially for short cold-start cases such as `zOtIpxMT9hU_cold66`.

## Workstream C — Forced-alignment decision

Workstream B recovered a near-baseline runtime (`86.6%`) but not a promotable
one, and the post-lock smoother probes regressed. Stop trying to squeeze
per-chunk classification. The next architecture should align the entire shabad
text to the audio as a sequence problem:

- known/committed shabad text;
- monotonic line progression with explicit loop/refrain handling;
- acoustic token/CTC or word-timestamp evidence;
- no per-shabad benchmark routing.

This is the principled path toward the 95% target. It matches the old notes: x5/x6 reached 91-93% only by route tables, while honest single-engine paths plateaued below 90%.

The next execution plan is [`phase2_9_plan.md`](phase2_9_plan.md).

## Stop Conditions

- Do not start Phase 3 LoRA scale-up until baseline reproducibility is explained and a timing/alignment prototype has a positive signal.
- Do not promote any paired-benchmark result without OOS.
- Do not count x5/x6-style route tables toward production.
