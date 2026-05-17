# Phase 2.13 — candidate retrieval / lock-evidence fusion

**Status:** completed diagnostic; keep as opt-in runtime evidence.

Phase 2.12 answered the "can we keep learning without hand validation?"
question: yes, but only as a silver loop. The silver lock-policy tuner also
showed that a simple scorer/window switch is not enough:

- best paired-safe rule: `chunk_vote@45s|min=0`, **11/12** paired and **3/5**
  assisted OOS;
- best assisted-OOS-only rule: `tfidf_then_topk3@45s|min=0`, **5/5** assisted
  OOS but only **7/12** paired.

That is a structural signal. The lock problem is not "choose TF-IDF instead of
chunk vote" or "wait 15 more seconds." It is candidate retrieval under ambiguous
early evidence.

## Expert decision

Do **not** jump straight to broad all-300h training.

Reason: the current failure mode is still wrong shabad commitment in one hard
paired case, not inability to run larger training. More ASR training can improve
acoustic evidence, but it will not automatically teach the runtime which
canonical shabad to lock when the candidate set contains superficially similar
hooks. The 48 GB M4 Pro is being used properly for adapter training; compute is
healthy. The next training step should therefore be a controlled Phase 3
warm-start slice with explicit silver/paired/OOS gates, not a blind all-data run.

## What Phase 2.13 tests

Build an evidence table for every candidate shabad in the current corpus cache:

- windows: 30s, 45s, 60s, 90s;
- features:
  - chunk-vote score;
  - TF-IDF document score;
  - top-3 line score;
- target: paired benchmark shabad ID or assisted-OOS silver shabad ID.

Then search sparse non-negative fusion policies, for example:

```text
0.5 * chunk_vote_45 + 1.0 * tfidf_45 + 1.0 * topk3_90
```

The objective remains a macro average of paired lock accuracy and assisted-OOS
lock accuracy, so the 12 paired cases do not drown out the five OOS cases.

## Output

```text
diagnostics/phase2_13_lock_evidence_fusion.md
```

This is still **silver**. A high score is evidence for the next runtime
experiment, not a production validation result.

## Executed checkpoint

Command:

```bash
make tune-lock-evidence-fusion
```

Important audit correction: the first version of the tuner evaluated cold-start
cases from `t=0` even though runtime starts its lock window at the case UEM start.
That made the diagnostic too pessimistic for cold cases. The tuner now carries
`uem.start` through the feature table; runtime behavior already did this.

Corrected best fusion:

```text
fusion:tfidf_45+0.5*chunk_vote_90
```

Silver lock result:

- Paired lock accuracy: **11/12** (91.7%)
- Assisted-OOS lock accuracy: **5/5** (100.0%)
- Silver macro lock objective: **95.8%**

This is a material improvement over Phase 2.12's single-policy trade-off:
`chunk_vote@45s` keeps paired safe but misses OOS, while `tfidf_then_topk3@45s`
gets OOS but regresses paired. Multi-feature evidence fusion gets both sides
near the guardrail.

Full-frame diagnostic with the opt-in fusion aggregate:

```bash
make eval-oos-lock-fusion-assisted
make eval-paired-lock-fusion
```

Results:

- Assisted OOS silver: **59.9%** frame accuracy, **5/5** locks correct
- Paired benchmark: **84.1%** frame accuracy

The paired drop is now concentrated in one case: full-start `zOtIpxMT9hU` locks
to shabad `4892` instead of `3712`. The cold-start variants now lock correctly.
Fusion is therefore useful as a diagnostic and opt-in runtime policy, but not a
promotion candidate.

Phase 2.14 tested longer windows and tail-window evidence to see whether later
evidence can repair the full-start `zOtIpxMT9hU` false candidate. Longer windows
and tail features reproduce the same 11/12 paired and 5/5 assisted-OOS lock
result. Heavier tail weighting can force 12/12 paired, but it collapses
assisted-OOS behavior; reject that as overfit.

## Decision rules

- If sparse evidence fusion beats Phase 2.12 while preserving paired regression,
  implement the best fusion as an opt-in runtime lock policy and run full frame
  scoring.
- If fusion only wins by overfitting assisted OOS and hurts paired, reject it
  and move toward explicit candidate retrieval: title/caption priors,
  metadata/user narrowing, or a learned retriever trained on broader online
  labeled data.
- If the GT candidate is often not near the top under any feature, expand or
  rebuild the corpus/retrieval layer before tuning alignment.

Current decision: fusion beats Phase 2.12 on assisted OOS and fixes most paired
locks under the corrected UEM-aware diagnostic. Keep it opt-in and continue
with two parallel tracks:

1. controlled Phase 3 warm-start ASR training on fresh shards, to test whether
   better acoustic evidence reduces lock ambiguity without wrecking OOS;
2. continued lock/retrieval work for high-confidence false candidates like
   `3712` vs `4892`.

## Architecture boundary

This phase stays in Layer 2. It may select *which shabad* to lock, but it must
not introduce benchmark-specific route tables or case-specific overrides. The
runtime engine remains:

```text
audio -> ASR evidence -> generic lock policy -> locked-shabad aligner
```
