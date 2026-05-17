# Phase 2.13 — candidate retrieval / lock-evidence fusion

**Status:** active next step after Phase 2.12.

Phase 2.12 answered the "can we keep learning without hand validation?"
question: yes, but only as a silver loop. The silver lock-policy tuner also
showed that a simple scorer/window switch is not enough:

- best paired-safe rule: `chunk_vote@45s|min=0`, **9/12** paired and **3/5**
  assisted OOS;
- best assisted-OOS-only rule: `tfidf_then_topk3@45s|min=0`, **5/5** assisted
  OOS but only **3/12** paired.

That is a structural signal. The lock problem is not "choose TF-IDF instead of
chunk vote" or "wait 15 more seconds." It is candidate retrieval under ambiguous
early evidence.

## Expert decision

Do **not** start broad Phase 3 / all-300h training yet.

Reason: the current failure mode is wrong shabad commitment. More ASR training
can make transcripts different, but it will not automatically teach the runtime
which canonical shabad to lock when the candidate set contains superficially
similar hooks. The 48 GB M4 Pro is being used properly for adapter training;
compute is not the active bottleneck.

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

Best fusion:

```text
fusion:tfidf_60+0.5*chunk_vote_90
```

Silver lock result:

- Paired lock accuracy: **9/12** (75.0%)
- Assisted-OOS lock accuracy: **5/5** (100.0%)
- Silver macro lock objective: **87.5%**

This is a material improvement over Phase 2.12's best balanced policy
(`9/12` paired, `3/5` assisted OOS). It proves multi-feature evidence fusion is
more promising than a single scorer/window switch.

Full-frame diagnostic with the opt-in fusion aggregate:

```bash
make eval-oos-lock-fusion-assisted
make eval-paired-lock-fusion
```

Results:

- Assisted OOS silver: **59.9%** frame accuracy, **5/5** locks correct
- Paired benchmark: **79.7%** frame accuracy

The paired drop is concentrated: `zOtIpxMT9hU` and `zOtIpxMT9hU_cold33` lock to
shabad `4892` instead of `3712`; `zOtIpxMT9hU_cold66` locks correctly and scores
86.9%. Fusion is therefore useful as a diagnostic, but not a promotion
candidate.

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

Current decision: fusion beats Phase 2.12 on assisted OOS and is now available
as an opt-in experimental aggregate, but it hurts paired frame accuracy too much
to promote. Next targeted problem: disambiguate high-confidence false candidates
like `3712` vs `4892` without losing the OOS gains.

## Architecture boundary

This phase stays in Layer 2. It may select *which shabad* to lock, but it must
not introduce benchmark-specific route tables or case-specific overrides. The
runtime engine remains:

```text
audio -> ASR evidence -> generic lock policy -> locked-shabad aligner
```
