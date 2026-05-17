# Phase 2.12 — silver learning without waiting on gold OOS

**Status:** active option.

The user asked whether we can keep learning instead of waiting for hand-corrected
OOS labels. Yes — with one important boundary:

- **Silver labels** can drive engineering and learning decisions.
- **Gold labels** are still required for promotion claims.

This lets the project keep moving without pretending the machine-assisted OOS
pack is final truth.

## What can be treated as labels now

| Source | Use now? | Role |
|---|---|---|
| Paired benchmark 4 shabads / 12 cases | yes | regression/dev benchmark only |
| `eval_data/oos_v1/assisted_test/` | yes | silver diagnostic labels; not gold |
| 300h canonical HF dataset | yes | training + silver held-out segment eval |
| `eval_data/oos_v1/test/` working files | no final claim | editable workspace until `HUMAN_CORRECTED_V1` |

Taking the assisted OOS labels as "correct enough" is acceptable for a **silver
learning loop**. It is not acceptable for saying we reached 95% production
accuracy.

## Expert decision

Do **not** jump straight to all-300h training as the next step.

Why:

- `v5b_mac_diverse` already proved the M4 Pro can train and that more data can
  change transcripts, but blind/live score regressed to 65.6%.
- `phase2_9_loop_align` reached 91.2% on the paired benchmark through runtime
  architecture, not through a larger adapter.
- Assisted OOS then failed mostly through shabad-lock/candidate selection.
- Oracle-shabad assisted OOS reached only 51.0%, so machine labels and timing
  are still too rough for promotion.

The best "learning" target is therefore not another broad ASR LoRA run. It is a
learned or tuned **shabad-lock policy** that uses the labeled paired cases,
assisted OOS cases, and cached ASR/corpus evidence to choose when and how to
commit a shabad.

## Phase 2.12.A — silver lock-policy tuner

Build a repeatable tuner that searches policy variants over cached ASR
transcripts:

- candidate lock windows: 30s, 45s, 60s, 90s;
- scorers: `chunk_vote`, `tfidf`, `topk:3`, `tfidf_then_topk3`;
- delayed-commit threshold: do not commit before the last window unless top
  score clears a minimum threshold;
- objective: macro average of paired lock accuracy and assisted-OOS lock
  accuracy, so the 12 paired cases do not drown out the 5 OOS cases.

Output:

```text
diagnostics/phase2_12_silver_lock_policy.md
```

This is a **learning report**, not a final model card.

Executed checkpoint:

```bash
make tune-shabad-lock-policy
```

Result: the best silver macro policy is `chunk_vote@45s|min=0`, with **67.5%**
macro lock accuracy:

- Paired benchmark locks: **9/12** (75.0%)
- Assisted-OOS locks: **3/5** (60.0%)
- Best assisted-OOS-only policy: `tfidf_then_topk3@45s|min=0`, **5/5** OOS but
  only **3/12** paired

This is a useful negative result. It says we can keep learning from silver
labels, but a hand-tuned scorer/window switch is not enough and should not be
promoted. The failure is not M4 Pro compute underuse and not lack of 300h
training data; it is candidate-sensitive shabad retrieval / lock evidence.

## Phase 2.12.B — decision rules

After tuning:

- If one generic policy improves assisted OOS without hurting paired locks, make
  it an opt-in runtime experiment and run full frame scoring.
- If every policy trades paired accuracy for OOS accuracy, do not promote a
  hand-tuned rule. Move to a real candidate-retrieval model or require shabad
  search narrowing from metadata/user interaction.
- If silver labels are too noisy to separate policies, go back to gold OOS
  correction before spending larger compute.

Current decision: the second branch applies. OOS-optimized scoring wins assisted
OOS but fails paired regression; paired-safe scoring still fails 2/5 assisted
OOS locks. The next learning step should be candidate retrieval / lock-evidence
modeling, not a larger ASR adapter run and not accepting the silver OOS result
as proof.

## What this means for the 300h dataset

The 300h labeled set remains central:

- keep using it for ASR adapter training and silver ASR diagnostics;
- use held-out shards to catch acoustic regressions;
- do not use it as a substitute for full-shabad runtime OOS unless we can map
  segment labels back to stable BaniDB shabad/verse IDs.

The learning loop continues now, but the promotion gate stays honest.
