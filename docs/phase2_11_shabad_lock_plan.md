# Phase 2.11 — shabad-lock robustness gate

**Status:** active checkpoint after the assisted OOS diagnostic.

Phase 2.9 produced the best paired-benchmark runtime so far:
`phase2_9_loop_align` scored **91.2%** with 12/12 paired-benchmark locks.
That is real progress, but it is not enough evidence for the 95%+ goal. The
OOS bridge has now exposed the next bottleneck: shabad identification.

## New evidence

`make eval-oos-loop-align-assisted` ran against the machine-assisted OOS labels.
This is **silver/diagnostic only** (`MACHINE_ASSISTED_V1_NOT_GOLD`), not a
promotion-grade OOS score.

Result:

| Case | GT shabad | Predicted shabad | Lock status | Frame accuracy |
|---|---:|---:|---|---:|
| case_001 | 2333 | 1821 | wrong | 0.0% |
| case_002 | 906 | 906 | correct | 60.0% |
| case_003 | 2600 | 2600 | correct | 75.0% |
| case_004 | 4892 | 906 | wrong | 16.1% |
| case_005 | 3297 | 906 | wrong | 0.0% |
| **overall** | — | — | **2/5 locks** | **29.5%** |

Interpretation:

- The OOS corpus cache includes the correct shabads (`2333`, `906`, `2600`,
  `4892`, `3297`), so this is **not** a missing-corpus failure.
- cases 004/005 had `top=0.0` / `runner_up=0.0` at the 30s lock point. The
  engine currently commits anyway, which effectively picks an arbitrary
  lowest-ID candidate. That is a real architecture bug: zero-evidence windows
  must not commit a shabad.
- case 001 had a non-zero tie and chose benchmark shabad `1821` over OOS shabad
  `2333`. That is candidate-set sensitivity: shared lyric hooks and repeated
  lines can confuse the current `chunk_vote` lock.
- A cached lock-variant audit found that simple alternatives can recover much
  of the OOS failure (`45s`/`60s` lookback plus `tfidf` with `topk:3` fallback
  gets 5/5 on the assisted OOS cases), but the same naive hybrid regresses the
  paired benchmark when the candidate set includes both benchmark and OOS
  shabads. So we should not blindly switch one knob.

## Expert decision

Do **not** start Phase 3 broad 300h training yet.

The current blocker is not M4 Pro compute and not lack of training hours. The
current blocker is the validity of the runtime architecture under a broader
candidate set. Scaling ASR training before fixing shabad lock can improve
oracle-shabad alignment while still producing wrong live captions.

Phase 2.11 therefore becomes the gate before any Phase 3 scale-up:

1. Build a repeatable shabad-lock audit over cached ASR transcripts.
2. Add a no-zero-evidence lock rule.
3. Prototype a confidence-gated / delayed lock policy.
4. Score the policy on both paired benchmark and OOS assisted labels.
5. Only then decide whether gold OOS correction or Phase 3 compute is the next
   bottleneck.

## Target architecture

The target is still clean and production-shaped:

```text
audio -> pre-lock ASR evidence -> shabad-lock policy -> locked-shabad aligner
```

The lock policy must be generic:

- no benchmark route tables;
- no hardcoded case IDs;
- no GT shabad hints in blind mode;
- no commit when all lock evidence is zero;
- bounded delayed lock so a live UI can keep tentative captions while waiting
  for real lyric evidence.

The first useful policy is likely:

1. evaluate lock evidence at 30s, 45s, 60s, 90s;
2. commit only when at least one scorer has non-zero evidence and sufficient
   margin;
3. prefer TF-IDF when it has a clear score/margin;
4. fall back to top-k line evidence for cases where TF-IDF has no signal but
   repeated lyric chunks are present;
5. keep pre-lock captions tentative and let the retro-buffered post engine
   rewrite after commit.

## Stop rules

- Do not count machine-assisted OOS as gold.
- Do not use GT shabad IDs to make the blind runtime look good.
- Do not tune a paired-only rule that breaks OOS, or an OOS-only rule that
  breaks paired.
- Do not start 50h/3-seed training until shabad-lock behavior is stable under a
  broader candidate set.

## Next action

Implement `scripts/audit_shabad_lock.py` and Makefile targets for:

```bash
make audit-oos-lock-assisted
make audit-paired-lock
```

These targets should produce markdown diagnostics under `diagnostics/` from
cached ASR transcripts. After the report is repeatable, prototype the smallest
runtime change: "no zero-evidence commit" plus delayed lock windows.

## Prototype checkpoint — delayed zero-evidence guard

Implemented an opt-in Layer 2 runtime guard in `src/idlock_engine.py`:

```text
--lock-lookbacks 30,45,60,90 --min-lock-score 1
```

Historical behavior is unchanged unless `--lock-lookbacks` is passed. With the
guard enabled, the pre-lock engine retries later windows if the top blind-ID
score is below `min_lock_score`; this prevents the 30s all-zero lock observed
in OOS cases 004/005.

Assisted-OOS diagnostic:

| Runtime | Correct locks | Overall assisted-OOS score |
|---|---:|---:|
| `phase2_9_loop_align` default | 2/5 | 29.5% |
| delayed zero-evidence guard | 3/5 | 40.5% |

Case-level change:

- case_005 improves from wrong lock (`906`) / 0.0% to correct lock (`3297`) /
  53.3%.
- case_004 delays from 30s to 45s but still locks incorrectly (`1341` instead
  of `4892`), so zero-evidence guarding is necessary but not sufficient.
- case_001 still fails due shared-hook ambiguity (`2333 -> 1821`) even with
  non-zero evidence.

Oracle-shabad diagnostic against the same machine-assisted labels:

| Mode | Overall assisted-OOS score |
|---|---:|
| Correct shabad forced, v5b + loop-align | 51.0% |

This means the assisted labels are useful for debugging lock failure modes, but
they are not promotion-grade. Even with GT shabad forced, the score remains low,
so some combination of rough machine labels, coarse boundaries, and OOS timing
alignment is still unresolved. The final promotion gate must remain gold OOS
(`HUMAN_CORRECTED_V1`) or a higher-quality public full-shabad labeled source.

## Revised next action

1. Keep the delayed zero-evidence guard as an opt-in runtime primitive.
2. Do **not** promote it; it is a partial fix.
3. Prototype candidate-set / scorer consensus next:
   - never commit on all-zero evidence;
   - delay lock until at least one lyric-bearing window exists;
   - compare `chunk_vote`, `tfidf`, and `topk:3` without choosing a rule that
     wins OOS but destroys paired benchmark behavior;
   - evaluate on both paired and assisted OOS before any codepath becomes
     default.
4. In parallel, continue replacing machine-assisted OOS labels with gold labels
   or locate a public full-shabad timestamped dataset with stable BaniDB IDs.
