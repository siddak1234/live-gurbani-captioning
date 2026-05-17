# OOS v1 labeling checkpoint

This is the current hand-curation checkpoint for the five-shabad OOS gate. The
paired benchmark score for `phase2_9_loop_align` is promising (`91.2%`), but it
is still only four shabads. These five cases are the first independent
generalization check before Phase 3 scale-up or any promotion claim.

For the machine triage of the current draft labels, see
[`docs/oos_v1_machine_audit.md`](oos_v1_machine_audit.md).

## Expert decision

Use these cases as **held-out evaluation**, not training data.

Why:

- The current benchmark is too small to estimate production robustness.
- Training on the OOS slate would remove the only near-term independent
  measurement of whether the 91.2% architecture generalizes.
- Training diversity is already handled by the HuggingFace pulls and the
  Phase 3/4 plan; the missing piece right now is variance in evaluation.

Decision rule:

- If OOS v1 passes with no catastrophic case, proceed to Phase 3 scale-up.
- If OOS v1 drops sharply, diagnose by case role before spending more training
  budget.

Parallel track: Phase 2.10 adds automated **silver** OOS from online
timestamped labels so engineering can continue without waiting on every gold
label. Silver is a diagnostic bridge, not a replacement for this gold gate.

## Current artifacts

| Artifact | Path | Status |
|---|---|---|
| Case manifest | `eval_data/oos_v1/cases.yaml` | committed |
| Local audio | `eval_data/oos_v1/audio/case_*_16k.wav` | present locally, gitignored |
| Draft labels | `eval_data/oos_v1/drafts/case_*.json` | committed, not ground truth |
| Review page | `eval_data/oos_v1/review/index.html` | generated locally, gitignored |
| Human-corrected GT | `eval_data/oos_v1/test/case_*.json` | **missing** |
| Validation gate | `make validate-oos-gt` | intentionally failing until GT exists |
| OOS score command | `make eval-oos-loop-align` | blocked on validation |

## Labeling queue

| Order | Case | Role | Shabad | Draft segments | Duration | Source |
|---:|---|---|---:|---:|---:|---|
| 1 | `case_001` | representative | 2333 | 7 | 180 s | Mera Baid Guru Govinda - Bhai Navdeep Singh Ji Manawan |
| 2 | `case_002` | representative | 906 | 8 | 160 s | Tu Thakur Tum Peh Ardas |
| 3 | `case_003` | representative | 2600 | 5 | 180 s | Jo Mange Thakur Apne Te - Bhai Sarabjit Singh Ji |
| 4 | `case_004` | acoustic / lexical stress | 4892 | 4 | 180 s | Awal Allah Noor Upaya - Bhai Jujhar Singh Ji |
| 5 | `case_005` | matcher stress | 3297 | 11 | 180 s | Koi Bolei Ram Ram - Delhi Wale |

Recommended order is representatives first, then stress cases. That gives a
quick read on whether failures are broad or concentrated in the harder slices.
The machine audit refines the review order slightly: start with `case_002`
because its expected opening line is absent from the draft, then review
`case_003` / `case_004` for large gaps and backward jumps.

## Hand-correction protocol

1. Run:

   ```bash
   make oos-review-pack
   ```

2. Open `eval_data/oos_v1/review/index.html`.

3. For each case, copy the matching draft JSON from `eval_data/oos_v1/drafts/`
   into `eval_data/oos_v1/test/`.

4. Listen to the full clipped audio once before editing.

5. For every draft segment:

   - adjust `start` / `end` to the actually sung line;
   - delete hallucinated or duplicated rows;
   - add missing repeated lines;
   - keep segment starts in ascending order;
   - verify `line_idx`, `verse_id`, and `banidb_gurmukhi` against the cached
     shabad corpus.

6. Set:

   ```json
   "curation_status": "HUMAN_CORRECTED_V1"
   ```

7. Run:

   ```bash
   make validate-oos-gt
   ```

8. Only after validation passes, run:

   ```bash
   make eval-oos-loop-align
   ```

## Pass / fail interpretation

The OOS pack is intentionally small, so do not overfit to the mean alone.

Use this hierarchy:

1. **Catastrophic-case check:** no single OOS case should collapse below the
   broad failure band. A one-case collapse means diagnose alignment / ID-lock
   behavior before scaling.
2. **Representative-case check:** the three representative cases should be
   stable. If these fail, the paired benchmark result is not robust.
3. **Stress-case check:** stress failures are useful diagnostics, not automatic
   architecture rejection, unless they reveal a general smoother or ASR issue.
4. **Mean score:** useful only after the per-case read. Five cases is not a
   production metric.

This is why `oos_v1` blocks Phase 3 but does not replace the larger Phase 5
`oos_v2` plan.
