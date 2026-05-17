# OOS v1 machine audit

This note answers the checkpoint question: "Can we parse through the drafts
instead of using the model again?"

Yes, but only as a **triage pass**. Parsing can prove that draft labels are
internally consistent with the cached shabad corpus and can flag suspicious
timing / ordering patterns. It cannot certify the actual sung line boundaries
without listening to the audio. That is why `make validate-oos-gt` still
requires `curation_status: HUMAN_CORRECTED_V1` before scoring.

## What the parser can check

- Every draft `line_idx` exists in `corpus_cache/<shabad_id>.json`.
- Every draft `verse_id` and `banidb_gurmukhi` matches the cached corpus for
  that line.
- Segment starts are ordered and segment durations are positive.
- Large silent / unlabeled gaps are visible.
- Backward jumps in line order are visible.
- The manifest's expected opening line is present somewhere in the draft.

## What the parser cannot check

- Whether the singer actually sang that line at that timestamp.
- Whether repeated rahao / refrain lines were skipped or over-inserted.
- Whether alap, simran, or non-shabad filler should be scored as null.
- Whether a line boundary should move by several seconds.

Those are audio-grounded labeling decisions. Using the model draft as truth
would make the OOS score circular.

## Draft audit summary

Generated from the committed drafts under `eval_data/oos_v1/drafts/` and the
cached corpora under `corpus_cache/`.

| Case | Corpus match | Coverage | Gaps >10s | Draft sequence | Opening line | Machine concern |
|---|---:|---:|---:|---|---|---|
| `case_001` | 7/7 | 72% | 1 | `[1, 2, 4, 3, 4, 3, 4]` | yes | large intro / unlabeled gap |
| `case_002` | 8/8 | 81% | 1 | `[1, 2, 1, 5, 1, 6, 7, 6]` | no | opening line absent, repeated backward jumps |
| `case_003` | 5/5 | 67% | 3 | `[6, 5, 6, 1, 2]` | yes | large gaps, strong backward jumps |
| `case_004` | 4/4 | 71% | 2 | `[2, 6, 4, 1]` | yes | large gaps, strong backward jumps |
| `case_005` | 11/11 | 76% | 2 | `[3, 4, 3, 4, 1, 2, 1, 2, 1, 5, 6]` | yes | refrain-like jumps; likely needs careful repeated-line review |

## Interpretation

Good news:

- The drafts are corpus-consistent. There are no obvious wrong `verse_id` /
  `banidb_gurmukhi` mismatches.
- Every case has enough draft structure to make hand curation much faster than
  labeling from scratch.

Risk areas:

- `case_002` is the highest-priority review item because the expected opening
  line is absent from the draft.
- `case_003` and `case_004` have strong backward jumps and large gaps; they may
  have intro / repeated refrain / alignment issues.
- `case_005` is intentionally a matcher-stress case, so repeated line jumps are
  expected, but they still need confirmation by ear.

## Recommended next action

Proceed with manual correction, but focus attention in this order:

1. `case_002` — resolve the absent opening line and repeated jumps.
2. `case_003` / `case_004` — verify whether the large gaps should be null,
   missing sung lines, or boundary errors.
3. `case_005` — verify repeated refrain structure.
4. `case_001` — likely easiest; confirm intro gap and refrain repeats.

After correction:

```bash
make validate-oos-gt
make eval-oos-loop-align
```

Do not move to Phase 3 until this gate is scored and interpreted.

## Workspace-prep checkpoint

`make prepare-oos-review` now seeds editable working files under
`eval_data/oos_v1/test/` from the machine drafts while preserving the gate:
each file is marked `curation_status: NEEDS_HUMAN_CORRECTION`, so
`make validate-oos-gt` still fails until review is complete.

After seeding the five working files, validation no longer reports missing GT
files. It reports:

- all five cases still need `curation_status: HUMAN_CORRECTED_V1`;
- `case_005` has `segments[10].end` beyond `total_duration` (`181.0s` vs a
  `180.0s` clip), so the final repeated-line boundary must be corrected during
  review.

That is the expected state: the project has moved from "missing GT files" to
"editable working GT exists, but gold correction is not yet complete."
