# Phase 2.10 — automated silver OOS from online labels

## Why this phase exists

The five-case OOS v1 pack is the right **gold** promotion gate, but it requires
ear-based line-boundary correction. That is high-quality work, but it should not
block every next engineering step.

There is now a better bridge: public timestamped kirtan data exists online.
`surindersinghssj/gurbani-kirtan-dataset-v2` is a Hugging Face dataset with
line-level audio segments, canonical Gurmukhi text matched against STTM,
`start_time` / `end_time`, `match_score`, `video_id`, and train/validation/test
splits by video ID.

So the plan changes:

- **Silver OOS**: automated, broad, cheap, useful for ASR / label-matching
  diagnostics. Use online timestamped datasets and high-confidence filters.
- **Gold OOS**: small, human-corrected, promotion-grade. Keep `oos_v1` as the
  final gate before Phase 3 / production promotion.

This answers the practical concern: we should not make the user label five
cases before we can learn anything else. We can continue with silver evaluation
now, and reserve human curation for final trust.

## Dataset candidate

Primary candidate:

- HF repo: `surindersinghssj/gurbani-kirtan-dataset-v2`
- License: `cc-by-4.0`
- Shape: line-level audio segments, canonical Gurmukhi, OCR text, translation,
  `match_score`, `start_time`, `end_time`, `duration`, `slide_index`,
  `video_id`, `shabad_title`, `channel`, `kirtan_style`, `segment_type`
- Published splits: train / validation / test by video ID
- Immediate use: line-segment ASR and canonical-text match evaluation

Important limitation: the exposed schema has `shabad_title` and canonical text,
but not an obvious BaniDB `shabad_id` / `verse_id` pair. That makes it
excellent for **segment-level ASR/canonical-text diagnostics**, but not a
drop-in replacement for benchmark-shaped full-shabad runtime scoring until we
add a mapping layer.

## What silver can answer

Silver can answer:

- Does `surt-small-v3` transcribe unseen line segments better than the base
  `faster_whisper` path?
- Does `v5b_mac_diverse` improve or degrade canonical text matching on broad
  held-out videos?
- Are failures concentrated by duration, title, channel, match score, or
  repeated-line patterns?
- Is the adapter worth scaling, or is alignment still the main bottleneck?

Silver cannot answer by itself:

- Whether the runtime ID-lock path tracks a full shabad correctly.
- Whether a 91.2% paired-benchmark architecture is production-ready.
- Whether exact live line boundaries are correct across a whole kirtan clip.

Those remain gold-OOS questions.

## Recommended implementation

### 2.10.A — access + schema probe

Try to access the dataset through `datasets` / `huggingface_hub`.

Current machine note: the public page is visible in browser/search, but the
local HF API client returned `401 Unauthorized` for the repo on 2026-05-16.
Before coding against it, resolve one of:

- repo became private/gated;
- HF API auth is required even though the viewer is public;
- local hub client/version is stale;
- repo name changed.

Success criterion: print the first 3 rows of the `test` split with columns and
audio payload shape.

### 2.10.B — silver segment-eval command

Add a script:

```bash
python scripts/eval_silver_kirtan_v2.py \
  --split test \
  --limit 100 \
  --min-match-score 95 \
  --backend huggingface_whisper \
  --model surindersinghssj/surt-small-v3 \
  --out submissions/silver_kirtan_v2_surt_small.json
```

For each row:

1. write row audio to a temporary 16 kHz WAV;
2. transcribe with the selected ASR backend;
3. normalize predicted text and target `gurmukhi_text`;
4. report exact normalized match, token-sort / WRatio, and length bucket;
5. archive per-row JSONL plus summary metrics.

No `jiwer` dependency required for the first pass; use existing
`src.matcher.normalize` + `rapidfuzz`.

### 2.10.C — compare model variants

Run the same silver set for:

1. `surt-small-v3` base HF model;
2. `surt-small-v3 + lora_adapters/v5b_mac_diverse`;
3. optionally `faster_whisper medium` as a non-Gurbani baseline.

Decision:

- If the adapter loses on silver segments too, pause adapter scaling.
- If the adapter wins on silver but runtime OOS still fails, focus on alignment
  / ID-lock rather than ASR training.
- If both silver and gold OOS win, Phase 3 scale-up is justified.

### 2.10.D — optional full-video reconstruction

If the dataset exposes stable `video_id`, `start_time`, `end_time`, and enough
canonical ID metadata, build benchmark-shaped cases automatically:

```text
eval_data/silver_oos_v2/
  audio/<video_id>_16k.wav
  test/<video_id>.json
```

This would let `make eval-oos-loop-align` score broader full-song cases. It
requires mapping canonical text rows into a consistent shabad corpus:

- find or derive BaniDB/STTM line IDs;
- group by `(video_id, shabad_title)`;
- drop mixed-shabad / low-score / OCR-noisy videos;
- hold out the resulting video IDs from all future training pulls.

This is worth doing after 2.10.B proves the dataset is accessible and useful.

## Relation to manual OOS v1

Do not delete the manual OOS plan. Reclassify it:

- `oos_v1`: **gold OOS**, five hand-corrected full-shabad cases, required before
  promotion / Phase 3 claim.
- `silver_kirtan_v2`: **silver OOS**, automated broad segment diagnostics,
  useful immediately and cheap to repeat.

Manual curation is not a sign the system is weak; it is how we keep the final
number honest. Silver lets us continue engineering without waiting on that
human-quality label pass.

## Next action

Resolve HF access for `surindersinghssj/gurbani-kirtan-dataset-v2`, then build
2.10.B. If HF access remains blocked, use the existing 300h canonical dataset to
construct a **held-out video split** from never-trained shards as a fallback
silver set.
