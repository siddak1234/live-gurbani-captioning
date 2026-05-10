# CLAUDE.md

Orientation for Claude Code sessions in this repo. Read the paired benchmark for deeper detail; this file is an index, not a duplicate.

## What this repo is

An engine that consumes kirtan audio and emits line-level predictions to be scored against the paired benchmark. We do **not** ship the benchmark, the ground truth, or the scorer — we produce submissions for it.

## Paired benchmark

`~/Desktop/coding_projects/live-gurbani-captioning-benchmark-v1/`

That repo is the source of truth for task definition, scoring rules, and submission format. Read its `README.md`, `eval.py`, and `test/*.json` files directly when you need detail — do not guess from this CLAUDE.md.

## Submission format (minimum to know)

One JSON per GT case under `<benchmark>/test/`, same filename stem, placed in a single output directory:

```json
{
  "video_id": "IZOsmkdmmcg",
  "segments": [
    { "start": 30.0, "end": 47.0, "line_idx": 1 },
    ...
  ]
}
```

- `start < end`, both in seconds relative to the start of the audio.
- `line_idx` is **0-indexed within the predicted shabad** (not the GT shabad).
- Segments may overlap; later segments overwrite earlier ones per-frame.
- Unsegmented regions = `null` prediction (accepted in GT gaps, wrong inside GT segment interiors).
- Predictions outside UEM are ignored.

A pred segment may optionally also carry `shabad_id`, `verse_id`, or `banidb_gurmukhi`. The scorer's `_resolve_pred_label` ([eval.py:91](../live-gurbani-captioning-benchmark-v1/eval.py#L91)) resolves preds against GT in this order, first hit wins:
  1. `(shabad_id, line_idx)` — both sides agree on numbering
  2. `verse_id` — canonical BaniDB pangti id
  3. `banidb_gurmukhi` — verbatim spaced unicode pangti text

Anything that doesn't resolve gets `NO_MATCH` and is scored wrong. Carry `verse_id` and/or `banidb_gurmukhi` when your engine's `line_idx` numbering might disagree with GT's.

## How to score

```bash
python ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/<run_name>/ \
  --gt   ../live-gurbani-captioning-benchmark-v1/test/
```

Default `--collar 1`. Primary metric is **frame accuracy** at 1s resolution. Empty submissions floor at ~26%; perfect copy of GT scores 100%.

## Task variants

Four modes, same submission schema:

|             | **Blind** (identify shabad from audio) | **Oracle** (given `shabad_id` upfront) |
|-------------|----------------------------------------|----------------------------------------|
| **Offline** | full audio available                   | full audio + GT shabad                 |
| **Live**    | causal: predictions at `t` only depend on audio ≤ `t` | causal + GT shabad                     |

Live causality is honor-system — the scorer can't tell. The output JSON looks identical in all four modes.

## Core design principle

**Snap to canonical.** The engine never displays raw ASR text. Output is always the canonical Gurmukhi line looked up by `(shabad_id, line_idx)` from BaniDB. Misspelled Gurmukhi in a religious context is unacceptable; the integer-id submission format makes that constraint structural rather than aspirational.

## Project stages

- **Stage 0** ✅ — empty submissions, ~26% baseline (plumbing + scoring loop). Currently committed: `submissions/v0_empty/`.
- **Stage 1** ⏳ next — audio download (`yt-dlp` + `ffmpeg` to 16kHz mono WAV under `audio/`) + BaniDB corpus cache for the four shabads.
- **Stage 2** — first real engine: Whisper ASR + fuzzy match (rapidfuzz) against the cached corpus. Run in offline + oracle mode first.
- **Stage 3** — drop oracle: blind shabad ID from audio.
- **Stage 4** — drop offline: live causal streaming.

## Repo layout (planned)

```
src/                  engine code (matchers, decoders, snap-to-canonical logic)
scripts/              entry points (run_benchmark.py, fetch_audio, build_corpus)
submissions/          one folder per experiment: v<N>_<name>/ + notes.md
audio/                downloaded WAVs (gitignored)
corpus_cache/         BaniDB lookups (gitignored)
requirements.txt      Python deps
```

`audio/` and `corpus_cache/` are gitignored — they're derived artifacts.

## External references

- **BaniDB** — https://api.banidb.com (e.g. `api.banidb.com/v2/shabads/{shabad_id}`). Source of canonical Gurmukhi lines and `verse_id`s.
- **SikhiToTheMax** — uses the same `shabad_id` and `line_idx` as BaniDB; either service can resolve a prediction to displayable text.

## Conventions for this repo

- **Never overwrite a submission folder.** Each experiment lives at `submissions/v<N>_<short_name>/` and ships with a `notes.md` describing the engine config (model, params, mode, score). The score is part of the experiment record — don't mutate a folder after scoring it.
- **Always run the scorer and visualizer on every new run.** After producing a submission, run `eval.py` and also generate `tiles.html` with the benchmark's `visualize.py`:

  ```bash
  python ../live-gurbani-captioning-benchmark-v1/visualize.py \
    --pred submissions/<run_name>/ \
    --gt   ../live-gurbani-captioning-benchmark-v1/test/ \
    --audio-dir audio/ \
    --out submissions/<run_name>/tiles.html
  ```

  The HTML strip view catches systematic failures that an aggregate accuracy number hides.
- **Keep the engine deterministic where possible** so re-runs are reproducible against committed submission JSONs.
