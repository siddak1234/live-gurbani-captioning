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

- **Stage 0** ✅ — empty-submission baseline. 26.0%. `submissions/v0_empty/`.
- **Stage 1** ✅ — audio download + BaniDB corpus cache. `scripts/fetch_audio.py`, `scripts/build_corpus.py`.
- **Stage 2** ✅ — Path A engine (Whisper + rapidfuzz + smoother). Iterated v1.0 → v1.5; **88.2%** oracle+offline with `0.5*token_sort_ratio + 0.5*WRatio` blend + stay-bias=6.
- **Stage 3** ✅ — blind shabad ID via per-chunk voting. **88.2%** blind+offline (zero drop from oracle, 12/12 IDs). `submissions/v2_pathA_blind/`.
- **Stage 4** ✅ — live causal mode + tentative emission during ID buffer. **86.0%** blind+live (strictly causal). `submissions/v3_1_pathA_live_tentative/`.
- **Stage 5** ⏳ — close the gap to 95%+. Path A's matcher/smoother are tapped out around 86-88%. Cheap probes (model size, mlx-whisper, ratio sweeps, TF-IDF) all fail to push past. Path A is now **frozen** at v3.2 = 86.5% live blind; Path B (CTC phoneme scoring + loop-aware HMM, in `src/path_b/`) is the principled next move with a realistic 95%+ ceiling.

### Current leaderboard

| Submission | Mode | Score |
|---|---|---|
| `v0_empty` | — | 26.0% |
| `v1_pathA_oracle` | oracle + offline | 68.4% |
| `v1_4_pathA_blend` | oracle + offline | 84.8% |
| `v1_5_pathA_staybias` | oracle + offline | **88.2%** |
| `v2_pathA_blind` | blind + offline | **88.2%** |
| `v3_pathA_live` | blind + live (strict causal) | 78.4% |
| `v3_1_pathA_live_tentative` | blind + live (causal + tentative) | 86.0% |
| `v3_2_pathA_no_title` | + skip line 0 (Path A canonical) | **86.5%** |
| `v4_mlx_medium` | mlx backend, medium model | 70.6% (regression) |
| `v4_mlx_large_v3` | mlx backend, large-v3, lookback=45 | 83.8% |

What didn't work (don't re-try without new info): score-threshold > 0 (nulls correct-but-low chunks), top-1/top-2 margin gate (correlated with confidence but not causally), TF-IDF (exact-token match breaks under unidecode schwa-drop).

## Repo layout

The repo holds **two engine implementations side-by-side** (Path A and Path B). They share infrastructure (audio fetcher, corpus cache, scoring) but their engines live in separate folders so changes to one never affect the other.

```
src/                        Path A engine (FROZEN at v3.2 = 86.5% live blind)
  asr.py                    Dual-backend ASR wrapper:
                              - faster_whisper (default): CPU, Silero VAD, produces v3.2
                              - mlx_whisper: Apple Silicon GPU, faster but different chunking
  matcher.py                rapidfuzz scoring + TfidfScorer + score_chunk()
  smoother.py               smooth() and smooth_with_stay_bias() — causal
  shabad_id.py              identify_shabad() (chunk_vote) + per_chunk_global_match
  path_b/                   Path B engine (in development)
    __init__.py             Scaffold; CTC scorer + loop-aware HMM go here
scripts/
  run_benchmark.py          Stage 0 empty submitter
  fetch_audio.py            yt-dlp + ffmpeg → audio/*.wav
  build_corpus.py           BaniDB API → corpus_cache/<id>.json
  run_path_a.py             Path A runner (--backend flag selects fw or mlx)
  run_path_b.py             (future) Path B runner
submissions/v<N>_<name>/    one folder per experiment + notes.md + tiles.html
audio/                      16kHz mono WAVs (gitignored)
corpus_cache/               BaniDB shabad lines (gitignored)
asr_cache/                  cached transcripts, key includes backend (gitignored)
```

### Reproducing Path A v3.2 (86.5% blind + live)

```bash
python scripts/run_path_a.py \
  --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 \
  --stay-bias 6 --blind --blind-aggregate chunk_vote \
  --blind-lookback 30 --live --tentative-emit \
  --out-dir submissions/v3_2_pathA_no_title
```

Default `--backend faster_whisper` is the canonical Path A backend. `--backend mlx_whisper` is available but produces different chunk granularity that requires retuning (see v4_mlx_* notes).

### Path B status

Empty scaffold. Will use the same audio/, corpus_cache/, and scoring as Path A.

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
