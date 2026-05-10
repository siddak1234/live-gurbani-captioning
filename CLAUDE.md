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

| Phase | Status | Result |
|---|---|---|
| B0-B3: MMS-1B + HMM | ✅ | 61.0% (below Path A 86.5%) |
| X1: Demucs vocal separation | ✅ | fw -20.5, mlx +10.5 — not a universal win |
| X3.0: swap to w2v-bert-punjabi | ✅ | 70.3% (+9.3 vs MMS, still below Path A) |
| B.1: LoRA fine-tune pipeline | ✅ | smoke-tested, ready for real data |
| B.2-B.6: collect kirtan data + fine-tune | ⏳ next | target 90-95% |

### Fine-tuning (LoRA on w2v-bert) — multi-week plan

The ASR plateau (Path A 86.5%, Path B 70.3%) is structural and only escapes via domain-adapted acoustic modeling.

**Pipeline (built):**
- `src/path_b/dataset.py` — manifest-driven (audio, text) data loader
- `scripts/build_training_dataset.py` — **the data acquisition tool**. Takes a CSV of `(youtube_id, shabad_id)` pairs, downloads each, runs Path A v3.2 in oracle mode to forced-align line → time, writes a training manifest. Refuses benchmark shabads by default.
- `scripts/finetune_path_b.py` — LoRA fine-tune of any HF CTC model (default `kdcyberdude/w2v-bert-punjabi`)
- `scripts/build_smoke_manifest.py` — generates a tiny pipeline-validation manifest (NOT for real training — contaminates test set)
- `scripts/run_path_b_hmm.py --adapter-dir <path>` — inference with a saved LoRA adapter
- Training cache: `lora_adapters/<name>/` (gitignored)
- Training data: `training_data/<manifest>/` (gitignored — keep audio off the public repo)

**End-to-end workflow once you have a CSV:**

```bash
# 1) Curate a CSV with non-benchmark shabads:
#    youtube_id,shabad_id,notes
#    abc123XYZ,5621,...

# 2) Build training manifest (downloads + auto-labels via Path A):
python scripts/build_training_dataset.py \
  --input-csv my_kirtan_sources.csv \
  --out-dir training_data/kirtan_v1 \
  --backend faster_whisper --model medium

# 3) Manually review/curate manifest.json (Path A labels are ~86% accurate; drop bad rows)

# 4) Fine-tune (locally for small data, cloud GPU for scale):
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/finetune_path_b.py \
  --manifest training_data/kirtan_v1/manifest.json \
  --output-dir lora_adapters/kirtan_v1 \
  --epochs 3 --batch-size 2

# 5) Evaluate fine-tuned adapter:
python scripts/run_path_b_hmm.py \
  --model-id kdcyberdude/w2v-bert-punjabi --target-lang "" \
  --adapter-dir lora_adapters/kirtan_v1 \
  --out-dir submissions/pb_kirtan_v1
python ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/pb_kirtan_v1 --gt ../live-gurbani-captioning-benchmark-v1/test/
```

**Smoke-test (validated):** `PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/finetune_path_b.py --manifest <smoke.json> --output-dir /tmp/lora_smoke --max-steps 20`. ~1.4 steps/sec on Apple Silicon (CTC loss falls back to CPU; PyTorch MPS doesn't yet implement aten::_ctc_loss). Trainable params: 3.3M of 617M (0.53% via LoRA r=16).

**Data sources (sorted by current relevance):**

1. **🎯 `surindersinghssj` on HuggingFace** — ALREADY EXISTS, ALREADY DONE.
   - [`surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical`](https://huggingface.co/datasets/surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical) — **300h of kirtan audio with line text labels**, 208k samples. Apache 2.0.
   - [`surindersinghssj/surt-small-v3`](https://huggingface.co/surindersinghssj/surt-small-v3) — Whisper-small fine-tuned on 660h of Gurbani audio, produces canonical Gurmukhi directly.
   - [`surindersinghssj/indicconformer-pa-v3-kirtan`](https://huggingface.co/surindersinghssj/indicconformer-pa-v3-kirtan) — IndicConformer fine-tune for kirtan (needs NeMo lib).
   - Phase X4 (committed) confirmed: pure surt substitution hits 74.0% overall but wins +10-17 points on kchMJPK9Axs vs Path A v3.2. Oracle ensemble of {v3.2, X4} per case ≈ 90.3%.
2. **YouTube kirtan with shabad-level metadata** — fallback if surindersinghssj isn't enough. `scripts/build_training_dataset.py` is built and ready.
3. **SikhiToTheMax / Khalis Foundation archives** — outreach to bod@khalisfoundation.org for ground-truth line-timed broadcast recordings.
4. **AI4Bharat IndicVoices Punjabi subset** — general Punjabi speech foundation.

**Compute reality:**
- Local Apple Silicon: fine for LoRA on tiny datasets (≤1h) and smoke tests
- Real fine-tune on 20-50h: needs cloud GPU (Colab Pro, RunPod). Single-GPU A100 should finish 20h dataset in a few hours.

**Anti-overfitting hygiene:**
- Hold out by *shabad identity*, not by recording
- Strict separation: benchmark's 4 shabads (4377, 1821, 1341, 3712) NEVER appear in train
- 80/10/10 train/val/test with shabad-level boundaries
- Early stopping on val loss
- LoRA's parameter efficiency (~0.5% trainable) is its own regularization

**Expected lift if executed well:** lyrics-alignment and Quran-recitation literature consistently report +10-15 points from in-domain fine-tuning. Target: blind+live 90-95%.

**Cloud training**: see [docs/cloud_training.md](docs/cloud_training.md) for cell-by-cell Google Colab + RunPod walkthrough. Mac is fine for smoke tests; real fine-tune (20+ hours of data) needs cloud GPU. A100 turns days of CPU into hours.

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
