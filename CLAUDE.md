# CLAUDE.md

Orientation for Claude Code sessions in this repo. Read the paired benchmark for deeper detail; this file is an index, not a duplicate.

## Two machines, two roles

If you're a Claude session on the **training machine** (a dedicated Apple Silicon Mac that runs fine-tunes), your job is narrow: `make start`. That runs Python check → `pip install -r requirements-mac.txt` → auto-pulls `surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical` from HuggingFace → LoRA-fine-tunes `surt-small-v3`. No ffmpeg, no paired benchmark repo, no smoke test. Adapter lands in `lora_adapters/surt_mac_v1/`; ship it back to the dev machine.

If you're a Claude session on the **dev machine** (where code review, benchmark scoring, OOS evaluation, and iOS export happen), your job is wider. Run `make start-dev` for the kitchen-sink chain (doctor-dev + install + fetch-audio + corpus + data + smoke + train), or pick targets individually: `make eval`, `make eval-oos`, `make ios-export`. Requires `ffmpeg` (`brew install ffmpeg`) and the paired benchmark repo cloned at `../live-gurbani-captioning-benchmark-v1/`. iOS deployment goes through Argmax's `whisperkittools` (HF → Core ML w/ ANE quantization) + the [WhisperKit](https://github.com/argmaxinc/WhisperKit) Swift package; see [`docs/ios_deployment.md`](docs/ios_deployment.md) and the `ios/` directory.

Run `make help` on either machine to see every target and its one-line purpose.

## Project goal

Build a **real, robust, scalable kirtan captioning engine** that works at user level on arbitrary recordings — not just the 4 shabads in the paired benchmark. The benchmark is a *measurement tool*, not the deliverable. Solutions that overfit to its 12 cases are leaderboard exercises, not deployable systems.

Two parallel deliverables:
1. **Honest benchmark scores** (we report them, *and* mark which are overfit vs generalizable).
2. **A generalizable engine** (single-model pipeline trained on broad data; performs well on novel recordings the benchmark hasn't seen).

This is not a stop-and-ship project. The solution is far from over — every committed submission is a step, not a finish line. Future sessions should keep pushing toward (2) even when (1)'s numbers look good.

## What this repo is

An engine that consumes kirtan audio and emits line-level predictions of which shabad and line is being sung at each moment. The paired benchmark provides a 12-case test set for measurement. We do **not** ship the benchmark, the ground truth, or the scorer — we produce submissions, but the actual production deliverable is the engine itself.

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

### Leaderboard (with generalization annotations)

**Read this column carefully.** Benchmark score ≠ deployable accuracy. Submissions marked *overfit* tune to specific shabads in the 12-case test set and would degrade on novel recordings.

| Submission | Mode | Score | Generalizes? |
|---|---|---|---|
| `v0_empty` | — | 26.0% | n/a (baseline floor) |
| `v3_2_pathA_no_title` | blind + live | **86.5%** | **Yes — current honest production candidate.** Generic fw-medium ASR; matcher/smoother tuning is mostly architecture-level, not shabad-specific. |
| `v4_mlx_large_v3` | blind + live | 83.8% | Yes — alternative ASR backend (Apple GPU). |
| `x4_pathA_surt` | blind + live | 74.0% | **Yes — best designed for generalization.** Uses surindersinghssj/surt-small-v3 (Whisper-small fine-tuned on 660h of kirtan). Lower benchmark score is real, but reflects honest accuracy on out-of-set kirtan. |
| `x7_surt_only` | blind + live | 68.6% | Yes — surt with longer blind buffer; didn't help (kept for negative-result record). |
| `x8_pb_finetuned` | blind + offline | 72.9% (Path B) | **Yes — proof of production training path.** w2v-bert-punjabi + LoRA adapter from 50-step fine-tune on 30 real kirtan clips. +2.6 over Path B baseline; +6 to +14 per-shabad on 3 of 4 shabads. Validates the end-to-end training pipeline; tiny scale, far from saturated. |
| `x5_ensemble` | blind + live | 91.2% | **No — benchmark-overfit.** Route table `{1341 → surt}` chosen from test-set scores. |
| **`x6_ensemble`** | **blind + live** | **92.8%** | **No — most overfit.** 3-way routing `{1341 → surt, 1821 → mlx, else → v3.2}` is empirically picked per test-shabad. Would degrade on shabads not in {4377, 1821, 1341, 3712}. |

**What benchmark-overfit means concretely:** if a Sewadar plays a new kirtan recording of, say, shabad 5621, our X5/X6 ensembles route it to whichever engine the *default* path picks — they don't have a special rule for that shabad. The hard-coded route table is calibrated for the 4 test shabads. Real-world accuracy on out-of-set shabads is closer to x4_surt's 74% than x6's 92.8%.

What didn't work (don't re-try without new info): score-threshold > 0 (nulls correct-but-low chunks), top-1/top-2 margin gate (correlated with confidence but not causally), TF-IDF (exact-token match breaks under unidecode schwa-drop), shorter blind-ID lookback than 30s (drops shabad-ID reliability), longer blind-ID lookback for surt (eats UEM in cold variants).

## What we're really building toward

The reference live captioning system at [bani.karanbirsingh.com](https://bani.karanbirsingh.com) is the **closest thing to the production target**. Its approach (per the writeup at [karanbirsingh.com/gurbani-captioning](http://www.karanbirsingh.com/gurbani-captioning)):
- One 118M Punjabi conformer model trained on ~300h of YouTube kirtan
- Forced-aligned to canonical SGGS lines via phonetic matching
- State machine for shabad confirmation and line tracking
- Runs against arbitrary Sikhnet Radio streams — true generalization, not per-shabad lookup

Surinder Singh has open-sourced the dataset and models:
- **[`gurbani-kirtan-yt-captions-300h-canonical`](https://huggingface.co/datasets/surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical)** — 300h labeled kirtan audio. Apache 2.0.
- **[`surt-small-v3`](https://huggingface.co/surindersinghssj/surt-small-v3)** — Whisper-small fine-tuned on 660h of Gurbani.
- **[`indicconformer-pa-v3-kirtan`](https://huggingface.co/surindersinghssj/indicconformer-pa-v3-kirtan)** — IndicConformer fine-tuned for kirtan (NeMo).

**The path to a real engine starts with these.** Continued ensembling on our existing components will keep raising the benchmark score, but is dead-end work for the production goal.

## Real roadmap (the actual remaining work)

Numbered roughly by order, not by difficulty:

1. **Establish honest evaluation hygiene.** Before claiming any new score, run on a held-out audio recording not used during tuning. Even just one new Sikhnet-Radio recording with a known-but-not-in-benchmark shabad lets us catch overfitting early.
2. **Ship surt-small-v3 with better integration as the v0 production engine.** The standalone 74% is held back by ASR-vs-matcher chunk-granularity mismatch, not by the model itself. Investigate: word-level timestamps from the HF pipeline, custom decoding that respects line boundaries, or use surt's text + faster-whisper's timestamps as a hybrid.
3. **Fine-tune surt-small-v3 on the 300h dataset (Mac-first).** Pipeline validated end-to-end on real data (see `submissions/x8_pb_finetuned/notes.md` for the smoke run that moved Path B HMM +2.6 points). Real training runs locally on M-series Macs via PyTorch + MPS — see [`docs/training_on_mac.md`](docs/training_on_mac.md). Tools: `scripts/pull_dataset.py kirtan` (pulls from `surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical` with benchmark holdout), `scripts/finetune_path_b.py --config configs/training/surt_lora_mac.yaml` (auto-detects Whisper vs CTC from `--model-id`; target_modules default to Whisper's `q_proj/k_proj/v_proj/out_proj` when `surt-small-v3` is the base). Cloud fallback (Colab/RunPod) documented in [`docs/cloud_training.md`](docs/cloud_training.md) for the day Mac wall-clock becomes the bottleneck.
4. **Move to forced alignment over the full shabad** (Path B done right) instead of per-chunk classification. Aligns the whole shabad text to the audio as one continuous problem; naturally handles line transitions including rapid ones. Architecture sketch is in `src/path_b/`.
5. **Replace the route table with a learned dispatcher** — small classifier picking the engine based on audio features (tempo, vocal/instrumental ratio, etc.), not on shabad ID lookup. Makes ensembling honest by design.
6. **Build the live deployment surface**: streaming audio in, captions out, Sewadar UI with confirm/reset buttons (matches the reference system's UX from karanbirsingh.com).

Steps 1, 2, 3 are the highest-leverage real-world improvements. Steps 4-6 are the polish.

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

### Fine-tuning (LoRA on surt-small-v3) — Mac-first plan

The ASR plateau (Path A 86.5%, Path B 70.3%) is structural and only escapes via domain-adapted acoustic modeling. The production target is `surt-small-v3` (Whisper-small architecture, kirtan-fine-tuned by surindersinghssj). Path B (w2v-bert + HMM) remains as research scaffolding but is not the deployment target.

**Pipeline (built):**
- `src/path_b/dataset.py` — manifest-driven (audio, text) data loader; has separate paths for CTC and Whisper Seq2Seq
- `scripts/pull_dataset.py` — unified data acquisition with subcommands (`kirtan`, `sehaj`, `sehajpath`); enforces benchmark holdout
- `scripts/build_training_dataset.py` — YouTube-source fallback when HF isn't enough
- `scripts/finetune_path_b.py` — LoRA fine-tune; auto-detects Whisper vs CTC from `--model-id` (or override with `--model-type`). Loads pinned hyperparameters from `configs/training/surt_lora_mac.yaml`.
- `scripts/build_smoke_manifest.py` — tiny pipeline-validation manifest (NOT for real training — contaminates test set)
- `scripts/run_path_a.py --backend huggingface_whisper --adapter-dir <path>` — inference with a saved LoRA adapter against the canonical Path A pipeline
- Training cache: `lora_adapters/<name>/` (gitignored)
- Training data: `training_data/<manifest>/` (gitignored — keep audio off the public repo)

**Canonical end-to-end workflow (Mac):**

```bash
# 1) Pull a labeled slice (benchmark shabads filtered out automatically):
python scripts/pull_dataset.py kirtan \
  --out-dir training_data/kirtan_v1 \
  --num-samples 200 --min-score 0.8

# 2) LoRA fine-tune surt-small-v3 (Whisper Seq2Seq path auto-detected):
python scripts/finetune_path_b.py \
  --config configs/training/surt_lora_mac.yaml \
  --manifest training_data/kirtan_v1/manifest.json \
  --output-dir lora_adapters/surt_mac_v1

# 3) Evaluate against the paired benchmark:
python scripts/run_path_a.py \
  --backend huggingface_whisper \
  --model surindersinghssj/surt-small-v3 \
  --adapter-dir lora_adapters/surt_mac_v1 \
  --blend "token_sort_ratio:0.5,WRatio:0.5" --threshold 0 \
  --stay-bias 6 --blind --blind-aggregate chunk_vote \
  --blind-lookback 30 --live --tentative-emit \
  --out-dir submissions/v5_surt_mac_v1

python ../live-gurbani-captioning-benchmark-v1/eval.py \
  --pred submissions/v5_surt_mac_v1 \
  --gt   ../live-gurbani-captioning-benchmark-v1/test/

# 4) Out-of-set evaluation (honest accuracy — required before claiming a number):
python scripts/eval_oos.py \
  --data-dir eval_data/oos_v1 \
  --pred-dir submissions/oos_v1_surt_mac_v1 \
  --engine-config configs/inference/v3_2.yaml
```

**Smoke-test (validated on prior w2v-bert path):** `PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/finetune_path_b.py --manifest <smoke.json> --output-dir /tmp/lora_smoke --max-steps 20 --batch-size 1`. ~1.4 steps/sec on Apple Silicon. For Whisper Seq2Seq specifically: cross-entropy loss is MPS-native (no CPU fallback like CTC has), so throughput is higher — measure on first real run.

**Data sources (sorted by current relevance):**

1. **`surindersinghssj` on HuggingFace** — primary source.
   - [`surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical`](https://huggingface.co/datasets/surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical) — 300h kirtan audio + line text, 208k samples, Apache 2.0
   - [`surindersinghssj/gurbani-sehajpath-yt-captions-canonical`](https://huggingface.co/datasets/surindersinghssj/gurbani-sehajpath-yt-captions-canonical) — ~160h spoken paath
   - [`surindersinghssj/surt-small-v3`](https://huggingface.co/surindersinghssj/surt-small-v3) — our base model, Whisper-small fine-tuned on 660h Gurbani
   - Phase X4 (committed) confirmed: pure surt substitution hits 74.0% overall but wins +10-17 points on kchMJPK9Axs vs Path A v3.2. Oracle ensemble of {v3.2, X4} per case ≈ 90.3%.
2. **YouTube kirtan with shabad-level metadata** — `scripts/build_training_dataset.py` covers this fallback path.
3. **AI4Bharat IndicVoices Punjabi subset** — general Punjabi speech regularization (stub subcommand in `pull_dataset.py`, lands when needed).
4. **SikhiToTheMax / Khalis Foundation archives** — potential broadcast-quality source for future iterations.

**Compute reality:**
- M4 Pro 48GB (or any M-series ≥16GB unified): fits surt-small-v3 LoRA comfortably. ~4-8 hours per epoch for a 50h dataset. **Primary path.**
- Cloud GPU (Colab A100, RunPod) covered by `requirements-cloud.txt` + [`docs/cloud_training.md`](docs/cloud_training.md). Use only if Mac wall-clock becomes the bottleneck.

**Anti-overfitting hygiene:**
- Hold out by *shabad identity*, not by recording
- Strict separation: benchmark's 4 shabads (4377, 1821, 1341, 3712) NEVER appear in train (enforced by `pull_dataset.py` and `configs/datasets.yaml`)
- 80/10/10 train/val/test with shabad-level boundaries
- Early stopping on val loss
- LoRA's parameter efficiency (~1% trainable) is its own regularization

**Expected lift if executed well:** lyrics-alignment and Quran-recitation literature consistently report +10-15 points from in-domain fine-tuning. Target: 90-95% on the paired benchmark, 85-90% on OOS eval.

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
