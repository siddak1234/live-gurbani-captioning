# scripts/legacy/

Files in this directory are **kept for historical reproducibility** but are not part of the current pipeline. They are documented here so new contributors don't mistake them for canonical entry points.

If you are setting up the repo for the first time, **you can ignore everything in this directory**.

## Why each lives here

### `separate_vocals.py` — Phase X1 vocal source separation (Demucs)

Tested whether running Demucs to isolate vocals before transcription helped. Result: asymmetric — hurt the canonical `faster_whisper` backend by 20.5 points, helped the `mlx_whisper` backend by 10.5 points. Not a universal win, and the artifacts Demucs introduces (clipping, phase smearing) break VAD assumptions in `faster_whisper`'s preprocessor.

Reproducing the Phase X1 submissions (`x1_pathA_vocals_fw`, `x1_pathA_vocals_mlx`) requires this script. The output `audio_vocals/` directory is still gitignored.

### `probe_ctc.py` — Phase B0 CTC viability probe (MMS-1B)

Earliest Path B exploration: dumps per-frame CTC posteriors from `facebook/mms-1b-all` and inspects blank-dominance. Confirmed the diagnosis in `submissions/pb1_hmm/notes.md` (slow kirtan → CTC emits blank ~99% of frames → frame-level discrimination weak).

MMS-1B was replaced by `kdcyberdude/w2v-bert-punjabi` in Phase X3 and is no longer in the active pipeline. The `mms_cache/` directory is still gitignored.

### `run_path_b.py` — Phase B0 simple-CTC baseline (no HMM)

First end-to-end Path B run before the loop-aware HMM landed. Just emits per-frame argmax line predictions with no transition prior. Superseded by `scripts/run_path_b_hmm.py` which is the canonical Path B runner.

## Current canonical entry points (live under `scripts/`)

| Script | What it does |
|---|---|
| `run_path_a.py` | Path A v3.2 benchmark runner (canonical, 86.5%) |
| `run_path_b_hmm.py` | Path B benchmark runner with loop-aware HMM |
| `eval_oos.py` | Out-of-set evaluation harness |
| `pull_dataset.py` | Unified data acquisition (kirtan/sehaj/sehajpath/...) |
| `pull_kirtan_data.py` | Back-compat shim → `pull_dataset.py kirtan` |
| `finetune_path_b.py` | LoRA fine-tune (Whisper Seq2Seq OR CTC, auto-detected) |
| `build_corpus.py` | BaniDB → corpus_cache/ |
| `build_ios_corpus.py` | corpus_cache/ → ios/.../shabads.json |
| `build_training_dataset.py` | YouTube → labeled training manifest |
| `build_smoke_manifest.py` | 4-snippet smoke manifest (do not score against benchmark) |
| `export_coreml.py` | LoRA + base → Core ML .mlpackage (whisperkittools) |
| `benchmark_ane_latency.py` | Measures inference latency on macOS Core ML |
| `fetch_audio.py` | yt-dlp + ffmpeg → audio/*.wav |
| `ensemble_submissions.py` | Per-shabad oracle ensembling tool |
| `run_benchmark.py` | Stage 0 empty-submission baseline (~26% floor) |
