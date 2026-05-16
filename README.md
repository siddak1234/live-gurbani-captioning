# live-gurbani-captioning

Real-time engine that transcribes Gurbani kirtan and snaps each line to its canonical SGGS text (shabad_id + line_idx + verse_id + Gurmukhi). Built to run on-device on iPhone Neural Engine via WhisperKit, trained on Apple Silicon Macs.

The repo holds two parallel ASR pipelines (Path A: Whisper + fuzzy matcher + smoother; Path B: CTC + loop-aware HMM), unified by a single `src/engine.py` library. Current best on the [paired benchmark](../live-gurbani-captioning-benchmark-v1/): **Path A v3.2 at 86.5%** (blind + live, faster-whisper-medium). Best honest out-of-set candidate: x4 with `surt-small-v3` (74%), held back by ASR-matcher chunk-granularity mismatch.

## End-to-end pipeline

```
[ pull HF data ] → [ fine-tune surt on M4 Pro ] → [ Core ML export ] → [ iOS app ]
   scripts/         scripts/finetune_path_b.py    scripts/             ios/
   pull_dataset.py  configs/training/             export_coreml.py     Package.swift
                    surt_lora_mac.yaml            configs/export/      (WhisperKit + 
                                                  coreml_ane.yaml      Swift matcher
                                                                       + state machine)
```

Two evaluation gates run alongside: `scripts/run_path_a.py` (paired benchmark) and `scripts/eval_oos.py` (held-out shabads — the honest accuracy number).

For deeper context, read in this order:
1. **[CLAUDE.md](CLAUDE.md)** — project memory, task framing, submission history
2. **[`docs/architecture.md`](docs/architecture.md)** — three-layer decoupling (engine library / runners / iOS frontend)
3. **[`docs/training_on_mac.md`](docs/training_on_mac.md)** — fine-tune recipe for M-series
4. **[`docs/ios_deployment.md`](docs/ios_deployment.md)** — end-to-end iOS pipeline
5. **[`ios/README.md`](ios/README.md)** — Swift package build instructions

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Training | PyTorch + transformers + PEFT + MPS | Mac-first; same code runs on CUDA for cloud fallback |
| ASR base model | [`surindersinghssj/surt-small-v3`](https://huggingface.co/surindersinghssj/surt-small-v3) | Whisper-small, fine-tuned on 660 h of Gurbani — smallest viable kirtan ASR |
| Data sources | [`surindersinghssj` HF datasets](https://huggingface.co/surindersinghssj) | 525 h labeled kirtan + sehaj already public, Apache 2.0 |
| Matcher | rapidfuzz (Python) / pure-Swift port (iOS) | Token-sort + WRatio blend, no native deps |
| Export | [`whisperkittools`](https://github.com/argmaxinc/whisperkittools) | HF → Core ML with ANE-targeted 4-bit palettization |
| iOS runtime | [`WhisperKit`](https://github.com/argmaxinc/WhisperKit) | Streaming-aware Core ML inference on the Apple Neural Engine |
| Corpus | BaniDB API → embedded JSON | Canonical SGGS line text, shipped offline in the iOS bundle |

## Quickstart — training machine

A fresh M-series Mac, zero config, one command:

```bash
git clone <this repo>
cd live-gurbani-captioning
make start
```

`make start` does exactly three things: checks Python ≥ 3.10, `pip install -r requirements-mac.txt`, then pulls the open-source training data from HuggingFace (`surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical` — 200 samples by default) and runs the full LoRA fine-tune of `surt-small-v3`. No ffmpeg required, no benchmark repo required, no smoke step. Idempotent — interrupt and re-run safely.

The trained LoRA adapter lands in `lora_adapters/surt_mac_v1/`. Hand that directory back to the dev machine for benchmark scoring and Core ML export.

## Quickstart — dev machine

Adds benchmark scoring + iOS deployment. Requires `ffmpeg` and the paired benchmark repo cloned alongside this one:

```bash
brew install ffmpeg
cd ..
git clone <paired benchmark repo> live-gurbani-captioning-benchmark-v1
cd live-gurbani-captioning
make start-dev          # doctor + install + fetch audio + corpus + data + smoke + train
make eval               # score the adapter against the paired benchmark
make eval-oos           # honest accuracy on held-out shabads (curate eval_data/oos_v1 first)
make ios-export         # merge LoRA + export Core ML for the iOS bundle
make help               # list every target with a one-line description
```

Each target wraps a single underlying script — `cat Makefile` to see the actual commands and edit defaults. Cloud-GPU fallback uses `pip install -r requirements-cloud.txt`; see [`docs/cloud_training.md`](docs/cloud_training.md).

For hyperparameter rationale, training troubleshooting, and the full Mac recipe: [`docs/training_on_mac.md`](docs/training_on_mac.md).
For the iOS deployment pipeline (Core ML export, WhisperKit integration, on-device matcher): [`docs/ios_deployment.md`](docs/ios_deployment.md).

## Repo layout

```
src/
  engine.py           callable inference library (audio → segments)
  asr.py              ASR backends (faster-whisper, mlx-whisper, HF Whisper)
  matcher.py          rapidfuzz line-to-text scoring
  smoother.py         causal segment smoothing
  shabad_id.py        blind shabad identification
  path_b/             CTC encoder + tokenizer + loop-aware HMM
scripts/
  pull_dataset.py     unified data acquisition (kirtan/sehaj/sehajpath/...)
  finetune_path_b.py  LoRA fine-tune (auto CTC vs Whisper Seq2Seq)
  run_path_a.py       benchmark runner (thin I/O around engine.predict)
  run_path_b_hmm.py   Path B runner
  eval_oos.py         out-of-set evaluation harness
  export_coreml.py    HF → Core ML for iOS deployment
  build_ios_corpus.py BaniDB → ios/.../shabads.json
  legacy/             deprecated experiments — see scripts/legacy/README.md
configs/              training / inference / export / dataset configs
ios/                  Swift package: WhisperKit wrapper + matcher + state machine
docs/                 architecture, training, deployment guides
eval_data/            OOS evaluation set (audio gitignored; GT JSONs committed)
submissions/          experiment history (~22 runs, each with notes.md + tiles.html)
.claude/              project Claude Code config + safety hooks
```

## License

Apache 2.0 (see [`LICENSE`](LICENSE)). Submissions and notes are documentation, not redistributable benchmark data — the paired benchmark repo holds the test set + scorer.
