# Sangat — Project & ML Overview

## What this project does

**Sangat** is an iOS app that listens to *kirtan* — live Sikh devotional singing — and, in real time, displays the exact line being sung from *Sri Guru Granth Sahib* (SGGS), the Sikh holy scripture. The engine identifies which of the ~5,500 hymns is being performed and tracks the current line within that hymn, then shows the official Gurmukhi text on screen so the congregation can read along.

### Glossary

| Term | Meaning |
|---|---|
| **Kirtan** | Sikh devotional singing of scripture — the audio our engine listens to. |
| **Shabad** | One hymn from SGGS. There are ~5,500 of them. |
| **SGGS / BaniDB** | The scripture / the public API ([api.banidb.com](https://api.banidb.com)) that returns the official Gurmukhi text for any `(shabad_id, line_idx)` pair. |
| **ASR** | Automatic Speech Recognition — turns audio into text. |
| **Whisper** | OpenAI's widely-used open-source ASR model (2022). |
| **Fine-tuning** | Continuing training of an already-trained model on more specific data. Cheaper than training from scratch. |
| **LoRA** | "Low-Rank Adaptation" — a fine-tuning technique that updates only ~1% of the model's weights. |
| **Benchmark / OOS** | A fixed test set (12 cases from 4 shabads) / an "out-of-set" test built from recordings the engine hasn't seen. |
| **WER / CER** | Word / Character Error Rate. Lower = better. |

---

## What we started with

Two pieces of public work. We use both; we built neither.

### Surinder Singh — model + datasets ([huggingface.co/surindersinghssj](https://huggingface.co/surindersinghssj))

Verified directly from the HuggingFace model card:

- **[`surt-small-v3`](https://huggingface.co/surindersinghssj/surt-small-v3)** — OpenAI Whisper-small architecture, fully fine-tuned by Surinder on **~660h of Gurbani audio** (~420h kirtan + ~220h spoken paath). **0.2B (200M) parameters. Apache 2.0.** This is our base model.
- **[`gurbani-kirtan-yt-captions-300h-canonical`](https://huggingface.co/datasets/surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical)** — 300h labeled kirtan corpus, 208k rows, YouTube captions auto-aligned to canonical SGGS via BaniDB. Our primary training data. (License not displayed on the public dataset page; CLAUDE.md notes Apache 2.0.)
- Surinder's own published benchmark numbers: **Kirtan WER 54.8%, CER 28.0%**. Sehaj WER 16.3%, CER 5.3%. That ~55% kirtan WER means the upstream model gets more than half of sung words wrong — so a runtime engine (matcher + state machine) around the model is essential, regardless of training quality.

### Karanbir Singh — reference system ([karanbirsingh.com/gurbani-captioning](http://www.karanbirsingh.com/gurbani-captioning))

A deployed live captioner at [bani.karanbirsingh.com](https://bani.karanbirsingh.com). **Code and model are not public.** From his writeup: AI4Bharat's Punjabi Conformer (a different model family than Whisper), 118M parameters, fine-tuned to INT8 ONNX. Pipeline is ASR → phonetic matcher → state machine. He reports **~65–70% on his live shabad-unaware task** — a more rigorous benchmark setup than ours, at the same approximate accuracy range as our OOS diagnostic.

**Net: we get a working 200M Whisper-Gurbani model, a 300h labeled dataset, and an architectural blueprint. We do not get a live engine, an iOS app, a way to collect user feedback, or a deployable system.**

---

## Our end goal

A real, deployable kirtan captioning engine that works on any recording, in any gurdwara, sung by any ragi, at user-acceptable accuracy and latency on a regular iPhone. The benchmark is a measurement tool, not the product. Success = real generalization on real-world audio + improving over time as real users surface real mistakes.

---

## Our ML approach — LoRA fine-tuning

### Why LoRA

Training a Whisper-size model from scratch takes hundreds of thousands of hours of audio and a GPU cluster running for weeks. Even *traditional* full fine-tuning of all 200M weights is heavy on memory, slow, and risks catastrophic forgetting (the model loses what it already knew). LoRA solves all three: **freeze the 200M base weights, attach ~3M trainable "adapter" parameters inside the attention layers, and only train those.**

At inference time the adapter is mathematically equivalent to one matrix-addition merged into the base model — zero runtime overhead. The adapter file is **~13 MB** versus the **~226 MB** base model, which means we can version it, swap it, A/B-test it, or eventually ship per-deployment adapters (e.g., one per gurdwara, trained on its ragis' voices).

### Our configuration ([`configs/training/surt_lora_mac.yaml`](../configs/training/surt_lora_mac.yaml))

| Knob | Value | Why |
|---|---|---|
| Where LoRA inserts | Attention layers: `q_proj` / `k_proj` / `v_proj` / `out_proj` | Where most domain adaptation lives in Whisper |
| Rank `r` / alpha / dropout | 16 / 32 / 0.1 | Standard PEFT defaults for Whisper-small (~3M trainable params, ~1% of base) |
| Learning rate / schedule | 1e-5, cosine annealing, 10% warmup | Whisper convention; gentler than CTC |
| Weight decay / max grad norm | 0.01 / 1.0 | Standard Whisper FT regularizers |
| Batch | 4 × grad accum 2 → effective 8 | Fits 48 GB MPS comfortably |
| Precision | fp32 on Apple Silicon MPS | fp16 blocked by our torch 2.5 pin (~30% throughput cost) |

### Holdout discipline

The benchmark covers 4 specific shabads (`4377`, `1821`, `1341`, `3712`). [`scripts/pull_dataset.py`](../scripts/pull_dataset.py) reads holdout rules from [`configs/datasets.yaml`](../configs/datasets.yaml) and **structurally excludes those 4 shabad IDs + 9 benchmark/OOS video IDs from every training pull**. Unit-tested ([`tests/test_pull_corrections.py`](../tests/test_pull_corrections.py)) — anyone who removes the rule breaks the build. This is the entire reason our paired-benchmark numbers can be honest at all.

---

## What our metrics say

| Metric | Current best | What it means |
|---|---|---|
| **Validation loss** (held-out shabads inside the 300h pull) | **0.035007 at step 19000**, down from 0.036197 at step 2000 (~3.3% relative improvement) | The model is still learning. No overfit signal at pause point. |
| **Paired benchmark frame accuracy** (12 cases, 4 shabads, 1s frames) | **93.1%** (12/12 correct shabad locks) | First non-cheating run to beat the prior best (92.8%). **But it's the full system score, not the LoRA alone** — Whisper + LoRA + a separate Whisper for shabad-ID + state machine + smoother + retro-buffer. The LoRA's marginal contribution is unmeasured. |
| **Assisted-OOS** (5 unseen shabads, *machine-assisted labels*) | **61.3%, 5/5 locks** | Suggestive, not honest — the labels were generated by a labeler scoring against itself, which is circular. Honest number requires hand-corrected gold labels (owed, see A2 below). |
| **Karanbir's reference** (his system, his benchmark) | "~65–70%" on his live shabad-unaware task | Roughly the same range as our assisted-OOS. Not apples-to-apples but a useful sanity check. |

**What 93.1% does NOT mean:** not a model-only score, not a deployment number, not yet promotable. Production claims require gold OOS, multi-seed variance, and a broader honest-eval set — none of which exist yet.

---

## Sequential checklist of remaining work

Ordered by dependency. Earlier items unblock later ones.

### Phase A — Validate what we have (model side)

- [ ] **A1.** Score the **step-19000 v7 checkpoint** through the confirmed runtime (paired + assisted-OOS). Tells us whether epoch-2's validation-loss improvement translated to runtime gains.
- [ ] **A2.** **Hand-correct the 5 OOS gold labels.** Drafts exist at `eval_data/oos_v1/drafts/`. Single most important unblocking task — without it, every "OOS" we report is suggestive.
- [ ] **A3.** Run the **LoRA-isolation ablation**: same runtime with vs without the v7 adapter. Quantifies how much of the 93.1% is the model vs the engine.
- [ ] **A4.** Decide: **resume v7 training** from `checkpoint-19000` to complete 3 epochs (~11 more wall-clock hours on M4 Pro), or freeze v7 and skip to B.

### Phase B — Multi-seed + ablations

- [ ] **B1.** Re-run v7 at 2 additional seeds. Promotion requires best-of-3 with cross-seed variance < 2 pts.
- [ ] **B2.** Phase 4 ablations: data-scale curve (10/25/50/100h), LoRA-rank sweep (8/16/32/64), target-modules sweep (attention-only vs attention+MLP), augmentation sweep.
- [ ] **B3.** **Decision gate:** if best ablation < 88% paired, switch base architecture (to `surt-medium` or to `indicconformer-pa-v3-kirtan`, the same family Karanbir uses).

### Phase C — Honest evaluation at scale

- [ ] **C1.** Curate **OOS v2**: ≥ 20 shabads, ≥ 5 ragis, mixed conditions. Hand-labeled.
- [ ] **C2.** Per-slice accuracy + calibration curve.

### Phase D — User-feedback flywheel (parallel)

- [ ] **D1.** Apple Developer enrollment (you). Gates everything iOS-side.
- [ ] **D2.** First signed TestFlight build; you on it as internal tester.
- [ ] **D3.** Real-device verification: Keychain persistence, ANE latency (target < 5s to first caption), end-to-end correction → Supabase.
- [ ] **D4.** Add beta testers; accumulate corrections.
- [ ] **D5.** **First fine-tune on corrections data.** The long-term moat — data Surinder doesn't have access to.

### Phase E — Production

- [ ] **E1.** Publish validated adapter to HuggingFace with a full model card.
- [ ] **E2.** Hosted inference endpoint (Modal recommended) for non-mobile clients.
- [ ] **E3.** App Store full release.

---

# iOS App (Sangat)

SwiftUI iOS app that runs `surt-small-v3` on-device via [WhisperKit](https://github.com/argmaxinc/WhisperKit) and snaps Whisper's noisy Gurmukhi to canonical SGGS lines via fuzzy matching.

### On-device inference

Model compiled to Core ML (6-bit ANE-quantized); loads in ~5s on iPhone with the Apple Neural Engine (CPU-only fallback on simulator). The full **226 MB model + 13 MB shabad corpus (5,505 shabads / 60,067 lines)** ships in the app bundle and works offline.

### Snap-to-canonical UI

The model produces noisy transcripts; the app **never displays them**. Output is always the BaniDB canonical line resolved from `(shabad_id, line_idx)`. Misspelled Gurmukhi in a religious context is unacceptable — the integer-id design makes correctness structural, not aspirational.

### Corrections feedback loop (the data flywheel)

When a user corrects a mis-identification, the app captures the last few seconds of audio (16 kHz mono AAC, retention-pruned) + the correction type + app/model version + a Keychain-persistent anonymous device UUID. Writes to a local SwiftData outbox; on Wi-Fi (and only when the user has opted in), syncs to a private Supabase project: audio to a private bucket, metadata to a corrections table. [`scripts/pull_corrections.py`](../scripts/pull_corrections.py) ingests confirmed corrections into a training-data manifest with the same benchmark-holdout discipline.

Three default-OFF consent gates (`audioCaptureOptIn`, `uploadOptIn`, `wifiOnlyUpload`) plus a server-side delete-my-data Edge Function. This audio is the only training signal Surinder doesn't already have — the only structural reason our LoRA can outperform the upstream baseline over time.

### Status

Sim-verified end-to-end. Real-device testing gated on Apple Developer enrollment (see [docs/beta_readiness_plan.md](beta_readiness_plan.md)). Corrections accumulating: **zero until users have the app.**
