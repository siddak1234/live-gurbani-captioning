# Architecture

This project is deliberately split into three decoupled layers so the same inference logic can run inside a benchmark script today and inside an iOS app tomorrow without a rewrite.

## Three-layer decoupling

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 3: Frontends                                              │
│                                                                  │
│  • scripts/run_path_a.py        — benchmark runner (CLI)         │
│  • scripts/run_path_b_hmm.py    — Path B benchmark runner        │
│  • ios/CaptionEngine.swift      — iOS app (planned, M5)          │
│                                                                  │
│  Knows about: argv, GT JSON, submission JSON, AVAudioEngine,     │
│  SwiftUI. Does not know about ASR backends or matcher details.   │
└────────────────────────┬─────────────────────────────────────────┘
                         │ calls predict() / streaming variants
┌────────────────────────▼─────────────────────────────────────────┐
│  Layer 2: Engine library (src/engine.py)                         │
│                                                                  │
│  • predict(audio, corpora, *, shabad_id, uem_start, config)      │
│        → PredictionResult                                        │
│  • EngineConfig, Segment, PredictionResult dataclasses           │
│                                                                  │
│  Knows about: how to chain ASR → match → smooth.                 │
│  Does not know about: file paths, submission JSON, benchmark     │
│  layouts, UI.                                                    │
└────────────────────────┬─────────────────────────────────────────┘
                         │ calls
┌────────────────────────▼─────────────────────────────────────────┐
│  Layer 1: Inference primitives                                   │
│                                                                  │
│  • src/asr.py            — transcribe(audio) → list[AsrChunk]    │
│  • src/matcher.py        — score_chunk, match_chunk, TfidfScorer │
│  • src/shabad_id.py      — identify_shabad, per_chunk_global     │
│  • src/smoother.py       — smooth, smooth_with_stay_bias         │
│  • src/path_b/           — CTC encoder + tokenizer + HMM         │
│                                                                  │
│  Pure: typed inputs → typed outputs. No file I/O beyond audio    │
│  loading and caching. No CLI, no JSON.                           │
└──────────────────────────────────────────────────────────────────┘
```

## Why this matters for iOS

iOS shipping is feasible because:

1. **Layer 1 primitives are pure logic.** `src/matcher.py` and `src/smoother.py` are stateless and Punjabi-text-only — portable to Swift line-for-line. `src/shabad_id.py` is a small voting algorithm — also portable.
2. **Layer 2 (`src/engine.py`) is the contract.** Anything calling `predict(audio, corpora, ...)` is interchangeable: today it's the benchmark runner, tomorrow it's the Swift app calling a Core ML model with the same logical surface.
3. **Layer 3 frontends are the only platform-specific code.** The CLI runner and the iOS app share no code with each other, only with Layer 2.

The audit gate that locks this in: `scripts/run_path_a.py` makes **zero direct calls** to ASR/matcher/smoother functions. All inference goes through `engine.predict()`. Verified by AST analysis during the engine extraction — running `grep -E 'transcribe\(|match_chunk\(|score_chunk\(|smooth\(' scripts/run_path_a.py` returns no hits.

## Data flow

### Batch (current — benchmark runner)

```
GT JSON → audio path → engine.predict() → submission JSON
                          │
                          ├─ transcribe (whole file)
                          ├─ identify_shabad (if blind)
                          ├─ match_chunk × N  (per ASR chunk)
                          ├─ smooth / smooth_with_stay_bias
                          └─ pre-commit segments (if live + tentative_emit)
```

### Streaming (M3 — planned)

```
mic / file → audio_buffer (ring, 30s lookback)
                ↓ every N seconds
              streaming_engine.process_pcm()
                ├─ append to internal transcript buffer
                ├─ if no shabad confirmed: re-run identify_shabad on buffer
                │     └─ if confidence ≥ threshold: commit shabad_id
                ├─ match latest chunk against committed shabad's lines
                ├─ smooth + state-machine (stay vs advance vs confirm)
                └─ emit Segment if line transition detected
```

### iOS (M5 — planned)

```
AVAudioEngine → 16kHz mono PCM
        ↓
  WhisperKit (Core ML on ANE)
        ↓ word-level transcripts + timestamps
  CaptionEngine.swift orchestrates:
        ├─ ShabadStateMachine (port of shabad_id.py)
        ├─ FuzzyMatcher (port of matcher.py — JaroWinkler + token-set)
        └─ corpus lookup (shabads.json embedded)
        ↓
  SwiftUI surface (live caption, confirm/reset buttons)
```

## Path A vs Path B (within Layer 1)

The repo currently ships two engines, frozen at different scores:

| Engine | Score | Architecture | Status |
|---|---|---:|---|
| Path A v3.2 | 86.5% blind+live | faster-whisper + rapidfuzz matcher + smoother | frozen canonical |
| Path B | 70.3% baseline / 72.9% with 50-step LoRA | w2v-bert CTC + loop-aware shabad HMM | development |
| x4 (Path A with surt) | 74.0% honest | surt-small-v3 + same matcher/smoother | best generalization candidate |
| v5_mac_baseline | 74.0% honest / neutral | surt-small-v3 + 200-clip Mac MPS LoRA | pipeline proof, not promoted |
| v5b_mac_diverse | 65.6% honest / regressed | 2,544-clip / 4.936h diverse surt-small-v3 Mac MPS LoRA | negative diagnostic; do not promote |
| v5b two-pass proxy | 87.1% diagnostic | v3.2 pre-lock segments + v5b post-lock alignment | validates ID-lock direction; not yet a runtime engine |

Path A is the deployment target because Whisper architectures have the cleanest Core ML / WhisperKit pipeline. Path B remains as research scaffolding for forced-alignment experiments.

## Phase 2.7 architecture direction

The Phase 2.6 diagnostic says the next architecture should be state-based, not shabad-route-based:

```
first 30s / no committed shabad:
    v3.2 faster-whisper path
    ├─ robust chunk-vote shabad ID
    └─ tentative captions

after committed shabad_id:
    v5b surt-small-v3 LoRA path
    └─ match only inside the locked shabad's line set
```

This is cleaner than x5/x6 route tables because the switch is based on engine state and time, not on benchmark shabad IDs. It must still pass OOS before promotion.

Implementation split:

- `src/idlock_engine.py` is the runtime Layer 2 orchestrator. It composes two `engine.predict()` calls and merges segments at `uem.start + blind_lookback`. It imports no argparse and knows no benchmark paths.
- `scripts/run_idlock_path.py` is the Layer 3 benchmark runner. It loads GT/audio/corpus files, builds pre/post configs, calls the library, and writes submission JSON.
- `scripts/merge_idlock_submissions.py` remains a Phase 2.6 diagnostic transformer only. It should not be used as evidence that the runtime path works.

Phase 2.7 result: the architecture was implemented cleanly, but `v5b_idlock_runtime` scored only `75.6%`. The failure was current runtime blind-ID, not the post-lock adapter: two `kZhIA8P6xWI` starts committed to the wrong shabad. The next architecture checkpoint is Phase 2.8: recover/pin ASR reproducibility, then prototype timestamp/alignment paths before more training scale.

Phase 2.8 first signal: `phase2_8_idlock_preword` scores `86.6%` by using word timestamps only for the pre-lock ID window and v5b after lock. This restores correct shabad locks but still misses the promotion gate; the remaining architecture work is post-lock alignment/timing, not more route tables.

## Configuration surface

Per-experiment settings live in `submissions/<run>/notes.md` (frozen historical record per CLAUDE.md convention). Per-pipeline reusable settings live in `configs/` (planned, M0.3 + M1.1): training hyperparameters, export quantization profiles, dataset registry.

## What this architecture forbids

- **Engine library importing argparse, sys.argv, or any benchmark-specific paths.** Audit: `grep -E '^(import|from) argparse' src/engine.py src/asr.py src/matcher.py src/smoother.py src/shabad_id.py` returns empty.
- **Runners doing inference calls directly.** All ASR/match/smooth calls must route through `engine.predict()`.
- **Path A and Path B sharing engine state.** They share infrastructure (audio fetch, corpus, scoring) but their inference internals are separate. A bug in one doesn't break the other.
- **Hardcoded user paths in tracked files.** Enforced by `.claude/hooks/pre-write-secret-scan.sh` and verified at commit time.
