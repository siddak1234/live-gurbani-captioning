# GurbaniCaptioning (iOS / macOS)

On-device live kirtan transcription. Wraps the [WhisperKit](https://github.com/argmaxinc/WhisperKit) Swift package around the project's fine-tuned `surt-small-v3` Core ML model, plus a Swift port of the Python matcher + shabad state machine.

This directory is a Swift Package — open it in Xcode by selecting `Package.swift`. No `.xcodeproj` to hand-edit.

## Layout

```
ios/
├── Package.swift                                 # SPM manifest (Xcode opens this)
├── README.md                                     # this file
└── Sources/
    ├── GurbaniCaptioning/                        # library target
    │   ├── CaptionEngine.swift                   # WhisperKit + state machine orchestrator
    │   ├── ShabadStateMachine.swift              # blind ID + commit/reset
    │   ├── FuzzyMatcher.swift                    # rapidfuzz-style scorer (pure Swift)
    │   ├── ShabadCorpus.swift                    # bundled JSON loader
    │   ├── Models.swift                          # data types
    │   └── Resources/
    │       └── shabads.json                      # corpus, generated (gitignored)
    └── GurbaniCaptioningApp/                     # executable target (the actual app)
        ├── GurbaniCaptioningApp.swift            # SwiftUI @main entry
        └── ContentView.swift                     # minimal Sewadar UI
```

## Build prerequisites

| | Required |
|---|---|
| Xcode | 15.0+ |
| macOS | 14.0+ (development) |
| iOS deployment target | 17.0+ |
| Apple Silicon Mac | yes (Core ML model compilation) |
| The fine-tuned `.mlpackage` | from `scripts/export_coreml.py` (M4) |
| The corpus JSON | from `scripts/build_ios_corpus.py` (run once before building) |

## One-time setup

From the repo root:

```bash
# 1. Make sure the BaniDB corpus cache is populated
python scripts/build_corpus.py

# 2. Generate the bundled corpus resource (~40-60 MB)
python scripts/build_ios_corpus.py

# 3. Export your fine-tuned model to Core ML
python scripts/export_coreml.py \
  --adapter-dir lora_adapters/surt_mac_v1 \
  --output-dir ios/Sources/GurbaniCaptioning/Resources/

# 4. Open in Xcode
open ios/Package.swift
```

Step 3 produces a `.mlpackage` that WhisperKit loads at runtime. The `CaptionEngine.Config.modelPath` in `ContentView.swift` references it by name.

## Run

In Xcode, select the `GurbaniCaptioningApp` scheme and a device target (iPhone 13 or newer recommended — ANE acceleration depends on A15+). Hit Run. First launch will compile the Core ML model for the device — takes ~30 seconds.

## Architecture mirror

The Swift side mirrors the Python side one layer at a time:

| Python (Layer 1) | Swift |
|---|---|
| `src/matcher.py` (rapidfuzz blend) | `FuzzyMatcher.swift` |
| `src/shabad_id.py` + `src/smoother.py` | `ShabadStateMachine.swift` |
| `src/engine.py` (batch) | n/a — streaming-only on iOS |
| `src/streaming_engine.py` (Python) | `CaptionEngine.swift` (native streaming) |

The Python streaming engine validates the API; the Swift one ships it.

## Parity tests

`Tests/GurbaniCaptioningTests/` (when added) holds fixtures dumped from Python: a set of (chunk_text, lines, top1_index, top1_score) tuples that the Swift matcher is expected to reproduce within tolerance. The audit gate for M5.3 is: ≥90% of fixture cases land the same top-1 line as Python's `match_chunk()`.

To generate fixtures from Python after a training run:

```bash
python scripts/dump_matcher_fixtures.py \
  --out ios/Tests/GurbaniCaptioningTests/Fixtures/matcher_v1.json
```

(That script is part of M6 — coming.)

## Known gaps (deliberate)

- **`transcribeStream` is a placeholder.** WhisperKit's exact streaming API surface changes per version; the call site is one method in `CaptionEngine.swift`. Fill it in against the installed WhisperKit version.
- **No tests committed yet.** The parity-fixture infrastructure lands in M6 along with the Python dump script.
- **No app icons / launch screen.** First-light functional only.
- **No raag info, history scroll, or shabad picker UI.** v0.1 ships the live caption only.

## License

Apache 2.0 (inherits from the repo root).
