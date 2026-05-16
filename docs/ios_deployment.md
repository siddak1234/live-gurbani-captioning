# iOS deployment — end-to-end

This doc is the cold-read recipe: a fresh engineer with no prior context follows it and ends up with the GurbaniCaptioning app running on an iPhone, using our fine-tuned `surt-small-v3` model on the Apple Neural Engine.

If anything below doesn't work in your environment, that's a documentation bug — open an issue, don't work around it silently.

## The pipeline

```
[ Train on Mac ]  →  [ Export to Core ML ]  →  [ Build iOS app ]  →  [ Ship to device ]

   M0–M3:                M4:                       M5:                    you, with Xcode
   PyTorch + MPS         whisperkittools           Swift + WhisperKit
   surt-small-v3         + ANE quantization        + corpus matcher
   LoRA fine-tune
```

Each arrow is one script invocation. No manual model surgery.

## Prerequisites

| Component | Required version | Why |
|---|---|---|
| macOS | 14.0 (Sonoma) or later | Core ML / coremltools features |
| Xcode | 15.0 or later | Swift 5.9, WhisperKit minimum |
| Apple Silicon Mac | yes | ANE is required to compile the .mlpackage |
| iPhone | A15 or later (iPhone 13+) | Faster ANE; older chips still work but slower |
| Python | 3.10–3.12 | for the desktop training + export side |
| Mac dependencies | `pip install -r requirements-mac.txt` | torch + transformers + whisperkittools |

## Step 1 — Fine-tune on the Mac (skip if you already have an adapter)

Follow [`docs/training_on_mac.md`](training_on_mac.md). The output you need at the end of that doc is a LoRA adapter directory, typically:

```
lora_adapters/surt_mac_v1/
├── adapter_model.safetensors
├── adapter_config.json
├── base_model.txt          # contains "surindersinghssj/surt-small-v3"
└── ...
```

If you skip fine-tuning and want to ship the stock model, that's fine — the export script accepts no adapter and uses the base directly. Expect slightly worse kirtan accuracy than a tuned model.

## Step 2 — Export to Core ML (M4)

```bash
python scripts/export_coreml.py \
  --config configs/export/coreml_ane.yaml \
  --adapter-dir lora_adapters/surt_mac_v1 \
  --output-dir ios/Sources/GurbaniCaptioning/Resources/
```

What this does:

1. Loads the HF base + LoRA adapter and merges them via `peft.merge_and_unload`.
2. Saves the merged model to a temp HF-format directory.
3. Invokes `whisperkittools` to compile it to a `.mlpackage` with 4-bit palettization on encoder + decoder, stateful audio encoder for streaming, ANE-targeted attention.
4. Optionally validates numerical parity against the original HF model on one of our smoke clips.

Resulting artifact: `ios/Sources/GurbaniCaptioning/Resources/surt-small-v3-kirtan.mlpackage`. Roughly 150–250 MB depending on quantization choices.

**Verify on the Mac before moving to the device:**

```bash
python scripts/benchmark_ane_latency.py \
  --mlpackage ios/Sources/GurbaniCaptioning/Resources/surt-small-v3-kirtan.mlpackage \
  --test-clip audio/IZOsmkdmmcg_16k.wav \
  --warmup 2 --iters 5
```

Expected on M4 Pro: RTF well under 0.1, first-token under 200 ms. If RTF > 1.0 (slower than realtime) on the Mac, the iPhone will struggle too — go back and lower the quantization aggressiveness in `configs/export/coreml_ane.yaml`.

## Step 3 — Generate the on-device corpus

The iOS app bundles a JSON copy of the SGGS corpus so it can do canonical line snapping fully offline.

```bash
# Populate the BaniDB cache if you haven't already:
python scripts/build_corpus.py

# Materialize the iOS bundle resource:
python scripts/build_ios_corpus.py
```

Output: `ios/Sources/GurbaniCaptioning/Resources/shabads.json` (~40–60 MB, gitignored).

## Step 4 — Open the Swift package in Xcode

```bash
open ios/Package.swift
```

Xcode reads `Package.swift` directly — there is no `.xcodeproj` to maintain. Wait for SPM to resolve dependencies (downloads WhisperKit from GitHub on first open).

Select the `GurbaniCaptioningApp` scheme and your iPhone as the destination. The first build is slow because Xcode has to compile the `.mlpackage` for the device's specific Neural Engine version — give it a minute.

## Step 5 — Wire the WhisperKit streaming call

The current `CaptionEngine.transcribeStream(using:)` is a placeholder — it throws so the call doesn't silently no-op. Replace its body with the streaming call from the WhisperKit version you actually installed. Two common shapes:

**A. Newer WhisperKit (≥ 0.9.x):**

```swift
let stream = try await whisper.streamingTranscribe(
    language: config.language,
    chunkSeconds: config.chunkSeconds
)
for await chunk in stream {
    let asr = AsrChunk(start: chunk.startTime, end: chunk.endTime, text: chunk.text)
    let guess = stateMachine.processChunk(asr)
    await notifyGuess(guess)
    await notifyState(stateMachine.state)
}
```

**B. Older WhisperKit + AVAudioEngine tap:**

Capture the mic with `AVAudioEngine.inputNode.installTap(...)`, accumulate into a `Float` array, every `chunkSeconds` call `whisper.transcribe(audioArray: accumulated)`, post-process the result through the state machine.

If neither shape compiles cleanly, run `swift package describe` in `ios/` and check the WhisperKit version that resolved. Match its public API surface.

## Step 6 — Run on device

Build and run on your iPhone. Permission prompts the first time: microphone access (required). Tap **Listen** — the status banner moves through *Listening → Tentative → Committed* as the engine identifies the shabad, and the Gurmukhi line updates each chunk.

**Sewadar workflow:**
- *Listen* to start. The model identifies the shabad within ~15–30 seconds (3 consecutive matching votes per `ShabadCommitConfig.default`).
- *Reset* drops the committed shabad and re-listens — use this when the kirtan transitions to a new shabad.
- Tap-and-hold a confidence number (planned in M6+) to manually pick a shabad if blind ID is flaky.

## What's deliberately not in v0.1

- **App icon, launch screen** — first-light only.
- **Raag info, history scroll, manual shabad picker** — UI scope is "the live caption works."
- **Persistent settings, account, sync** — none.
- **Sewadar UI for line correction** — coming when we have real Sewadar feedback to design against.
- **Background audio mode / phone-call interruption handling** — needs `AVAudioSession` category tuning.

## Failure modes and how to read them

| Symptom | Likely cause | Where to look |
|---|---|---|
| App launches, taps "Listen" → immediate error | `transcribeStream` placeholder hasn't been wired | `CaptionEngine.swift` — replace the stub |
| "ShabadCorpus: missing bundle resource 'shabads.json'" | Forgot Step 3 | Run `scripts/build_ios_corpus.py` |
| Crash on `WhisperKit(model:)` init | `.mlpackage` not in bundle, or wrong name | Check `CaptionEngine.Config.modelPath` matches the file in `Resources/` |
| Listening forever, never commits | Confidence margin too high for the chunk quality | Lower `ShabadCommitConfig.minVoteScore` / `.minVoteMargin` in `ContentView.swift`'s viewModel |
| Committed shabad is consistently wrong | Either the corpus is missing the actual shabad, or the matcher is too lenient on margins | Check `ShabadCorpus.allShabadIds` contains the GT shabad; raise `.minVoteMargin` |
| Audio captured but transcript is garbage | Wrong language tag, or model loading the wrong weights | Verify `CaptionEngine.Config.language == "punjabi"` and the `.mlpackage` came from a kirtan-fine-tuned model |
| Battery drains absurdly fast | ANE attention path not being used | Confirm `configs/export/coreml_ane.yaml` had `ane_attention: true` and verify with WhisperKit's compute-unit info logs |

## Parity tests (M5 audit gate)

Before shipping, the Swift matcher must agree with the Python matcher on ≥ 90% of test cases. Generate the fixtures from Python:

```bash
# (planned — script lands as part of M6 hardening)
python scripts/dump_matcher_fixtures.py \
  --out ios/Tests/GurbaniCaptioningTests/Fixtures/matcher_v1.json
```

Then in Xcode, ⌘U to run the test target. If the parity check fails below 90%, the Swift matcher needs tuning (most likely the `wRatio` blend weights) before the app is trustworthy.

## Where to go from here

Once v0.1 ships internally:

- Add a "Sewadar correction" UI: tap the displayed line to advance/retreat. Feeds a labeled correction log we can use to fine-tune further.
- Background audio capture so the app can run while the iPhone is locked.
- Distillation: train a smaller (~50M param) student model from the production teacher, ship a 30 MB `.mlpackage` for Watch / older iPhones.
