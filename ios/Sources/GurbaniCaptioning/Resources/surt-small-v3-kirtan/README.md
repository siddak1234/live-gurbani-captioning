# surt-small-v3-kirtan Core ML bundle

This directory holds the Core ML compiled model files that
`CaptionEngine` loads via `WhisperKit` at runtime. The files are large
(~226 MB total, 6-bit uniform palettization) and **gitignored**; this
README is committed so the directory itself is trackable and
`Package.swift`'s `.copy(...)` resource declaration always finds it.

The HuggingFace tokenizer files (`tokenizer.json`, `vocab.json`,
`merges.txt`, etc.) also live here — also gitignored — and are loaded by
WhisperKit via `TextDecoder`'s tokenizer fallback chain (modelFolder is
the second search path, used when the explicit `tokenizerFolder` is
unset). They're copied here as part of `make ios-bundle-model` from the
`whisperkit-generate-model` output.

## Expected contents

```
surt-small-v3-kirtan/
├── README.md              (committed — this file)
├── AudioEncoder.mlmodelc/    (gitignored, ~67 MB)
├── MelSpectrogram.mlmodelc/  (gitignored, ~370 KB)
├── TextDecoder.mlmodelc/     (gitignored, ~158 MB)
└── tokenizer.json, vocab.json, merges.txt, …  (gitignored, ~5 MB)
```

## Which quantization variant (and why not 4-bit)

Validated by [`ModelParityTests`](../../../Tests/GurbaniCaptioningTests/ModelParityTests.swift):

| Variant | Generate flags | Size | Parity |
|---|---|---|---|
| **6-bit uniform** | `--allowed-nbits 6 --force-recipe-nbits` | ~226 MB | **PASS — current** |
| 8-bit uniform | `--allowed-nbits 8 --force-recipe-nbits` | ~273 MB | pass (fallback) |
| fp16 | (auto `-fp16-fallback`) | ~465 MB | pass (baseline) |
| 4-bit + OD | `--allowed-nbits 4 --outlier-decomp` | ~212 MB | **FAIL** |

The 4-bit + outlier-decomposition recipe crushes weights critical to
surt-small-v3's Punjabi fine-tune: the decoder emits `<|endoftext|>` as
its first generated token after prefill, producing an empty transcript.
The fix was **`--force-recipe-nbits`** — it applies uniform N-bit instead
of whisperkittools' mixed-bit recipe search, whose quality target is
measured against English test data and therefore selectively crushes the
Gurmukhi-critical layers. 6-bit uniform reproduces the HF Python
reference exactly (`ਨ ਪਰਮੇਸਰ ਕਾ ਥਾਨੁ`) at roughly the same size the
broken 4-bit variant would have been.

If any of the three `.mlmodelc/` directories is missing,
`CaptionEngine.resolveModelFolder` will throw `.modelFolderNotFound` at
`prepare()` time and `AppEnvironment` falls back to `DemoCaptionSource`.

## How to populate

1. Run the Core ML export pipeline from the repo root:

   ```bash
   .venv/bin/python scripts/export_coreml.py \
     --base-model surindersinghssj/surt-small-v3 \
     --output-dir coreml_export \
     --skip-validation
   ```

   (Requires `.venv-coreml`-style isolated Python env per
   `docs/ios_deployment.md`; takes ~30-50 min on M-series Macs for the
   4-bit palletized variant.)

2. Copy the resulting `.mlmodelc/` directories here:

   ```bash
   make ios-bundle-model
   ```

   The default `COREML_EXPORT_VARIANT` is the fp16-fallback variant
   (`coreml_export/surindersinghssj_surt-small-v3-fp16-fallback`).
   Override on the CLI to bundle a different variant.

   …or manually:

   ```bash
   cp -R coreml_export/surindersinghssj_surt-small-v3-fp16-fallback/*.mlmodelc \
         ios/Sources/GurbaniCaptioning/Resources/surt-small-v3-kirtan/
   ```

## Why these aren't committed

- Each `.mlmodelc` is a directory of binary weight blobs (~150 MB
  total per quantized variant). Git handles them poorly: diffs are
  meaningless, pushes are slow, clones balloon.
- The artifacts are *reproducible* — given the same base model + same
  `whisperkit-generate-model` invocation, the bytes match. No
  information is lost by gitignoring them.
- Reviewers of an M5.x PR don't need the 220 MB to reason about a
  Swift wiring change.

## Why a placeholder model name

`config.modelPath` in `AppEnvironment.production()` is hardcoded to
`"surt-small-v3-kirtan"`, which is what `Bundle.module.url(forResource:
"surt-small-v3-kirtan", withExtension: nil)` looks for. The
directory name here matches that lookup — change one, change the
other. The leading suffix `surindersinghssj_surt-small-v3` is
whisperkittools' on-disk format; the iOS app uses a friendlier name.

## Updating the model

When a new fine-tune lands (e.g. Phase 3 v7 LoRA merged into surt):

1. Re-run `scripts/export_coreml.py` with the new adapter.
2. `make ios-bundle-model` (or manual `cp -R`).
3. Smoke-test in the simulator — `prepare()` should now succeed and
   `LiveCaptionSource` should drive the app instead of
   `DemoCaptionSource`.

No code changes required — this is a single-file (well,
single-directory) swap.
