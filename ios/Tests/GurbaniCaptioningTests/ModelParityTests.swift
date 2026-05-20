//
//  ModelParityTests.swift
//  GurbaniCaptioningTests — M5.7d (numerical parity)
//
//  Compares the bundled Core ML model's transcription against an HF
//  Python reference dumped by `scripts/dump_parity_fixture.py`. The
//  point is to catch:
//    1. A broken Core ML export (weights, tokenizer, vocab mismatch).
//    2. WhisperKit configuration / decoder option drift between
//       Swift and the HF reference.
//
//  Why gated behind `WHISPER_PARITY_TEST=1`: loading + specializing
//  the .mlmodelc takes ~30–60s on first run (Core ML compiles the
//  ANE-targeted kernels for the local machine). We don't want
//  routine `swift test` to pay that cost on every change.
//
//  When to run:
//    - On every regen of the bundled model.
//    - On every WhisperKit version bump.
//    - Periodically as a sanity check that the fixture stays valid.

import XCTest
import AVFoundation
@testable import GurbaniCaptioning

#if canImport(WhisperKit)
import WhisperKit
#endif

@MainActor
final class ModelParityTests: XCTestCase {

    // MARK: - Fixture shape

    private struct ExpectedTranscript: Codable {
        let schemaVersion: Int
        let audioFilename: String
        let durationSeconds: Double
        let sampleRateHz: Int
        let modelId: String
        let language: String
        let expectedText: String

        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case audioFilename = "audio_filename"
            case durationSeconds = "duration_seconds"
            case sampleRateHz = "sample_rate_hz"
            case modelId = "model_id"
            case language = "language"
            case expectedText = "expected_text"
        }
    }

    // MARK: - Gate

    /// Skip rationale shared across every parity-mode test.
    private func skipIfNotEnabled() throws {
        guard ProcessInfo.processInfo.environment["WHISPER_PARITY_TEST"] == "1" else {
            throw XCTSkip("Set WHISPER_PARITY_TEST=1 to run model parity tests (~60s each).")
        }
    }

    // MARK: - Tests

    #if canImport(WhisperKit)

    /// Whisperkit-side smoke: load the bundled .mlmodelc set, run
    /// inference on the fixture audio, and assert the result is
    /// non-empty Gurmukhi (a positive signal that the export
    /// shipped a working model). Strictly weaker than the WER
    /// comparison below but useful when the fixture-vs-output WER
    /// is naturally noisy.
    func testBundledModelProducesGurmukhiOutput() async throws {
        try skipIfNotEnabled()
        let (whisper, audioSamples, _) = try await loadParityRig()
        // Diagnostic dump so a failing test points at the cause
        // instead of leaving you to grep the WhisperKit logs.
        let sampleMin = audioSamples.min() ?? 0
        let sampleMax = audioSamples.max() ?? 0
        let rms = sqrt(audioSamples.reduce(0) { $0 + $1 * $1 } / Float(audioSamples.count))
        print("DIAG: modelVariant=\(whisper.modelVariant) modelState=\(whisper.modelState)")
        print("DIAG: tokenizer=\(whisper.tokenizer == nil ? "nil" : "loaded")")
        print("DIAG: audio samples=\(audioSamples.count), duration=\(Double(audioSamples.count) / 16000.0)s")
        print("DIAG: audio range=[\(sampleMin), \(sampleMax)], rms=\(rms)")

        // (Diagnostic A) Use WhisperKit's own audio loader via the
        // file-path API. Sidesteps any subtle issue in our Swift-side
        // AVFoundation extraction (channel layout, sample range).
        let audioURL = Bundle.module.url(forResource: "izos_30s_snippet_5s",
                                          withExtension: "wav")!
        let options = DecodingOptions(
            verbose: true,
            task: .transcribe,
            language: "pa",
            temperature: 0.0,
            skipSpecialTokens: true,
            withoutTimestamps: true
        )
        let pathResults = try await whisper.transcribe(
            audioPath: audioURL.path,
            decodeOptions: options
        )
        print("DIAG: file-path API: \(pathResults.count) result(s)")
        for (i, t) in pathResults.enumerated() {
            print("DIAG: pathResult[\(i)] text=\"\(t.text)\" segments=\(t.segments.count) language=\(t.language)")
            for (j, seg) in t.segments.enumerated() {
                // Dump the raw token IDs + each token's tokenizer-resolved
                // string. When `text=""` despite a non-zero token count,
                // this is the only way to tell whether the decoder hit
                // `<|endoftext|>` immediately, looped on a special token,
                // or generated real content that got filtered downstream.
                let decoded: [String] = seg.tokens.map { id in
                    let tok = whisper.tokenizer?.convertIdToToken(id) ?? "<id?>"
                    return "\(id):\(tok)"
                }
                print("  DIAG: pSegment[\(j)] start=\(seg.start) end=\(seg.end) text=\"\(seg.text)\" tokens=\(seg.tokens.count) noSpeechProb=\(seg.noSpeechProb)")
                print("    DIAG: pSegment[\(j)] tokenIds=\(seg.tokens)")
                print("    DIAG: pSegment[\(j)] tokenStrs=\(decoded)")
            }
        }

        // (Diagnostic B) The original Float-array API.
        let (text, transcriptions) = try await transcribeWithDiagnostics(whisper, audioSamples: audioSamples)
        for (i, t) in transcriptions.enumerated() {
            print("DIAG: arrayResult[\(i)] text=\"\(t.text)\" segments=\(t.segments.count) language=\(t.language)")
            for (j, seg) in t.segments.enumerated() {
                print("  DIAG: aSegment[\(j)] start=\(seg.start) end=\(seg.end) text=\"\(seg.text)\" tokens=\(seg.tokens.count) noSpeechProb=\(seg.noSpeechProb)")
            }
        }
        XCTAssertFalse(text.isEmpty, "Empty transcript — model load probably broken")
        XCTAssertTrue(containsGurmukhi(text),
                      "Expected Gurmukhi script in transcript — got: \(text)")
    }

    /// Strict parity: compare the bundled Core ML model's output
    /// against the HF reference at a WER threshold. Both sides use
    /// greedy decoding to keep things deterministic. WER target is
    /// loose (~25%) because the bundled model is 4-bit quantized
    /// + OD-MBP-decomposed; some token-level drift is expected vs
    /// the fp16 reference. A tighter bound is M5.7e territory once
    /// we benchmark on a real device with the production decoder
    /// options dialed in.
    func testWERAgainstHFReference() async throws {
        try skipIfNotEnabled()
        let (whisper, audioSamples, expected) = try await loadParityRig()
        let actual = try await transcribe(whisper, audioSamples: audioSamples)

        let wer = computeWER(reference: expected.expectedText, hypothesis: actual)
        let threshold = 0.25  // 25% — generous for 4-bit Core ML
        XCTAssertLessThan(wer, threshold,
                          """
                          WER \(wer) above threshold \(threshold).
                          Reference: \(expected.expectedText)
                          Hypothesis: \(actual)
                          """)
    }

    // MARK: - Helpers (only meaningful when WhisperKit is linked)

    private func loadParityRig() async throws -> (WhisperKit, [Float], ExpectedTranscript) {
        // (1) model location: env override > bundled
        let modelURL: URL
        if let override = ProcessInfo.processInfo.environment["PARITY_MODEL_PATH"],
           !override.isEmpty {
            modelURL = URL(fileURLWithPath: override, isDirectory: true)
            print("DIAG: using PARITY_MODEL_PATH=\(override)")
        } else if let bundled = BundledResources.modelFolderURL(name: "surt-small-v3-kirtan") {
            modelURL = bundled
        } else {
            throw XCTSkip(
                "Bundled .mlmodelc missing — run `make ios-bundle-model` to populate "
                + "ios/Sources/GurbaniCaptioning/Resources/surt-small-v3-kirtan/ before "
                + "WHISPER_PARITY_TEST=1 ./swift test"
            )
        }
        // (2) fixture audio
        guard let audioURL = Bundle.module.url(forResource: "izos_30s_snippet_5s",
                                                withExtension: "wav") else {
            throw XCTSkip("Audio fixture missing — Tests/GurbaniCaptioningTests/Fixtures/")
        }
        let audioSamples = try loadWAVAsFloatArray(audioURL)

        // (3) expected transcript
        guard let expectedURL = Bundle.module.url(forResource: "izos_30s_snippet_5s.expected",
                                                    withExtension: "json") else {
            throw XCTSkip(
                "Expected-transcript JSON missing — regenerate via `python scripts/dump_parity_fixture.py`"
            )
        }
        let expectedData = try Data(contentsOf: expectedURL)
        let expected = try JSONDecoder().decode(ExpectedTranscript.self, from: expectedData)

        // (4) WhisperKit — enable verbose logging during parity runs
        // so any silent model-load issue surfaces in the test log.
        let config = WhisperKitConfig(
            modelFolder: modelURL.path,
            verbose: true,
            logLevel: .debug,
            prewarm: false,  // tests don't need the 2x load tax
            load: true,
            download: false
        )
        let whisper = try await WhisperKit(config)
        return (whisper, audioSamples, expected)
    }

    /// Variant of `transcribe` that returns both the joined text and
    /// the raw `[TranscriptionResult]` so the failing test can dump
    /// segment-level diagnostics.
    private func transcribeWithDiagnostics(
        _ whisper: WhisperKit, audioSamples: [Float]
    ) async throws -> (String, [TranscriptionResult]) {
        let options = DecodingOptions(
            verbose: true,
            task: .transcribe,
            language: "pa",
            temperature: 0.0,
            skipSpecialTokens: true,
            withoutTimestamps: true
        )
        let results = await whisper.transcribeWithResults(
            audioArrays: [audioSamples],
            decodeOptions: options
        )
        guard let first = results.first else {
            return ("", [])
        }
        switch first {
        case .success(let transcriptions):
            let joined = transcriptions.map(\.text).joined(separator: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return (joined, transcriptions)
        case .failure(let error):
            throw error
        }
    }

    private func transcribe(_ whisper: WhisperKit, audioSamples: [Float]) async throws -> String {
        let options = DecodingOptions(
            verbose: false,
            task: .transcribe,
            language: "pa",
            temperature: 0.0,
            skipSpecialTokens: true,
            withoutTimestamps: true
        )
        let results = await whisper.transcribeWithResults(
            audioArrays: [audioSamples],
            decodeOptions: options
        )
        guard let first = results.first else {
            return ""
        }
        switch first {
        case .success(let transcriptions):
            return transcriptions.map(\.text).joined(separator: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        case .failure(let error):
            throw error
        }
    }

    private func loadWAVAsFloatArray(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: file.fileFormat.sampleRate,
            channels: 1,
            interleaved: false
        )!
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(file.length)
        ) else {
            throw NSError(domain: "ModelParityTests", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Could not allocate PCM buffer"])
        }
        try file.read(into: buffer)
        guard let channelData = buffer.floatChannelData?[0] else {
            throw NSError(domain: "ModelParityTests", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "PCM buffer empty"])
        }
        let frameCount = Int(buffer.frameLength)
        return Array(UnsafeBufferPointer(start: channelData, count: frameCount))
    }

    private func containsGurmukhi(_ s: String) -> Bool {
        // Gurmukhi Unicode block: U+0A00..U+0A7F.
        return s.unicodeScalars.contains { (0x0A00...0x0A7F).contains($0.value) }
    }

    /// Token-level WER (word error rate). Splits on whitespace,
    /// computes Levenshtein distance over tokens, divides by
    /// reference token count. Returns `0.0` for identical inputs,
    /// `1.0` when every reference token has to change.
    private func computeWER(reference: String, hypothesis: String) -> Double {
        let refTokens = reference.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        let hypTokens = hypothesis.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        guard !refTokens.isEmpty else { return hypTokens.isEmpty ? 0.0 : 1.0 }
        let dist = levenshtein(refTokens, hypTokens)
        return Double(dist) / Double(refTokens.count)
    }

    /// Standard Levenshtein over arrays of equatable tokens.
    private func levenshtein<T: Equatable>(_ a: [T], _ b: [T]) -> Int {
        let n = a.count, m = b.count
        if n == 0 { return m }
        if m == 0 { return n }
        var prev = Array(0...m)
        var curr = Array(repeating: 0, count: m + 1)
        for i in 1...n {
            curr[0] = i
            for j in 1...m {
                let cost = a[i - 1] == b[j - 1] ? 0 : 1
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            }
            swap(&prev, &curr)
        }
        return prev[m]
    }

    #else

    func testWhisperKitNotLinked() {
        // Build configurations without WhisperKit (e.g., the macOS
        // unit test host that doesn't pull the framework) treat this
        // suite as a no-op rather than a hard fail.
        XCTAssertTrue(true, "WhisperKit not linked — parity test skipped at compile time")
    }

    #endif
}
