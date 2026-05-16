//  CaptionEngine.swift
//
//  Top-level orchestrator: WhisperKit → AsrChunk → ShabadStateMachine →
//  CaptionSegment → UI. Mirrors the Python ``StreamingEngine`` contract
//  (reset/start/stop/observe) but with iOS-native types.
//
//  Threading model
//  ---------------
//  WhisperKit runs inference on a dedicated background actor. The state
//  machine is owned by this class and mutated from one dispatch queue
//  (``processingQueue``) to keep it simple. Delegate callbacks are dispatched
//  on the main queue for SwiftUI binding.
//
//  WhisperKit-version compatibility
//  --------------------------------
//  WhisperKit's streaming API has evolved across releases. The call into the
//  framework lives in one method (``transcribeStream``) so future SDK bumps
//  only touch that. If WhisperKit's signature differs from what's coded
//  here, that's the file to update.

import Foundation
#if canImport(WhisperKit)
import WhisperKit
#endif

public protocol CaptionEngineDelegate: AnyObject {
    func captionEngine(_ engine: CaptionEngine, didUpdate guess: LineGuess?)
    func captionEngine(_ engine: CaptionEngine, didChangeState state: ShabadState)
    func captionEngine(_ engine: CaptionEngine, didEncounterError error: Error)
}

public enum CaptionEngineError: Error, LocalizedError {
    case whisperKitUnavailable
    case modelLoadFailed(String)
    case audioCaptureFailed(String)

    public var errorDescription: String? {
        switch self {
        case .whisperKitUnavailable:
            return "CaptionEngine: WhisperKit framework not linked. Add the SPM dependency."
        case .modelLoadFailed(let s):  return "CaptionEngine: model load failed — \(s)"
        case .audioCaptureFailed(let s): return "CaptionEngine: audio capture failed — \(s)"
        }
    }
}

public final class CaptionEngine {

    // Public surface
    public weak var delegate: CaptionEngineDelegate?
    public private(set) var isRunning: Bool = false
    public private(set) var stateMachine: ShabadStateMachine

    // Configuration
    public struct Config {
        /// Local path or HF model id of the Core ML .mlpackage WhisperKit will load.
        /// In production this is your fine-tuned surt converted by scripts/export_coreml.py.
        public let modelPath: String
        /// Whisper generation language tag. Surt is Punjabi-trained.
        public let language: String
        /// Streaming chunk size in seconds (how often the state machine sees a new ASR chunk).
        public let chunkSeconds: TimeInterval

        public init(modelPath: String,
                    language: String = "punjabi",
                    chunkSeconds: TimeInterval = 5.0) {
            self.modelPath = modelPath
            self.language = language
            self.chunkSeconds = chunkSeconds
        }
    }

    private let config: Config
    private let corpus: ShabadCorpus

#if canImport(WhisperKit)
    private var whisper: WhisperKit?
#endif

    public init(config: Config, corpus: ShabadCorpus,
                commitConfig: ShabadCommitConfig = .default) {
        self.config = config
        self.corpus = corpus
        self.stateMachine = ShabadStateMachine(corpus: corpus, config: commitConfig)
    }

    // MARK: - Lifecycle

    /// Load the Core ML model and prepare the audio pipeline.
    public func prepare() async throws {
#if canImport(WhisperKit)
        do {
            // WhisperKit's initializer differs slightly across versions; the
            // common shape takes a model name or absolute path. Adjust here
            // if the SDK signature changes upstream.
            self.whisper = try await WhisperKit(model: config.modelPath)
        } catch {
            throw CaptionEngineError.modelLoadFailed(String(describing: error))
        }
#else
        throw CaptionEngineError.whisperKitUnavailable
#endif
    }

    /// Begin streaming transcription from the microphone.
    public func start() async throws {
#if canImport(WhisperKit)
        guard let whisper else {
            throw CaptionEngineError.modelLoadFailed("call prepare() first")
        }
        isRunning = true
        await notifyState(stateMachine.state)
        try await transcribeStream(using: whisper)
#else
        throw CaptionEngineError.whisperKitUnavailable
#endif
    }

    /// Stop transcription. Safe to call multiple times.
    public func stop() {
        isRunning = false
#if canImport(WhisperKit)
        // WhisperKit-specific cancellation hook — depends on SDK version.
        // Latest versions expose audioProcessor.stopRecording() or similar.
#endif
    }

    /// Sewadar pressed "Reset shabad" — drop the committed shabad and re-listen.
    public func resetShabad() {
        stateMachine.reset()
        Task { await notifyState(stateMachine.state) }
    }

    /// Manually commit a shabad (e.g. Sewadar picked one from a list).
    public func manuallyCommit(shabadId: Int) {
        stateMachine.commit(shabadId: shabadId)
        Task { await notifyState(stateMachine.state) }
    }

    // MARK: - Inference loop

#if canImport(WhisperKit)
    private func transcribeStream(using whisper: WhisperKit) async throws {
        // The actual WhisperKit streaming API varies; current published surface
        // includes whisperKit.transcribe(audioPath:) for files and an async
        // streaming path via AudioProcessor + bufferCallback. The block below
        // is a structural placeholder that mirrors what the production hookup
        // looks like — adapt to the installed WhisperKit version's signature.
        //
        // Pseudocode:
        //
        //   for await chunk in whisper.streamingChunks(language: config.language,
        //                                              chunkSeconds: config.chunkSeconds) {
        //       let asr = AsrChunk(start: chunk.start, end: chunk.end, text: chunk.text)
        //       let guess = stateMachine.processChunk(asr)
        //       await notifyGuess(guess)
        //       await notifyState(stateMachine.state)
        //   }
        //
        // Until that's wired against the installed WhisperKit version, this
        // throws so calls go to the error delegate rather than silently no-op.
        throw CaptionEngineError.audioCaptureFailed(
            "transcribeStream is not yet wired to WhisperKit's streaming API. See comments inside CaptionEngine.swift."
        )
    }
#endif

    // MARK: - Delegate dispatch (main thread)

    @MainActor
    private func notifyGuess(_ guess: LineGuess?) {
        delegate?.captionEngine(self, didUpdate: guess)
    }

    @MainActor
    private func notifyState(_ state: ShabadState) {
        delegate?.captionEngine(self, didChangeState: state)
    }

    @MainActor
    private func notifyError(_ error: Error) {
        delegate?.captionEngine(self, didEncounterError: error)
    }
}
