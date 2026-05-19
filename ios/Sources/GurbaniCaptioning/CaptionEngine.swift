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
    case modelFolderNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .whisperKitUnavailable:
            return "CaptionEngine: WhisperKit framework not linked. Add the SPM dependency."
        case .modelLoadFailed(let s):  return "CaptionEngine: model load failed — \(s)"
        case .audioCaptureFailed(let s): return "CaptionEngine: audio capture failed — \(s)"
        case .modelFolderNotFound(let s): return "CaptionEngine: model folder not found — \(s)"
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
    private var streamTranscriber: AudioStreamTranscriber?
    /// Count of `confirmedSegments` we've already routed into the state
    /// machine. WhisperKit's stream callback fires on every state mutation;
    /// we use this to dedupe and only process newly-confirmed segments.
    private var processedConfirmedSegmentCount: Int = 0
#endif

    public init(config: Config, corpus: ShabadCorpus,
                commitConfig: ShabadCommitConfig = .default) {
        self.config = config
        self.corpus = corpus
        self.stateMachine = ShabadStateMachine(corpus: corpus, config: commitConfig)
    }

    // MARK: - Lifecycle

    /// Load the Core ML model and prepare the audio pipeline.
    ///
    /// Resolution order for `config.modelPath`:
    ///   1. If it's an absolute path that exists on disk, use it directly
    ///      (dev path — sideload models without re-bundling).
    ///   2. Otherwise, look it up via `Bundle.module` as a resource folder
    ///      name (production path — model ships in the SPM bundle once
    ///      M5.7c lands the `.copy("Resources/...")` declaration).
    ///   3. Otherwise, throw `.modelFolderNotFound` — `LiveCaptionSource`
    ///      catches this and `AppEnvironment` falls back to the demo
    ///      source, so the app remains usable.
    public func prepare() async throws {
#if canImport(WhisperKit)
        let folder = try resolveModelFolder(name: config.modelPath)
        do {
            // WhisperKitConfig parameter order is locked by the init
            // signature (prewarm → load → download). The comments below
            // explain each choice; the order is mechanical.
            let kitConfig = WhisperKitConfig(
                modelFolder: folder.path,
                // Specialize each model individually to minimize peak
                // memory during first launch. Argmax recommends prewarm
                // on for mobile apps where peak RAM matters more than
                // load-time latency (~1-2s tax on cold start).
                prewarm: true,
                // Auto-load once specialization is done.
                load: true,
                // Don't allow runtime download of remote weights. The
                // production deliverable is the bundled .mlmodelc set;
                // if it's missing, the demo source path takes over
                // rather than silently pulling 220MB over cellular.
                download: false
            )
            self.whisper = try await WhisperKit(kitConfig)
        } catch let e as CaptionEngineError {
            throw e
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
        guard let tokenizer = whisper.tokenizer else {
            throw CaptionEngineError.modelLoadFailed(
                "WhisperKit tokenizer not loaded; prepare() must complete before start()"
            )
        }

        // Reset per-session state so a second start() on the same
        // engine doesn't replay old confirmed segments.
        processedConfirmedSegmentCount = 0
        stateMachine.reset()

        let decodingOptions = DecodingOptions(
            verbose: false,
            task: .transcribe,
            language: config.language,
            // Skip the special-token text in the streamed transcript so
            // the matcher sees clean Gurmukhi (Whisper's <|punjabi|>
            // / <|transcribe|> tokens would otherwise leak through).
            skipSpecialTokens: true,
            withoutTimestamps: false,
            // VAD is on by default in AudioStreamTranscriber; we let it
            // gate the decoder so we don't burn ANE on silence.
            noSpeechThreshold: 0.6
        )

        let transcriber = AudioStreamTranscriber(
            audioEncoder: whisper.audioEncoder,
            featureExtractor: whisper.featureExtractor,
            segmentSeeker: whisper.segmentSeeker,
            textDecoder: whisper.textDecoder,
            tokenizer: tokenizer,
            audioProcessor: whisper.audioProcessor,
            decodingOptions: decodingOptions,
            useVAD: true,
            stateChangeCallback: { [weak self] oldState, newState in
                guard let self else { return }
                // WhisperKit fires this on every state mutation. We only
                // care about *newly-confirmed* segments (the unconfirmed
                // bucket churns as the decoder revises tail text).
                let oldCount = oldState.confirmedSegments.count
                let newCount = newState.confirmedSegments.count
                guard newCount > oldCount else { return }
                let newSegments = Array(newState.confirmedSegments.suffix(newCount - oldCount))
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.ingestConfirmedSegments(newSegments)
                }
            }
        )
        self.streamTranscriber = transcriber
        self.isRunning = true
        await notifyState(stateMachine.state)

        do {
            try await transcriber.startStreamTranscription()
        } catch {
            self.isRunning = false
            self.streamTranscriber = nil
            await notifyError(CaptionEngineError.audioCaptureFailed(String(describing: error)))
            throw CaptionEngineError.audioCaptureFailed(String(describing: error))
        }
#else
        throw CaptionEngineError.whisperKitUnavailable
#endif
    }

    /// Stop transcription. Safe to call multiple times.
    public func stop() {
        isRunning = false
#if canImport(WhisperKit)
        if let transcriber = streamTranscriber {
            // `stopStreamTranscription` is an actor method; we don't need
            // its result and the underlying audio session can be torn
            // down asynchronously.
            Task { await transcriber.stopStreamTranscription() }
            streamTranscriber = nil
        }
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

    // MARK: - Model folder resolution

#if canImport(WhisperKit)
    /// Locate the `.mlmodelc` set that WhisperKit will load from.
    /// `config.modelPath` may be:
    ///   - an absolute filesystem path (dev / sideload case), or
    ///   - a resource folder name inside `Bundle.module` (production case
    ///     once M5.7c lands the `.copy("Resources/...")` declaration).
    /// Throws `.modelFolderNotFound` if neither resolves. Callers
    /// (`LiveCaptionSource`) translate this into a graceful fallback
    /// rather than a crash.
    private func resolveModelFolder(name: String) throws -> URL {
        let fm = FileManager.default
        // (1) Absolute path
        if name.hasPrefix("/") || name.hasPrefix("~") {
            let expanded = (name as NSString).expandingTildeInPath
            var isDir: ObjCBool = false
            if fm.fileExists(atPath: expanded, isDirectory: &isDir), isDir.boolValue {
                return URL(fileURLWithPath: expanded, isDirectory: true)
            }
        }
        // (2) Bundle.module resource folder
        if let bundleURL = Bundle.module.url(forResource: name, withExtension: nil) {
            var isDir: ObjCBool = false
            if fm.fileExists(atPath: bundleURL.path, isDirectory: &isDir), isDir.boolValue {
                return bundleURL
            }
        }
        throw CaptionEngineError.modelFolderNotFound(
            "Could not find Core ML model folder \"\(name)\". " +
            "Expected either an absolute path or a resource bundled via " +
            "`.copy(\"Resources/<name>\")` in Package.swift (lands in M5.7c). " +
            "Until then, `LiveCaptionSource` will throw and `AppEnvironment` " +
            "falls back to `DemoCaptionSource`."
        )
    }

    // MARK: - Segment ingestion

    /// Route newly-confirmed WhisperKit segments through the matcher /
    /// state machine and notify the delegate of the resulting guess.
    /// Called from the stream callback on the MainActor so we don't
    /// race with view code reading `stateMachine.state`.
    @MainActor
    private func ingestConfirmedSegments(_ segments: [TranscriptionSegment]) {
        guard !segments.isEmpty else { return }
        for seg in segments {
            let chunk = AsrChunk(
                start: TimeInterval(seg.start),
                end: TimeInterval(seg.end),
                text: seg.text
            )
            let guess = stateMachine.processChunk(chunk)
            delegate?.captionEngine(self, didUpdate: guess)
            delegate?.captionEngine(self, didChangeState: stateMachine.state)
            processedConfirmedSegmentCount += 1
        }
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
