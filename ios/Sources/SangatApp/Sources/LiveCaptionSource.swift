//
//  LiveCaptionSource.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Real `CaptionSource` backed by `CaptionEngine` + `WhisperKit`. Skeleton
//  in M5.1 — `start()` throws `.notWired` until the Core ML model is
//  exported from the training Mac and `CaptionEngine.transcribeStream` is
//  filled in. The protocol surface is complete so any UI built against
//  `CaptionSource` will Just Work once the engine wire is in place.
//
//  This file is intentionally small. All real work is in `CaptionEngine.swift`
//  (library); this wrapper translates between the library's delegate-based
//  callback surface and the `CaptionSource` AsyncStream + properties.

import Foundation
import GurbaniCaptioning

/// Error specific to `LiveCaptionSource` wiring.
public enum LiveCaptionSourceError: Error, LocalizedError {
    /// `CaptionEngine.transcribeStream` is still the placeholder from
    /// `docs/ios_deployment.md` Step 5. Wire the WhisperKit streaming API
    /// for the installed SDK version, then this error goes away.
    case notWired
    /// The underlying engine failed.
    case engine(Error)

    public var errorDescription: String? {
        switch self {
        case .notWired:
            return """
            LiveCaptionSource: WhisperKit streaming is not yet wired. Fill in \
            CaptionEngine.transcribeStream against the installed WhisperKit \
            version (see docs/ios_deployment.md Step 5), then this error goes \
            away. Until then, use DemoCaptionSource for app development.
            """
        case .engine(let e):
            return "LiveCaptionSource: engine error — \(e.localizedDescription)"
        }
    }
}

@MainActor
public final class LiveCaptionSource: CaptionSource {

    // MARK: - CaptionSource state

    public private(set) var state: ShabadState = .listening
    public private(set) var currentGuess: LineGuess?
    public private(set) var runnerUps: [LineGuess] = []
    public private(set) var isRunning: Bool = false

    public let events: AsyncStream<CaptionSourceEvent>
    private let continuation: AsyncStream<CaptionSourceEvent>.Continuation

    // MARK: - Collaborators

    private let engine: CaptionEngine
    private let corpus: ShabadCorpus
    private var delegateBridge: CaptionEngineBridge?

    // MARK: - Init

    /// Construct the live source over a corpus + engine config.
    ///
    /// The caller (typically `AppEnvironment.production`) is responsible for
    /// loading the corpus from the bundle (`ShabadCorpus.loadFromBundle()`).
    /// We do not load it here so test code can pass in a smaller corpus.
    public init(corpus: ShabadCorpus, config: CaptionEngine.Config) {
        self.corpus = corpus
        self.engine = CaptionEngine(config: config, corpus: corpus)
        self.state = engine.stateMachine.state

        var localContinuation: AsyncStream<CaptionSourceEvent>.Continuation!
        self.events = AsyncStream { localContinuation = $0 }
        self.continuation = localContinuation

        // Bridge the delegate callbacks onto the main actor so we can write
        // our @MainActor-isolated properties.
        let bridge = CaptionEngineBridge()
        bridge.source = self
        self.delegateBridge = bridge
        engine.delegate = bridge

        AppLogger.source.info("LiveCaptionSource initialized for model: \(config.modelPath, privacy: .public)")
    }

    deinit {
        continuation.finish()
    }

    // MARK: - CaptionSource lifecycle

    public func prepare() async throws {
        do {
            try await engine.prepare()
        } catch {
            throw LiveCaptionSourceError.engine(error)
        }
    }

    public func start() async throws {
        guard !isRunning else { return }
        do {
            try await engine.start()
            isRunning = true
            continuation.yield(.started)
        } catch let error as CaptionEngineError {
            // Re-shape the engine's not-wired placeholder error into our
            // domain-specific one so the UI can match on it.
            if case .audioCaptureFailed = error {
                continuation.yield(.error(LiveCaptionSourceError.notWired.localizedDescription))
                throw LiveCaptionSourceError.notWired
            }
            continuation.yield(.error(error.localizedDescription))
            throw LiveCaptionSourceError.engine(error)
        } catch {
            continuation.yield(.error(error.localizedDescription))
            throw LiveCaptionSourceError.engine(error)
        }
    }

    public func stop() {
        engine.stop()
        guard isRunning else { return }
        isRunning = false
        continuation.yield(.stopped)
    }

    public func resetShabad() {
        engine.resetShabad()
        currentGuess = nil
        runnerUps = []
        continuation.yield(.guessUpdated(nil))
        continuation.yield(.runnerUpsUpdated([]))
    }

    public func manuallyCommit(shabadId: Int) {
        engine.manuallyCommit(shabadId: shabadId)
    }

    // MARK: - Internal — receive delegate callbacks

    fileprivate func handleGuess(_ guess: LineGuess?) {
        currentGuess = guess
        continuation.yield(.guessUpdated(guess))
    }

    fileprivate func handleState(_ newState: ShabadState) {
        state = newState
        continuation.yield(.stateChanged(newState))
    }

    fileprivate func handleError(_ error: Error) {
        continuation.yield(.error(error.localizedDescription))
    }
}

// MARK: - Engine delegate bridge

/// Forwards `CaptionEngineDelegate` callbacks (which may arrive on any
/// thread depending on `WhisperKit`'s threading model) onto the main actor
/// and into the `LiveCaptionSource` instance.
private final class CaptionEngineBridge: CaptionEngineDelegate {
    weak var source: LiveCaptionSource?

    func captionEngine(_ engine: CaptionEngine, didUpdate guess: LineGuess?) {
        Task { @MainActor [weak source] in
            source?.handleGuess(guess)
        }
    }

    func captionEngine(_ engine: CaptionEngine, didChangeState state: ShabadState) {
        Task { @MainActor [weak source] in
            source?.handleState(state)
        }
    }

    func captionEngine(_ engine: CaptionEngine, didEncounterError error: Error) {
        Task { @MainActor [weak source] in
            source?.handleError(error)
        }
    }
}
