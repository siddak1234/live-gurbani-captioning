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
    /// The live engine isn't ready to drive the app yet — either the
    /// `.mlmodelc` set isn't bundled (M5.7c lands this), the WhisperKit
    /// framework isn't linked, or `prepare()` hasn't completed. The
    /// UI surfaces this distinct from `.engine(...)` so the
    /// fallback to `DemoCaptionSource` can be reasoned about. Pre-
    /// M5.7b this was the "transcribeStream is a placeholder" error;
    /// M5.7b replaced the placeholder with a real wire and broadened
    /// the case to cover bundle / load failures.
    case notWired
    /// The underlying engine failed for a *runtime* reason (mic
    /// permission denied, AVAudioSession setup, decoder error) —
    /// distinct from `.notWired` which means the engine never got
    /// off the ground.
    case engine(Error)

    public var errorDescription: String? {
        switch self {
        case .notWired:
            return """
            LiveCaptionSource: live engine not ready — the bundled \
            Core ML model is missing or WhisperKit didn't load. Until \
            M5.7c lands the .mlmodelc set under \
            Sources/GurbaniCaptioning/Resources/, DemoCaptionSource \
            drives the app (the AppEnvironment fallback handles this).
            """
        case .engine(let e):
            return "LiveCaptionSource: engine error — \(e.localizedDescription)"
        }
    }
}

@MainActor
public final class LiveCaptionSource: CaptionSource, AudioClipCapturing {

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
        } catch let error as CaptionEngineError {
            // Map engine "not ready" cases (model missing, SDK absent)
            // to the unified `.notWired` so callers can route through
            // the same fallback path. Real runtime failures stay
            // `.engine(...)`.
            if Self.isNotWiredError(error) {
                throw LiveCaptionSourceError.notWired
            }
            throw LiveCaptionSourceError.engine(error)
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
            // M5.7b: the placeholder is gone. `audioCaptureFailed`
            // now means a real runtime audio failure (mic denied,
            // AVAudioSession setup error); surface it as an engine
            // error. The "wiring isn't done yet" semantics moved to
            // `.modelLoadFailed` / `.modelFolderNotFound` /
            // `.whisperKitUnavailable` — translate those to
            // `.notWired` so the UI surface stays identical to
            // pre-M5.7b for those classes of failure.
            if Self.isNotWiredError(error) {
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

    /// Classify a `CaptionEngineError` as either "engine never got
    /// off the ground" (→ `.notWired`) or "real runtime audio /
    /// decoder failure" (→ `.engine(...)`). Single source of truth
    /// for both `prepare()` and `start()` so they can't drift.
    private static func isNotWiredError(_ error: CaptionEngineError) -> Bool {
        switch error {
        case .whisperKitUnavailable,
             .modelLoadFailed,
             .modelFolderNotFound:
            return true
        case .audioCaptureFailed:
            return false
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

    public private(set) var isPaused: Bool = false

    public func nudge(by delta: Int) {
        // M5.3: stub. Real nudge requires CaptionEngine to expose a
        // setLineIdx hook on the state machine. Once wired, this mirrors
        // DemoCaptionSource.nudge — clamp lower bound, construct a new
        // LineGuess, yield .guessUpdated.
        AppLogger.source.warning("LiveCaptionSource.nudge(\(delta, privacy: .public)) — not yet wired to CaptionEngine; ignored")
    }

    public func pause() {
        // M5.3: stub. Live engine pause needs to suspend WhisperKit's
        // audio buffer without tearing down the session. M5.7 wires the
        // real implementation alongside the engine swap.
        guard !isPaused else { return }
        isPaused = true
        AppLogger.source.warning("LiveCaptionSource.pause — not yet wired; flag set but engine continues processing")
    }

    public func resume() {
        guard isPaused else { return }
        isPaused = false
        AppLogger.source.warning("LiveCaptionSource.resume — not yet wired")
    }

    // MARK: - AudioClipCapturing (corrections feedback loop, Phase 2)

    /// Snapshot recent mic audio and persist it as a correction clip. Returns
    /// the local path, or nil if no audio is available. The `audioCaptureOptIn`
    /// consent gate is the caller's responsibility (see `AudioClipCapturing`).
    /// Live-engine only; verified end-to-end on device in Phase 2b.
    public func captureCorrectionClip(id: UUID, seconds: Double, writer: AudioClipWriter) -> String? {
        guard let samples = engine.snapshotRecentAudio(seconds: seconds), !samples.isEmpty else {
            return nil
        }
        do {
            let url = try writer.writeClip(id: id, samples: samples)
            AppLogger.sync.info("Captured correction clip (\(samples.count, privacy: .public) samples) → \(url.lastPathComponent, privacy: .public)")
            return url.path
        } catch {
            AppLogger.sync.error("captureCorrectionClip failed: \(error.localizedDescription, privacy: .public)")
            return nil
        }
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
