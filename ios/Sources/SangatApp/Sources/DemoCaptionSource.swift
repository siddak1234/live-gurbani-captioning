//
//  DemoCaptionSource.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Scripted `CaptionSource`. Runs the full app without WhisperKit, without
//  a microphone, without a Core ML model. Drives the design canvas, demo
//  recordings, App Store screenshots, and all SwiftUI Previews.
//
//  Threading
//  ---------
//  The class is `@MainActor` — all property reads/writes happen on the main
//  actor, so views observing `state` / `currentGuess` never race the
//  playback loop. The playback loop is a single child `Task` that inherits
//  main-actor isolation.
//
//  Lifecycle
//  ---------
//  `prepare()` is a no-op (matches the protocol; the live source needs it).
//  `start()` kicks off the playback task; `stop()` cancels it but leaves the
//  current state intact. Re-`start()` resumes from the beginning by default
//  (set `restartFromCurrent: false` to preserve position).

import Foundation
import GurbaniCaptioning

@MainActor
public final class DemoCaptionSource: CaptionSource {

    // MARK: - CaptionSource state

    public private(set) var state: ShabadState = .listening
    public private(set) var currentGuess: LineGuess?
    public private(set) var runnerUps: [LineGuess] = []
    public private(set) var isRunning: Bool = false
    public private(set) var isPaused: Bool = false

    public let events: AsyncStream<CaptionSourceEvent>
    private let continuation: AsyncStream<CaptionSourceEvent>.Continuation

    // MARK: - Configuration

    private let script: DemoScript
    private var playbackTask: Task<Void, Never>?

    // MARK: - Init

    /// Build a demo source for the given script. Defaults to the canonical
    /// Tati Vao Na Lagai timeline so a no-arg construction "just works".
    public init(script: DemoScript = .tatiVaoNaLagai) {
        self.script = script

        var localContinuation: AsyncStream<CaptionSourceEvent>.Continuation!
        self.events = AsyncStream { localContinuation = $0 }
        // Force-unwrap: the AsyncStream closure runs synchronously inside
        // `init`, so `localContinuation` is guaranteed assigned. This is the
        // Apple-blessed pattern for AsyncStream init outside of a function.
        self.continuation = localContinuation

        AppLogger.source.info("DemoCaptionSource initialized with script: \(script.name, privacy: .public)")
    }

    deinit {
        // Cancel any in-flight playback task. Task cancellation is
        // Sendable-safe so this is OK from `deinit`.
        playbackTask?.cancel()
        continuation.finish()
    }

    // MARK: - CaptionSource lifecycle

    public func prepare() async throws {
        // Demo source has nothing to prepare.
    }

    public func start() async throws {
        guard !isRunning else {
            AppLogger.source.debug("DemoCaptionSource.start() ignored — already running")
            return
        }

        // Reset session state. After the user taps the back chevron
        // (which calls `stop()`), state + currentGuess + isPaused are
        // preserved so a Sevadar can still see what they were reading.
        // But the next `start()` call is intent-to-begin-a-fresh-
        // session — we must clear stale committed state, otherwise
        // RootView jumps straight from the IdleView "Listen" button to
        // the previous shabad's reading view, skipping the listening
        // → tentative → committed transitions entirely. Locked in by
        // `testStartAfterStopResetsToListening`.
        if state != .listening || currentGuess != nil || isPaused {
            state = .listening
            currentGuess = nil
            runnerUps = []
            isPaused = false
            continuation.yield(.stateChanged(.listening))
            continuation.yield(.guessUpdated(nil))
            continuation.yield(.runnerUpsUpdated([]))
        }

        isRunning = true
        continuation.yield(.started)
        AppLogger.source.info("DemoCaptionSource started — fresh session")

        playbackTask = Task { @MainActor [weak self] in
            guard let self else { return }
            for step in self.script.steps {
                // Wait the requested delay, observing cancellation.
                let nanos = UInt64(step.delay * 1_000_000_000)
                do {
                    try await Task.sleep(nanoseconds: nanos)
                } catch {
                    return  // cancelled
                }
                if Task.isCancelled { return }
                self.apply(step: step)
            }
            // End of script — leave the source running (final committed state
            // visible) but mark the playback as drained.
            AppLogger.source.info("DemoCaptionSource script drained")
        }
    }

    public func stop() {
        playbackTask?.cancel()
        playbackTask = nil
        guard isRunning else { return }
        isRunning = false
        continuation.yield(.stopped)
        AppLogger.source.info("DemoCaptionSource stopped")
    }

    public func resetShabad() {
        state = .listening
        currentGuess = nil
        runnerUps = []
        continuation.yield(.stateChanged(.listening))
        continuation.yield(.guessUpdated(nil))
        continuation.yield(.runnerUpsUpdated([]))
        AppLogger.source.info("DemoCaptionSource resetShabad — back to .listening")
    }

    public func manuallyCommit(shabadId: Int) {
        let newState: ShabadState = .committed(shabadId: shabadId)
        state = newState
        continuation.yield(.stateChanged(newState))

        // Also reset currentGuess to point at the new shabad's first
        // line. Without this, manuallyCommit changes state but the
        // reading view keeps rendering the PREVIOUS shabad's lines —
        // the picker's "I just changed shabads" intent silently fails.
        let newGuess = LineGuess(
            chunk: AsrChunk(start: 0, end: 0, text: ""),
            shabadId: shabadId,
            lineIdx: 0,
            confidence: 100.0,
            isCommitted: true
        )
        currentGuess = newGuess
        continuation.yield(.guessUpdated(newGuess))

        // Runner-ups from the previous shabad are no longer relevant.
        runnerUps = []
        continuation.yield(.runnerUpsUpdated([]))

        // Auto-pause the scripted engine. The Sevadar's intent when
        // manually picking a different shabad is "I'm taking control"
        // — without this, the playback task keeps running and the
        // next scripted `guessUpdated` event overwrites the manual
        // selection a few seconds later (the screen flips back to
        // Tati Vao even though the user picked Hum Aadmi). Sevadar
        // hits Resume on the dock to let auto-detection drive again.
        isPaused = true

        AppLogger.source.info("DemoCaptionSource manuallyCommit to shabad #\(shabadId, privacy: .public) — engine auto-paused")
    }

    public func nudge(by delta: Int) {
        guard case .committed = state, let guess = currentGuess else {
            AppLogger.source.debug("DemoCaptionSource.nudge ignored — not in .committed with a guess")
            return
        }
        let newIdx = max(0, guess.lineIdx + delta)
        guard newIdx != guess.lineIdx else { return }
        let updated = LineGuess(
            chunk: guess.chunk,
            shabadId: guess.shabadId,
            lineIdx: newIdx,
            confidence: guess.confidence,
            isCommitted: guess.isCommitted
        )
        currentGuess = updated
        continuation.yield(.guessUpdated(updated))
        AppLogger.source.info("DemoCaptionSource nudge \(delta, privacy: .public) → lineIdx=\(newIdx, privacy: .public)")
    }

    public func pause() {
        guard !isPaused else { return }
        isPaused = true
        AppLogger.source.info("DemoCaptionSource paused — engine emissions gated")
    }

    public func resume() {
        guard isPaused else { return }
        isPaused = false
        AppLogger.source.info("DemoCaptionSource resumed")
    }

    /// Test-only seam for unit tests that need to drive the source into
    /// a known guess without playing the full script. Not exposed
    /// publicly — only callers inside the SangatApp module (including
    /// the test target via @testable import) can reach this. Don't call
    /// from production view code.
    @_spi(Testing)
    public func _testInject(guess: LineGuess) {
        currentGuess = guess
    }

    // MARK: - Internal

    private func apply(step: DemoStep) {
        // While paused, scripted engine emissions are suppressed so manual
        // nudges from the Sevadar dock stay put. The playback loop keeps
        // ticking through delays — on resume, the next scripted step will
        // apply, matching the live engine's "audio kept coming but UI was
        // gated" model.
        guard !isPaused else { return }

        if let newState = step.state {
            state = newState
            continuation.yield(.stateChanged(newState))
        }
        if step.clearsGuess {
            currentGuess = nil
            continuation.yield(.guessUpdated(nil))
        } else if let newGuess = step.guess {
            currentGuess = newGuess
            continuation.yield(.guessUpdated(newGuess))
        }
        if let newRunnerUps = step.runnerUps {
            runnerUps = newRunnerUps
            continuation.yield(.runnerUpsUpdated(newRunnerUps))
        }
    }
}
