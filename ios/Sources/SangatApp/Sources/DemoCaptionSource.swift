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

    /// Single playback task. Holds either the scripted timeline (after
    /// `start()`) or the synthetic auto-advance loop (after
    /// `manuallyCommit`). Cancelled on `stop()` and replaced on every
    /// transition between the two modes.
    private var playbackTask: Task<Void, Never>?

    /// M5.6.x: `manuallyCommit` switches the source into "synthetic
    /// auto-advance" mode — the original scripted timeline can't tick
    /// through a shabad it has no line data for, so we drive a
    /// per-line cadence using `totalLinesProvider` for wrap behavior.
    /// `nil` (the default) means "no auto-advance after manualCommit"
    /// — Resume becomes a no-op in that scenario. Production passes
    /// a real provider; tests can stub it.
    private let totalLinesProvider: ((Int) -> Int)?

    /// Seconds between synthetic-mode line advances. Default 4s in
    /// production; tests pass a shorter interval to keep `swift test`
    /// snappy. Has no effect on the scripted timeline, which uses
    /// per-step delays from the `DemoScript`.
    private let syntheticAdvanceInterval: TimeInterval

    // MARK: - Init

    /// Build a demo source for the given script. Defaults to the canonical
    /// Tati Vao Na Lagai timeline so a no-arg construction "just works".
    ///
    /// `totalLinesProvider` and `syntheticAdvanceInterval` drive the
    /// post-`manuallyCommit` auto-advance behavior (M5.6.x). Both are
    /// optional and default to safe values so the 26 existing
    /// call sites compile unchanged.
    public init(
        script: DemoScript = .tatiVaoNaLagai,
        totalLinesProvider: ((Int) -> Int)? = nil,
        syntheticAdvanceInterval: TimeInterval = 4.0
    ) {
        self.script = script
        self.totalLinesProvider = totalLinesProvider
        self.syntheticAdvanceInterval = syntheticAdvanceInterval

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

        playbackTask = makeScriptedPlaybackTask()
    }

    /// Scripted timeline: walks `script.steps` once, applying each
    /// step's mutations after its `delay`. M5.6.x: between the
    /// per-step sleep and the apply, hold the loop while `isPaused`
    /// so pause/resume stops the clock instead of dropping steps.
    private func makeScriptedPlaybackTask() -> Task<Void, Never> {
        Task { @MainActor [weak self] in
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

                // Hold the step at the gate while paused. Canonical
                // "Pause" semantic is "stop the clock", not "drop any
                // step whose delay finishes during the pause window"
                // (which is what the M5.3-era code did, producing the
                // line 3 → line 5 jump on resume).
                while self.isPaused {
                    do {
                        try await Task.sleep(nanoseconds: 100_000_000) // 100ms poll
                    } catch {
                        return  // cancelled mid-pause
                    }
                    if Task.isCancelled { return }
                }
                if Task.isCancelled { return }
                self.apply(step: step)
            }
            // End of script — leave the source running (final committed state
            // visible) but mark the playback as drained.
            AppLogger.source.info("DemoCaptionSource script drained")
        }
    }

    /// Synthetic auto-advance: drives line-by-line progression within
    /// the *currently committed* shabad, wrapping at `totalLines`.
    /// Started by `manuallyCommit` (and only by `manuallyCommit`) —
    /// the scripted task can't help once the user has overridden the
    /// engine's pick, because the demo script has line data for only
    /// one pre-baked shabad. Honors the same pause gate as the
    /// scripted task.
    private func makeSyntheticPlaybackTask() -> Task<Void, Never> {
        Task { @MainActor [weak self] in
            guard let self else { return }
            // Snapshot the interval at task start (avoid touching
            // self on every loop iteration just for a constant).
            let nanos = UInt64(self.syntheticAdvanceInterval * 1_000_000_000)
            while !Task.isCancelled {
                do {
                    try await Task.sleep(nanoseconds: nanos)
                } catch {
                    return  // cancelled
                }
                if Task.isCancelled { return }

                while self.isPaused {
                    do {
                        try await Task.sleep(nanoseconds: 100_000_000)
                    } catch {
                        return
                    }
                    if Task.isCancelled { return }
                }
                if Task.isCancelled { return }
                self.advanceOneLineSynthetic()
            }
        }
    }

    /// Bump `currentGuess.lineIdx` by 1, wrapping at the provider's
    /// returned line count for the current shabad. No-op when there's
    /// no committed guess (e.g., `resetShabad` happened) or no
    /// provider was injected. Public state changes go through the
    /// continuation so views observing `events` see the update.
    private func advanceOneLineSynthetic() {
        guard case .committed = state, let guess = currentGuess else {
            return  // no shabad to advance within
        }
        let total: Int
        if let provider = totalLinesProvider {
            total = max(provider(guess.shabadId), 1)
        } else {
            // No provider → can't safely wrap. Hold position.
            return
        }
        let nextIdx = (guess.lineIdx + 1) % total
        let next = LineGuess(
            chunk: guess.chunk,
            shabadId: guess.shabadId,
            lineIdx: nextIdx,
            confidence: guess.confidence,
            isCommitted: guess.isCommitted
        )
        currentGuess = next
        continuation.yield(.guessUpdated(next))
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

        // Auto-pause first. The Sevadar's intent when manually
        // picking is "I'm taking control" — Resume on the dock is the
        // explicit "ok, follow this shabad now" signal.
        isPaused = true

        // M5.6.x: swap the scripted task for the synthetic auto-
        // advance task. The scripted task only knows the script's
        // pre-baked shabad; after a manual pick it would either
        // overwrite the user's selection (if it has remaining steps)
        // or sit silent (if it has drained). Neither is what the
        // dock's "Pause/Resume" affordance is supposed to mean. The
        // synthetic task ticks `currentGuess.lineIdx` by 1 every
        // `syntheticAdvanceInterval` seconds while not paused.
        playbackTask?.cancel()
        playbackTask = makeSyntheticPlaybackTask()

        AppLogger.source.info("DemoCaptionSource manuallyCommit to shabad #\(shabadId, privacy: .public) — engine auto-paused; synthetic auto-advance armed")
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
