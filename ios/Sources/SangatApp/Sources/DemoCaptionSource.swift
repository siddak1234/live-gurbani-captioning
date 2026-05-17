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
        isRunning = true
        continuation.yield(.started)
        AppLogger.source.info("DemoCaptionSource started")

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
        AppLogger.source.info("DemoCaptionSource manuallyCommit to shabad #\(shabadId, privacy: .public)")
    }

    // MARK: - Internal

    private func apply(step: DemoStep) {
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
