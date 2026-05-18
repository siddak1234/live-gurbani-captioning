//
//  CaptionSourceModel.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Observable adapter over `CaptionSource`. Views never observe the source
//  directly — they observe this model. The model:
//
//    - subscribes to `source.events` and republishes them as observable
//      properties suitable for SwiftUI body re-renders;
//    - forwards lifecycle calls (`start`/`stop`/`reset`/`manuallyCommit`)
//      to the underlying source;
//    - exposes a `lastError` field for surfacing transient errors in the UI.
//
//  The `@Observable` macro (iOS 17+) generates Observation framework
//  conformance so SwiftUI re-renders depending views automatically.

import Foundation
import Observation
import GurbaniCaptioning

@MainActor
@Observable
public final class CaptionSourceModel {

    // MARK: - Published state

    public private(set) var state: ShabadState = .listening
    public private(set) var currentGuess: LineGuess?
    public private(set) var runnerUps: [LineGuess] = []
    public private(set) var isRunning: Bool = false
    /// Stored mirror of the source's pause flag. Stored (not computed)
    /// so the `@Observable` macro tracks reads and triggers view
    /// re-renders when `pause()` / `resume()` flip it. A computed
    /// pass-through to `source.isPaused` does NOT trigger Observation.
    public private(set) var isPaused: Bool = false
    /// Most recent non-fatal error message, or `nil`. Cleared on next event.
    public private(set) var lastError: String?

    // MARK: - Collaborators

    @ObservationIgnored
    public let source: any CaptionSource

    @ObservationIgnored
    private var subscriptionTask: Task<Void, Never>?

    // MARK: - Init

    public init(source: any CaptionSource) {
        self.source = source
        // Seed from initial state.
        self.state = source.state
        self.currentGuess = source.currentGuess
        self.runnerUps = source.runnerUps
        self.isRunning = source.isRunning
        self.isPaused = source.isPaused

        // Subscribe to events. We hold the task so we can cancel on deinit.
        let stream = source.events
        self.subscriptionTask = Task { @MainActor [weak self] in
            for await event in stream {
                guard let self else { return }
                self.apply(event: event)
            }
        }
    }

    deinit {
        subscriptionTask?.cancel()
    }

    // MARK: - Lifecycle forwarders

    public func prepare() async throws {
        try await source.prepare()
    }

    public func start() async throws {
        do {
            try await source.start()
        } catch {
            lastError = error.localizedDescription
            throw error
        }
    }

    public func stop() {
        source.stop()
    }

    public func resetShabad() {
        source.resetShabad()
    }

    public func manuallyCommit(shabadId: Int) {
        source.manuallyCommit(shabadId: shabadId)
    }

    public func nudge(by delta: Int) {
        source.nudge(by: delta)
    }

    public func pause() {
        source.pause()
        // Mirror the source flag onto the tracked stored property so
        // SwiftUI re-renders the dock's Pause/Resume label.
        isPaused = source.isPaused
    }

    public func resume() {
        source.resume()
        isPaused = source.isPaused
    }

    // MARK: - Convenience

    /// `true` when the engine has settled on a shabad.
    ///
    /// Reads from the model's own `state` (a tracked stored property)
    /// rather than from `source.isCommitted` (a computed pass-through
    /// SwiftUI's Observation framework cannot track). The pass-through
    /// version caused the Sevadar dock to not appear when state
    /// transitioned to `.committed` — re-renders only fired when some
    /// other observable property changed.
    public var isCommitted: Bool {
        switch state {
        case .committed: return true
        case .listening, .tentative: return false
        }
    }

    /// The committed shabad id, or `nil` if not committed.
    public var committedShabadId: Int? {
        if case let .committed(shabadId: sid) = state { return sid }
        return nil
    }

    // MARK: - Internal

    private func apply(event: CaptionSourceEvent) {
        switch event {
        case .stateChanged(let s):
            state = s
            lastError = nil
        case .guessUpdated(let g):
            currentGuess = g
        case .runnerUpsUpdated(let r):
            runnerUps = r
        case .error(let message):
            lastError = message
            AppLogger.source.error("CaptionSourceModel error: \(message, privacy: .public)")
        case .started:
            isRunning = true
        case .stopped:
            isRunning = false
        }
    }
}
