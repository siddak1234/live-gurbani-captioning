//
//  CaptionSource.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  THE seam between UI and engine. Every screen binds to a `CaptionSource`,
//  never to `CaptionEngine` directly. This lets us:
//
//    - Run the entire app without WhisperKit or a Core ML model
//      (`DemoCaptionSource` scripts a timeline).
//    - Drop in fixtures for parity testing (`ReplayCaptionSource`, future).
//    - Swap real implementations later without touching any view.
//
//  Why a protocol + AsyncSequence rather than `@Observable`
//  -------------------------------------------------------
//  The Observation framework's `@Observable` macro can't be applied to a
//  protocol; it requires a concrete class declaration. To preserve
//  substitutability we instead expose:
//
//    - Synchronous `var` getters for "current value" reads. These are safe
//      because the source is `@MainActor` — there are no other isolation
//      domains a view could read from.
//    - An `AsyncStream` of `CaptionSourceEvent` values so the UI can
//      subscribe to changes. The `CaptionSourceModel` adapter (next file)
//      wraps the stream in an `@Observable` class for SwiftUI binding.
//
//  This is the standard "service + observable adapter" pattern, gives us
//  protocol-based dependency injection AND a SwiftUI-friendly observable
//  surface.

import Foundation
import GurbaniCaptioning

/// The contract every caption source implements.
///
/// Conforming types:
///   - `LiveCaptionSource` — wraps the real `CaptionEngine` + `WhisperKit`.
///   - `DemoCaptionSource` — scripted timeline for development & demos.
///   - `ReplayCaptionSource` (future) — plays back saved audio/transcripts
///     for parity testing.
///
/// Conformers must be reference types (`AnyObject`) because they hold
/// long-lived state (the state machine, audio session, etc.). They are
/// constrained to `@MainActor` so view reads of `state`/`currentGuess`
/// are isolation-safe.
@MainActor
public protocol CaptionSource: AnyObject {

    // MARK: - Synchronous "current value" reads

    /// Current engine state. Use this for immediate reads (e.g. on view
    /// appear); subscribe to `events` for live updates.
    var state: ShabadState { get }

    /// Most recent `LineGuess` emitted by the state machine, or `nil` if
    /// the source has not produced one yet (idle / very early listening).
    var currentGuess: LineGuess? { get }

    /// Runner-up shabad candidates from the most recent chunk vote, ordered
    /// high-to-low score. Empty when committed (we only score one shabad)
    /// or when no candidates met threshold.
    ///
    /// The Sevadar confidence view and the Sangat "wrong shabad" sheet
    /// both read this to surface alternative picks.
    var runnerUps: [LineGuess] { get }

    /// Whether the source is currently running (capturing audio / streaming
    /// scripted events).
    var isRunning: Bool { get }

    // MARK: - Event stream

    /// Stream of `CaptionSourceEvent` values. Views subscribe via
    /// `CaptionSourceModel` (the `@Observable` adapter) which republishes
    /// state into SwiftUI-friendly published properties.
    ///
    /// One stream per source instance; iterating it multiple times is
    /// undefined behavior — use the model adapter.
    var events: AsyncStream<CaptionSourceEvent> { get }

    // MARK: - Lifecycle

    /// Prepare any resources (model load, audio session config) without
    /// starting capture. May be expensive (~30s for first-run Core ML
    /// compile in `LiveCaptionSource`).
    func prepare() async throws

    /// Begin emitting events. Idempotent — calling on an already-running
    /// source is a no-op.
    func start() async throws

    /// Stop emitting events. The stream is *not* terminated — call again
    /// without re-creating the source.
    func stop()

    // MARK: - Commands

    /// Drop the committed shabad and re-listen. Maps to the Sevadar "Reset"
    /// affordance.
    func resetShabad()

    /// Override blind detection — Sevadar selected a shabad manually, or
    /// the Sangat corrected a wrong commit. Emits a `.stateChanged(.committed)`
    /// event downstream.
    ///
    /// This is also the entry point through which the correction loop
    /// triggers in M5.6 — the call site decides whether to record a
    /// `CorrectionEvent`; the source itself remains correction-agnostic.
    func manuallyCommit(shabadId: Int)
}

// MARK: - Convenience

extension CaptionSource {

    /// `true` when the engine has settled on a shabad. Read it in views to
    /// branch reading-mode UI on commit/no-commit.
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
}
