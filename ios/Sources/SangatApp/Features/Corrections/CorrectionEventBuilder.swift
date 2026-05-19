//
//  CorrectionEventBuilder.swift
//  GurbaniCaptioningApp · SangatApp · Features/Corrections
//
//  Created for the Sangat iOS app, M5.6 (Correction loop surfaces).
//
//  Pure factory that assembles a `CorrectionEvent` from the live engine
//  state at the moment of correction. Keeps the per-kind construction
//  rules in one auditable place so RootView's emission sites stay short
//  and the contract is covered by unit tests without touching SwiftUI.
//
//  Each `make<Kind>(...)` builds a single kind; callers don't pick the
//  kind out of an enum, the function name does.
//
//  All factories take an explicit `now: Date` and `id: UUID` (with
//  sensible defaults) so tests can assert exact shape without injecting
//  clocks at the framework level.

import Foundation
import GurbaniCaptioning

/// Pure assembly of `CorrectionEvent` values from engine state. Used by
/// RootView and the correction sheets to keep emission logic uniform.
///
/// Why a struct of `static func`s rather than `CorrectionEvent.makeFoo`:
/// keeping the factories on a separate type makes the dependency
/// direction explicit — `CorrectionEvent` knows nothing about app-level
/// state (`CaptionSourceModel`, `LineGuess`, etc.), while the builder
/// does.
public enum CorrectionEventBuilder {

    /// Hard-negative + hard-positive: engine committed shabad A, user
    /// explicitly picked shabad B. Highest-value training signal.
    ///
    /// `predictedSnapshot` is built from the *pre-correction* engine
    /// state — callers must capture `currentGuess` / `committedShabadId`
    /// BEFORE calling `captionModel.manuallyCommit(shabadId:)`.
    public static func makeHardNegPos(
        sessionId: UUID,
        predictedShabadId: Int,
        predictedLineIdx: Int?,
        predictedConfidence: Double?,
        runnerUps: [Int: Double],
        correctedShabadId: Int,
        engineStateRaw: String,
        recentChunks: [CorrectionEvent.ChunkSnapshot] = [],
        notes: String? = nil,
        now: Date = Date(),
        id: UUID = UUID()
    ) -> CorrectionEvent {
        CorrectionEvent(
            id: id,
            timestamp: now,
            sessionId: sessionId,
            kind: .hardNegPos,
            predicted: .init(
                shabadId: predictedShabadId,
                lineIdx: predictedLineIdx,
                confidence: predictedConfidence,
                runnerUps: runnerUps
            ),
            groundTruth: .init(shabadId: correctedShabadId, lineIdx: nil),
            engineStateRaw: engineStateRaw,
            recentChunks: recentChunks,
            audioBufferPath: nil,
            audioStart: recentChunks.first?.start ?? 0,
            audioEnd: recentChunks.last?.end ?? 0,
            notes: notes
        )
    }

    /// User picked from displayed runner-up candidates *before* the
    /// engine committed. Pre-emptive correction — engine had A in
    /// view but user picked B from the visible list.
    public static func makeRunnerUpEndorsed(
        sessionId: UUID,
        predictedShabadId: Int,
        predictedConfidence: Double?,
        runnerUps: [Int: Double],
        endorsedShabadId: Int,
        engineStateRaw: String,
        recentChunks: [CorrectionEvent.ChunkSnapshot] = [],
        now: Date = Date(),
        id: UUID = UUID()
    ) -> CorrectionEvent {
        CorrectionEvent(
            id: id,
            timestamp: now,
            sessionId: sessionId,
            kind: .runnerUpEndorsed,
            predicted: .init(
                shabadId: predictedShabadId,
                lineIdx: nil,
                confidence: predictedConfidence,
                runnerUps: runnerUps
            ),
            groundTruth: .init(shabadId: endorsedShabadId, lineIdx: nil),
            engineStateRaw: engineStateRaw,
            recentChunks: recentChunks,
            audioBufferPath: nil,
            audioStart: recentChunks.first?.start ?? 0,
            audioEnd: recentChunks.last?.end ?? 0,
            notes: nil
        )
    }

    /// Within a committed shabad, user nudged the displayed line by
    /// `delta` (typically ±1). Feeds the smoother / loop-aligner.
    public static func makeLineNudge(
        sessionId: UUID,
        shabadId: Int,
        predictedLineIdx: Int,
        delta: Int,
        engineStateRaw: String,
        now: Date = Date(),
        id: UUID = UUID()
    ) -> CorrectionEvent {
        CorrectionEvent(
            id: id,
            timestamp: now,
            sessionId: sessionId,
            kind: .lineNudge,
            predicted: .init(
                shabadId: shabadId,
                lineIdx: predictedLineIdx,
                confidence: nil,
                runnerUps: [:]
            ),
            groundTruth: .init(shabadId: shabadId, lineIdx: predictedLineIdx + delta),
            engineStateRaw: engineStateRaw,
            recentChunks: [],
            audioBufferPath: nil,
            audioStart: 0,
            audioEnd: 0,
            notes: nil
        )
    }

    /// User flagged a past session-history entry as wrong, supplying
    /// the correct shabad. Lowest-confidence kind because the user
    /// didn't catch the miss live.
    public static func makeRetroactive(
        sessionId: UUID,
        flaggedShabadId: Int,
        correctedShabadId: Int,
        flaggedAt: Date,
        engineStateRaw: String,
        notes: String? = nil,
        now: Date = Date(),
        id: UUID = UUID()
    ) -> CorrectionEvent {
        CorrectionEvent(
            id: id,
            timestamp: now,
            sessionId: sessionId,
            kind: .retroactive,
            predicted: .init(
                shabadId: flaggedShabadId,
                lineIdx: nil,
                confidence: nil,
                runnerUps: [:]
            ),
            groundTruth: .init(shabadId: correctedShabadId, lineIdx: nil),
            engineStateRaw: engineStateRaw,
            recentChunks: [],
            audioBufferPath: nil,
            audioStart: flaggedAt.timeIntervalSince1970,
            audioEnd: flaggedAt.timeIntervalSince1970,
            notes: notes
        )
    }

    /// Stringify a `ShabadState` for the `engineStateRaw` field. Kept
    /// here so emission sites can pass it without rebuilding the switch.
    public static func engineStateRaw(_ state: ShabadState) -> String {
        switch state {
        case .listening:                  return "listening"
        case .tentative(let id):          return "tentative(\(id))"
        case .committed(let id):          return "committed(\(id))"
        }
    }
}
