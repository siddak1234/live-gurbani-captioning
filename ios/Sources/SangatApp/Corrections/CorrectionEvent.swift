//
//  CorrectionEvent.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Value type written to the correction log on every user-triggered
//  correction. Shape is deliberately rich because the off-device fine-tune
//  pipeline reads these directly — every field is one we'll want during
//  training set construction (see `docs/ios_app_architecture.md`).
//
//  Privacy
//  -------
//  `audioBufferPath` is optional and only populated when the user opts in
//  (Settings → Help improve detection). The README's "audio never leaves
//  the phone" guarantee is preserved even when the field is set — the path
//  is local-only. Network sync is a separate decision behind a separate
//  toggle and is *not* part of M5.6.

import Foundation
import GurbaniCaptioning

/// A single user-triggered correction.
public struct CorrectionEvent: Codable, Sendable, Identifiable, Equatable {

    public let id: UUID
    public let timestamp: Date
    public let sessionId: UUID
    public let kind: CorrectionKind

    /// What the engine had predicted at the moment of correction.
    public let predicted: PredictedSnapshot
    /// What the user said is correct.
    public let groundTruth: GroundTruthSnapshot
    /// Engine state at the moment of correction (`.committed(A)` typically).
    public let engineStateRaw: String
    /// Recent ASR chunks fed to the matcher leading up to the correction.
    public let recentChunks: [ChunkSnapshot]
    /// Path to a saved audio buffer covering `[audioStart, audioEnd]`, or
    /// `nil` if the user did not opt in to audio retention.
    public let audioBufferPath: String?
    /// Window of audio time covered by `recentChunks`.
    public let audioStart: TimeInterval
    public let audioEnd: TimeInterval
    /// Optional free-text note from Sevadar.
    public let notes: String?

    public init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        sessionId: UUID,
        kind: CorrectionKind,
        predicted: PredictedSnapshot,
        groundTruth: GroundTruthSnapshot,
        engineStateRaw: String,
        recentChunks: [ChunkSnapshot] = [],
        audioBufferPath: String? = nil,
        audioStart: TimeInterval = 0,
        audioEnd: TimeInterval = 0,
        notes: String? = nil
    ) {
        self.id = id
        self.timestamp = timestamp
        self.sessionId = sessionId
        self.kind = kind
        self.predicted = predicted
        self.groundTruth = groundTruth
        self.engineStateRaw = engineStateRaw
        self.recentChunks = recentChunks
        self.audioBufferPath = audioBufferPath
        self.audioStart = audioStart
        self.audioEnd = audioEnd
        self.notes = notes
    }

    // MARK: - Snapshots

    public struct PredictedSnapshot: Codable, Sendable, Equatable {
        public let shabadId: Int
        public let lineIdx: Int?
        public let confidence: Double?
        /// Up-to-N runner-up scores keyed by shabad id, captured at the
        /// moment of correction. Useful for analyzing what the engine almost
        /// chose.
        public let runnerUps: [Int: Double]

        public init(shabadId: Int, lineIdx: Int? = nil, confidence: Double? = nil, runnerUps: [Int: Double] = [:]) {
            self.shabadId = shabadId
            self.lineIdx = lineIdx
            self.confidence = confidence
            self.runnerUps = runnerUps
        }
    }

    public struct GroundTruthSnapshot: Codable, Sendable, Equatable {
        public let shabadId: Int
        public let lineIdx: Int?

        public init(shabadId: Int, lineIdx: Int? = nil) {
            self.shabadId = shabadId
            self.lineIdx = lineIdx
        }
    }

    public struct ChunkSnapshot: Codable, Sendable, Equatable {
        public let start: TimeInterval
        public let end: TimeInterval
        public let text: String

        public init(start: TimeInterval, end: TimeInterval, text: String) {
            self.start = start
            self.end = end
            self.text = text
        }

        /// Build from a library `AsrChunk`.
        public init(_ chunk: AsrChunk) {
            self.start = chunk.start
            self.end = chunk.end
            self.text = chunk.text
        }
    }
}
