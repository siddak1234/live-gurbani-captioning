//
//  CorrectionEventBuilderTests.swift
//  SangatAppTests — M5.6 (Correction loop surfaces)
//
//  Pure-shape tests for `CorrectionEventBuilder`. Each kind has one
//  factory; each factory has a deterministic mapping from inputs to
//  field values. We assert that mapping verbatim so future edits to
//  the builder can't silently change the wire format (these events
//  are the eventual fine-tune pipeline's training input — wire-format
//  changes are real refactors, not casual reshuffles).

import XCTest
import GurbaniCaptioning
@testable import SangatApp

@MainActor
final class CorrectionEventBuilderTests: XCTestCase {

    // MARK: - hardNegPos

    func testHardNegPosCapturesFullPredictedSnapshot() {
        let fixedNow = Date(timeIntervalSince1970: 1_700_000_000)
        let fixedId = UUID()
        let session = UUID()
        let event = CorrectionEventBuilder.makeHardNegPos(
            sessionId: session,
            predictedShabadId: 1789,
            predictedLineIdx: 3,
            predictedConfidence: 88.4,
            runnerUps: [1341: 84.1, 4377: 79.2],
            correctedShabadId: 1341,
            engineStateRaw: "committed(1789)",
            now: fixedNow,
            id: fixedId
        )

        XCTAssertEqual(event.id, fixedId)
        XCTAssertEqual(event.timestamp, fixedNow)
        XCTAssertEqual(event.sessionId, session)
        XCTAssertEqual(event.kind, .hardNegPos)
        XCTAssertEqual(event.predicted.shabadId, 1789)
        XCTAssertEqual(event.predicted.lineIdx, 3)
        XCTAssertEqual(event.predicted.confidence, 88.4)
        XCTAssertEqual(event.predicted.runnerUps, [1341: 84.1, 4377: 79.2])
        XCTAssertEqual(event.groundTruth.shabadId, 1341)
        XCTAssertNil(event.groundTruth.lineIdx, "Line-level truth not captured by hardNegPos")
        XCTAssertEqual(event.engineStateRaw, "committed(1789)")
        XCTAssertEqual(event.recentChunks, [])
        XCTAssertNil(event.audioBufferPath, "Audio retention is opt-in M5.8 work; M5.6 must not set this")
        XCTAssertNil(event.notes)
    }

    func testHardNegPosThreadsAudioBufferPathWhenProvided() {
        // Phase 2b: the picker / wrong-shabad sites pass a captured clip path.
        let event = CorrectionEventBuilder.makeHardNegPos(
            sessionId: UUID(),
            predictedShabadId: 1, predictedLineIdx: nil, predictedConfidence: nil,
            runnerUps: [:], correctedShabadId: 2,
            engineStateRaw: "committed(1)",
            audioBufferPath: "/tmp/correction-clip.m4a"
        )
        XCTAssertEqual(event.audioBufferPath, "/tmp/correction-clip.m4a")
    }

    // MARK: - runnerUpEndorsed

    func testRunnerUpEndorsedKindAndTruth() {
        let event = CorrectionEventBuilder.makeRunnerUpEndorsed(
            sessionId: UUID(),
            predictedShabadId: 8,
            predictedConfidence: 72.0,
            runnerUps: [3: 70.1, 1789: 65.4],
            endorsedShabadId: 3,
            engineStateRaw: "tentative(8)"
        )

        XCTAssertEqual(event.kind, .runnerUpEndorsed)
        XCTAssertEqual(event.predicted.shabadId, 8)
        XCTAssertEqual(event.groundTruth.shabadId, 3)
        XCTAssertEqual(event.engineStateRaw, "tentative(8)")
    }

    // MARK: - lineNudge

    func testLineNudgeKeepsSameShabadAndAdvancesLine() {
        let event = CorrectionEventBuilder.makeLineNudge(
            sessionId: UUID(),
            shabadId: 1789,
            predictedLineIdx: 4,
            delta: 1,
            engineStateRaw: "committed(1789)"
        )

        XCTAssertEqual(event.kind, .lineNudge)
        XCTAssertEqual(event.predicted.shabadId, 1789)
        XCTAssertEqual(event.predicted.lineIdx, 4)
        XCTAssertEqual(event.groundTruth.shabadId, 1789, "Nudge stays within the same shabad")
        XCTAssertEqual(event.groundTruth.lineIdx, 5)
        XCTAssertEqual(event.engineStateRaw, "committed(1789)")
    }

    func testLineNudgeBackwardEncodesNegativeDeltaCorrectly() {
        let event = CorrectionEventBuilder.makeLineNudge(
            sessionId: UUID(),
            shabadId: 1789,
            predictedLineIdx: 4,
            delta: -1,
            engineStateRaw: "committed(1789)"
        )
        XCTAssertEqual(event.groundTruth.lineIdx, 3)
    }

    // MARK: - retroactive

    func testRetroactiveCapturesFlaggedTime() {
        let flaggedAt = Date(timeIntervalSince1970: 1_700_000_500)
        let event = CorrectionEventBuilder.makeRetroactive(
            sessionId: UUID(),
            flaggedShabadId: 1789,
            correctedShabadId: 1341,
            flaggedAt: flaggedAt,
            engineStateRaw: "listening",
            notes: "Wrong shabad from earlier today"
        )

        XCTAssertEqual(event.kind, .retroactive)
        XCTAssertEqual(event.predicted.shabadId, 1789)
        XCTAssertEqual(event.groundTruth.shabadId, 1341)
        XCTAssertEqual(event.audioStart, flaggedAt.timeIntervalSince1970)
        XCTAssertEqual(event.audioEnd, flaggedAt.timeIntervalSince1970)
        XCTAssertEqual(event.notes, "Wrong shabad from earlier today")
    }

    // MARK: - engineStateRaw stringification

    func testEngineStateRawListening() {
        XCTAssertEqual(CorrectionEventBuilder.engineStateRaw(.listening), "listening")
    }

    func testEngineStateRawTentativeIncludesShabadId() {
        XCTAssertEqual(CorrectionEventBuilder.engineStateRaw(.tentative(shabadId: 1789)),
                       "tentative(1789)")
    }

    func testEngineStateRawCommittedIncludesShabadId() {
        XCTAssertEqual(CorrectionEventBuilder.engineStateRaw(.committed(shabadId: 1341)),
                       "committed(1341)")
    }

    // MARK: - Invariants

    func testEveryFactoryClearsAudioBufferPath() {
        // Audio retention is M5.8 territory. None of the M5.6 factories
        // may set `audioBufferPath` — privacy boundary.
        let session = UUID()
        let hardNegPos = CorrectionEventBuilder.makeHardNegPos(
            sessionId: session,
            predictedShabadId: 1, predictedLineIdx: nil, predictedConfidence: nil,
            runnerUps: [:], correctedShabadId: 2,
            engineStateRaw: "listening"
        )
        let runnerUp = CorrectionEventBuilder.makeRunnerUpEndorsed(
            sessionId: session,
            predictedShabadId: 1, predictedConfidence: nil,
            runnerUps: [:], endorsedShabadId: 2, engineStateRaw: "listening"
        )
        let nudge = CorrectionEventBuilder.makeLineNudge(
            sessionId: session,
            shabadId: 1, predictedLineIdx: 0, delta: 1, engineStateRaw: "committed(1)"
        )
        let retro = CorrectionEventBuilder.makeRetroactive(
            sessionId: session,
            flaggedShabadId: 1, correctedShabadId: 2,
            flaggedAt: Date(), engineStateRaw: "listening"
        )

        for event in [hardNegPos, runnerUp, nudge, retro] {
            XCTAssertNil(event.audioBufferPath,
                         "Kind \(event.kind.rawValue) must not set audioBufferPath in M5.6")
        }
    }

    func testEveryFactoryStampsSessionId() {
        let session = UUID()
        let events = [
            CorrectionEventBuilder.makeHardNegPos(
                sessionId: session,
                predictedShabadId: 1, predictedLineIdx: nil, predictedConfidence: nil,
                runnerUps: [:], correctedShabadId: 2, engineStateRaw: "listening"
            ),
            CorrectionEventBuilder.makeRunnerUpEndorsed(
                sessionId: session,
                predictedShabadId: 1, predictedConfidence: nil,
                runnerUps: [:], endorsedShabadId: 2, engineStateRaw: "listening"
            ),
            CorrectionEventBuilder.makeLineNudge(
                sessionId: session,
                shabadId: 1, predictedLineIdx: 0, delta: 1, engineStateRaw: "committed(1)"
            ),
            CorrectionEventBuilder.makeRetroactive(
                sessionId: session,
                flaggedShabadId: 1, correctedShabadId: 2,
                flaggedAt: Date(), engineStateRaw: "listening"
            )
        ]
        for event in events {
            XCTAssertEqual(event.sessionId, session,
                           "Kind \(event.kind.rawValue) must stamp the supplied sessionId")
        }
    }
}
