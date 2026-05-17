//
//  CorrectionLogTests.swift
//  SangatAppTests — M5.1 (Foundations)
//
//  Pins the no-op behavior of `NoopCorrectionLog`. The protocol is the
//  M5.1 deliverable; the real implementation lands in M5.6 alongside the
//  correction touchpoints in the UI.

import XCTest
import GurbaniCaptioning
@testable import SangatApp

final class CorrectionLogTests: XCTestCase {

    func testNoopAcceptsRecordWithoutPersistence() {
        let log = NoopCorrectionLog()
        log.record(makeEvent())
        XCTAssertEqual(log.approximateCount, 0)
        XCTAssertTrue(log.recent(limit: 100).isEmpty)
    }

    func testNoopRecordIsIdempotent() {
        let log = NoopCorrectionLog()
        for _ in 0..<10 {
            log.record(makeEvent())
        }
        XCTAssertEqual(log.approximateCount, 0)
    }

    func testNoopClearIsSafe() {
        let log = NoopCorrectionLog()
        log.record(makeEvent())
        log.clear()
        XCTAssertEqual(log.approximateCount, 0)
    }

    func testCorrectionEventRoundtripsThroughJSON() throws {
        let event = makeEvent()
        let data = try JSONEncoder().encode(event)
        let decoded = try JSONDecoder().decode(CorrectionEvent.self, from: data)
        XCTAssertEqual(event, decoded)
    }

    func testCorrectionEventCarriesAllRequiredFields() {
        let event = makeEvent()
        XCTAssertEqual(event.kind, .hardNegPos)
        XCTAssertEqual(event.predicted.shabadId, 1789)
        XCTAssertEqual(event.groundTruth.shabadId, 1788)
        XCTAssertFalse(event.engineStateRaw.isEmpty)
    }

    // MARK: - Fixture

    private func makeEvent() -> CorrectionEvent {
        CorrectionEvent(
            sessionId: UUID(),
            kind: .hardNegPos,
            predicted: .init(
                shabadId: 1789,
                lineIdx: 0,
                confidence: 87.2,
                runnerUps: [1788: 62.0, 941: 41.0]
            ),
            groundTruth: .init(shabadId: 1788, lineIdx: 0),
            engineStateRaw: "committed(1789)",
            recentChunks: [
                .init(start: 0, end: 5, text: "ਤਾਤੀ ਵਾਉ ਨ ਲਗਈ"),
            ],
            audioStart: 0,
            audioEnd: 5,
            notes: "test"
        )
    }
}
