//
//  InMemoryCorrectionLogTests.swift
//  SangatAppTests — M5.6.x (carry-forward)
//
//  Behavior contract for the session-scoped log. The protocol surface
//  is the same one `CorrectionLogSpy` and `NoopCorrectionLog` honor;
//  these tests assert this impl honors it correctly.

import XCTest
@testable import SangatApp

@MainActor
final class InMemoryCorrectionLogTests: XCTestCase {

    // MARK: - Append + count

    func testEmptyLogReportsZeroCount() {
        let log = InMemoryCorrectionLog()
        XCTAssertEqual(log.approximateCount, 0)
    }

    func testRecordIncrementsCount() {
        let log = InMemoryCorrectionLog()
        log.record(sampleEvent())
        XCTAssertEqual(log.approximateCount, 1)
        log.record(sampleEvent())
        XCTAssertEqual(log.approximateCount, 2)
    }

    // MARK: - recent(limit:)

    func testRecentReturnsNewestFirst() {
        let log = InMemoryCorrectionLog()
        let first = sampleEvent(shabadId: 1)
        let second = sampleEvent(shabadId: 2)
        let third = sampleEvent(shabadId: 3)
        log.record(first)
        log.record(second)
        log.record(third)

        let recent = log.recent(limit: 10)
        XCTAssertEqual(recent.map(\.predicted.shabadId), [3, 2, 1])
    }

    func testRecentRespectsLimit() {
        let log = InMemoryCorrectionLog()
        log.record(sampleEvent(shabadId: 1))
        log.record(sampleEvent(shabadId: 2))
        log.record(sampleEvent(shabadId: 3))
        XCTAssertEqual(log.recent(limit: 2).count, 2)
        XCTAssertEqual(log.recent(limit: 2).map(\.predicted.shabadId), [3, 2])
    }

    // MARK: - clear()

    func testClearEmptiesTheLog() {
        let log = InMemoryCorrectionLog()
        log.record(sampleEvent())
        log.record(sampleEvent())
        log.clear()
        XCTAssertEqual(log.approximateCount, 0)
        XCTAssertEqual(log.recent(limit: 10), [])
    }

    func testClearOnEmptyLogIsNoOp() {
        let log = InMemoryCorrectionLog()
        log.clear()
        XCTAssertEqual(log.approximateCount, 0)
    }

    // MARK: - Thread-safety smoke

    func testConcurrentRecordsDoNotCrash() async {
        let log = InMemoryCorrectionLog()
        // Stress: 200 concurrent records. NSLock has to serialize.
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<200 {
                group.addTask {
                    log.record(self.sampleEvent(shabadId: i))
                }
            }
        }
        XCTAssertEqual(log.approximateCount, 200,
                       "All records must land; NSLock prevents tearing")
    }

    // MARK: - Helpers

    /// `nonisolated` so the concurrency stress test can call it from
    /// `TaskGroup.addTask`'s global-actor context. The factory itself
    /// touches no actor-isolated state.
    nonisolated private func sampleEvent(shabadId: Int = 1789) -> CorrectionEvent {
        CorrectionEventBuilder.makeHardNegPos(
            sessionId: UUID(),
            predictedShabadId: shabadId,
            predictedLineIdx: nil,
            predictedConfidence: nil,
            runnerUps: [:],
            correctedShabadId: shabadId + 1,
            engineStateRaw: "listening"
        )
    }
}
