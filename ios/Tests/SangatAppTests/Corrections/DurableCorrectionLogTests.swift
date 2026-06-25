//
//  DurableCorrectionLogTests.swift
//  SangatAppTests — corrections feedback loop Phase 1
//
//  Verifies the SwiftData-backed correction store: insert/count, newest-first
//  ordering, clear, id dedupe, the sync-status lifecycle, and — the headline
//  Phase 1 claim — that records survive across container instances (≈ an app
//  relaunch) when backed by an on-disk store.

import XCTest
import SwiftData
@testable import SangatApp
import GurbaniCaptioning

final class DurableCorrectionLogTests: XCTestCase {

    // MARK: - Helpers

    private func event(shabad: Int = 4377, at time: TimeInterval? = nil) -> CorrectionEvent {
        CorrectionEvent(
            timestamp: time.map { Date(timeIntervalSince1970: $0) } ?? Date(),
            sessionId: UUID(),
            kind: .hardNegPos,
            predicted: .init(shabadId: shabad, lineIdx: 1, confidence: 0.6, runnerUps: [1821: 0.5]),
            groundTruth: .init(shabadId: 1821, lineIdx: nil),
            engineStateRaw: "committed(\(shabad))"
        )
    }

    private func inMemoryLog() throws -> DurableCorrectionLog {
        try DurableCorrectionLog.inMemory()
    }

    // MARK: - Tests

    func testRecordIncrementsCount() throws {
        let log = try inMemoryLog()
        XCTAssertEqual(log.approximateCount, 0)
        log.record(event())
        XCTAssertEqual(log.approximateCount, 1)
    }

    func testRecentReturnsNewestFirst() throws {
        let log = try inMemoryLog()
        let older = event(shabad: 1, at: 100)
        let newer = event(shabad: 2, at: 200)
        log.record(older)
        log.record(newer)
        let recent = log.recent(limit: 10)
        XCTAssertEqual(recent.count, 2)
        XCTAssertEqual(recent.first?.id, newer.id, "recent must be newest-first")
    }

    func testRecentRoundTripsTheFullEvent() throws {
        let log = try inMemoryLog()
        let e = event()
        log.record(e)
        let decoded = log.recent(limit: 1).first
        XCTAssertEqual(decoded, e, "payload must decode back to the original event")
    }

    func testClearEmptiesStore() throws {
        let log = try inMemoryLog()
        log.record(event())
        log.record(event())
        log.clear()
        XCTAssertEqual(log.approximateCount, 0)
    }

    func testDedupesOnId() throws {
        let log = try inMemoryLog()
        let e = event()
        log.record(e)
        log.record(e) // same id → upsert, not duplicate
        XCTAssertEqual(log.approximateCount, 1)
    }

    func testNewRecordsStartPending() throws {
        let log = try inMemoryLog()
        let e = event()
        log.record(e)
        XCTAssertEqual(log.status(forId: e.id), .pending)
        XCTAssertEqual(log.pending(limit: 10).map(\.id), [e.id])
    }

    func testStatusTransitionsRemoveFromPending() throws {
        let log = try inMemoryLog()
        let e = event()
        log.record(e)
        log.markStatus(.uploading, forId: e.id)
        XCTAssertEqual(log.status(forId: e.id), .uploading)
        log.markStatus(.uploaded, forId: e.id)
        XCTAssertEqual(log.status(forId: e.id), .uploaded)
        XCTAssertTrue(log.pending(limit: 10).isEmpty, "uploaded records leave the outbox")
    }

    // MARK: - Observability + robustness (Phase 7)

    func testStatusCountsReflectState() throws {
        let log = try inMemoryLog()
        let a = event(); let b = event(); let c = event()
        log.record(a); log.record(b); log.record(c)
        log.markStatus(.uploaded, forId: a.id)
        log.markStatus(.failed, forId: b.id)
        let counts = log.statusCounts()
        XCTAssertEqual(counts.pending, 1)
        XCTAssertEqual(counts.uploaded, 1)
        XCTAssertEqual(counts.failed, 1)
        XCTAssertEqual(counts.total, 3)
    }

    func testParkExhaustedMarksOverAttemptedAsFailed() throws {
        let log = try inMemoryLog()
        let e = event()
        log.record(e)
        // Each markStatus(.uploading) bumps attemptCount.
        for _ in 0..<3 { log.markStatus(.uploading, forId: e.id) }
        log.parkExhausted(maxAttempts: 3)
        XCTAssertEqual(log.status(forId: e.id), .failed)
        XCTAssertTrue(log.pending(limit: 10).isEmpty, "parked record leaves the outbox")
    }

    func testParkExhaustedSparesUnderAttempted() throws {
        let log = try inMemoryLog()
        let e = event()
        log.record(e)
        log.markStatus(.uploading, forId: e.id)   // attemptCount = 1
        log.markStatus(.pending, forId: e.id)
        log.parkExhausted(maxAttempts: 5)
        XCTAssertEqual(log.status(forId: e.id), .pending, "under the cap stays in the outbox")
    }

    func testPersistsAcrossContainersOnDisk() throws {
        let url = URL.temporaryDirectory.appending(path: "corr-\(UUID().uuidString).store")
        defer {
            for suffix in ["", "-shm", "-wal"] {
                try? FileManager.default.removeItem(at: URL(fileURLWithPath: url.path + suffix))
            }
        }
        let e = event()

        // First "launch": record, then read count to drain the async write.
        do {
            let config = ModelConfiguration(url: url)
            let container = try ModelContainer(for: CorrectionRecord.self, configurations: config)
            let log = DurableCorrectionLog(container: container)
            log.record(e)
            XCTAssertEqual(log.approximateCount, 1)
        }

        // Second "launch": a fresh container over the same file must see it.
        let config = ModelConfiguration(url: url)
        let container = try ModelContainer(for: CorrectionRecord.self, configurations: config)
        let reopened = DurableCorrectionLog(container: container)
        XCTAssertEqual(reopened.approximateCount, 1, "records must survive across launches")
        XCTAssertEqual(reopened.recent(limit: 10).first?.id, e.id)
    }
}
