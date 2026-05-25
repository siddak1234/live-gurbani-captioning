//
//  SyncCoordinatorTests.swift
//  SangatAppTests — corrections feedback loop Phase 4
//
//  Drives the outbox logic with a fake uploader + fake reachability over a real
//  in-memory durable store, covering the consent/connectivity gates and every
//  upload outcome. No network.

import XCTest
@testable import SangatApp
import GurbaniCaptioning

@MainActor
final class SyncCoordinatorTests: XCTestCase {

    // MARK: - Fakes

    private final class FakeUploader: CorrectionsUploading, @unchecked Sendable {
        var outcome: UploadOutcome
        private(set) var uploadCount = 0
        init(_ outcome: UploadOutcome) { self.outcome = outcome }
        func upload(_ envelope: CorrectionEnvelope) async -> UploadOutcome {
            uploadCount += 1
            return outcome
        }
    }

    private struct FakeReachability: ReachabilityProviding {
        let isOnline: Bool
        let isOnWifi: Bool
    }

    // MARK: - Helpers

    private func event() -> CorrectionEvent {
        CorrectionEvent(
            sessionId: UUID(), kind: .hardNegPos,
            predicted: .init(shabadId: 4377, lineIdx: 1, confidence: 0.6, runnerUps: [:]),
            groundTruth: .init(shabadId: 1821, lineIdx: nil),
            engineStateRaw: "committed(4377)"
        )
    }

    private func makeLog() throws -> DurableCorrectionLog { try .inMemory() }

    private func makeCoordinator(
        log: DurableCorrectionLog,
        uploader: any CorrectionsUploading,
        online: Bool = true,
        wifi: Bool = true,
        uploadOptIn: Bool = true,
        wifiOnly: Bool = true
    ) -> SyncCoordinator {
        let prefs = Preferences.inMemory()
        prefs.uploadOptIn = uploadOptIn
        prefs.wifiOnlyUpload = wifiOnly
        return SyncCoordinator(
            log: log,
            uploader: uploader,
            reachability: FakeReachability(isOnline: online, isOnWifi: wifi),
            preferences: prefs,
            deviceId: UUID()
        )
    }

    // MARK: - Tests

    func testUploadsPendingAndMarksUploaded() async throws {
        let log = try makeLog()
        let e = event(); log.record(e)
        let uploader = FakeUploader(.success)
        let coordinator = makeCoordinator(log: log, uploader: uploader)

        let count = await coordinator.sync()
        XCTAssertEqual(count, 1)
        XCTAssertEqual(uploader.uploadCount, 1)
        XCTAssertEqual(log.status(forId: e.id), .uploaded)
        XCTAssertTrue(log.pending(limit: 10).isEmpty)
    }

    func testSkipsWhenNotOptedIn() async throws {
        let log = try makeLog()
        let e = event(); log.record(e)
        let uploader = FakeUploader(.success)
        let coordinator = makeCoordinator(log: log, uploader: uploader, uploadOptIn: false)

        let count = await coordinator.sync()
        XCTAssertEqual(count, 0)
        XCTAssertEqual(uploader.uploadCount, 0, "must not upload when opted out")
        XCTAssertEqual(log.status(forId: e.id), .pending)
    }

    func testSkipsWhenOffline() async throws {
        let log = try makeLog()
        log.record(event())
        let uploader = FakeUploader(.success)
        let coordinator = makeCoordinator(log: log, uploader: uploader, online: false)
        let count = await coordinator.sync()
        XCTAssertEqual(count, 0)
        XCTAssertEqual(uploader.uploadCount, 0)
    }

    func testWifiOnlyBlocksCellular() async throws {
        let log = try makeLog()
        log.record(event())
        let uploader = FakeUploader(.success)
        let coordinator = makeCoordinator(log: log, uploader: uploader, wifi: false, wifiOnly: true)
        let count = await coordinator.sync()
        XCTAssertEqual(count, 0)
        XCTAssertEqual(uploader.uploadCount, 0)
    }

    func testCellularAllowedWhenWifiOnlyOff() async throws {
        let log = try makeLog()
        log.record(event())
        let uploader = FakeUploader(.success)
        let coordinator = makeCoordinator(log: log, uploader: uploader, wifi: false, wifiOnly: false)
        let count = await coordinator.sync()
        XCTAssertEqual(count, 1)
    }

    func testRetryableLeavesPending() async throws {
        let log = try makeLog()
        let e = event(); log.record(e)
        let uploader = FakeUploader(.retryable("offline blip"))
        let coordinator = makeCoordinator(log: log, uploader: uploader)
        _ = await coordinator.sync()
        XCTAssertEqual(log.status(forId: e.id), .pending, "retryable stays in the outbox")
    }

    func testPermanentMarksFailed() async throws {
        let log = try makeLog()
        let e = event(); log.record(e)
        let uploader = FakeUploader(.permanent("422"))
        let coordinator = makeCoordinator(log: log, uploader: uploader)
        _ = await coordinator.sync()
        XCTAssertEqual(log.status(forId: e.id), .failed)
        XCTAssertTrue(log.pending(limit: 10).isEmpty)
    }

    func testAlreadyUploadedTreatedAsSuccess() async throws {
        let log = try makeLog()
        let e = event(); log.record(e)
        let uploader = FakeUploader(.alreadyUploaded)
        let coordinator = makeCoordinator(log: log, uploader: uploader)
        let count = await coordinator.sync()
        XCTAssertEqual(count, 1)
        XCTAssertEqual(log.status(forId: e.id), .uploaded)
    }

    func testRequeuesStaleUploadingThenUploads() async throws {
        let log = try makeLog()
        let e = event(); log.record(e)
        // Simulate a prior run interrupted mid-upload.
        log.markStatus(.uploading, forId: e.id)
        XCTAssertEqual(log.status(forId: e.id), .uploading)

        let uploader = FakeUploader(.success)
        let coordinator = makeCoordinator(log: log, uploader: uploader)
        let count = await coordinator.sync()
        XCTAssertEqual(count, 1, "stale uploading record requeued and sent")
        XCTAssertEqual(log.status(forId: e.id), .uploaded)
    }
}
