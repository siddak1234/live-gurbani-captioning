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
        private(set) var lastStorageAudioPath: String?
        init(_ outcome: UploadOutcome) { self.outcome = outcome }
        func upload(_ envelope: CorrectionEnvelope, storageAudioPath: String?) async -> UploadOutcome {
            uploadCount += 1
            lastStorageAudioPath = storageAudioPath
            return outcome
        }
    }

    private final class FakeAudioUploader: CorrectionAudioUploading, @unchecked Sendable {
        var outcome: UploadOutcome
        private(set) var uploadCount = 0
        private(set) var lastKey: String?
        init(_ outcome: UploadOutcome) { self.outcome = outcome }
        func upload(fileURL: URL, toKey key: String) async -> UploadOutcome {
            uploadCount += 1
            lastKey = key
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
        wifiOnly: Bool = true,
        audioUploader: (any CorrectionAudioUploading)? = nil,
        audioWriter: AudioClipWriter? = nil,
        maxAttempts: Int = 5
    ) -> SyncCoordinator {
        let prefs = Preferences.inMemory()
        prefs.uploadOptIn = uploadOptIn
        prefs.wifiOnlyUpload = wifiOnly
        return SyncCoordinator(
            log: log,
            uploader: uploader,
            reachability: FakeReachability(isOnline: online, isOnWifi: wifi),
            preferences: prefs,
            deviceId: UUID(),
            audioUploader: audioUploader,
            audioWriter: audioWriter,
            maxAttempts: maxAttempts
        )
    }

    /// Event referencing a real temp clip file, plus the writer that owns it.
    private func eventWithClip() throws -> (CorrectionEvent, AudioClipWriter, URL) {
        let dir = URL.temporaryDirectory.appending(path: "clips-\(UUID().uuidString)")
        let writer = try AudioClipWriter(directory: dir)
        let clipURL = dir.appendingPathComponent("clip.m4a")
        try Data([0, 1, 2, 3, 4]).write(to: clipURL)
        let e = CorrectionEvent(
            sessionId: UUID(), kind: .hardNegPos,
            predicted: .init(shabadId: 4377, lineIdx: 1, confidence: 0.6, runnerUps: [:]),
            groundTruth: .init(shabadId: 1821, lineIdx: nil),
            engineStateRaw: "committed(4377)",
            audioBufferPath: clipURL.path
        )
        return (e, writer, clipURL)
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

    // MARK: - Audio upload (Phase 6a)

    func testUploadsAudioThenMetadataAndDeletesLocalClip() async throws {
        let log = try makeLog()
        let (e, writer, clipURL) = try eventWithClip()
        defer { try? FileManager.default.removeItem(at: writer.directory) }
        log.record(e)

        let audioUploader = FakeAudioUploader(.success)
        let metaUploader = FakeUploader(.success)
        let coordinator = makeCoordinator(log: log, uploader: metaUploader,
                                          audioUploader: audioUploader, audioWriter: writer)

        let count = await coordinator.sync()
        XCTAssertEqual(count, 1)
        XCTAssertEqual(audioUploader.uploadCount, 1)
        XCTAssertEqual(audioUploader.lastKey?.hasSuffix("\(e.id.uuidString).m4a"), true)
        XCTAssertEqual(metaUploader.lastStorageAudioPath, audioUploader.lastKey,
                       "metadata row carries the Storage key")
        XCTAssertFalse(FileManager.default.fileExists(atPath: clipURL.path),
                       "local clip deleted after successful upload")
        XCTAssertEqual(log.status(forId: e.id), .uploaded)
    }

    func testAudioRetryableKeepsRecordPendingAndClip() async throws {
        let log = try makeLog()
        let (e, writer, clipURL) = try eventWithClip()
        defer { try? FileManager.default.removeItem(at: writer.directory) }
        log.record(e)

        let audioUploader = FakeAudioUploader(.retryable("net blip"))
        let metaUploader = FakeUploader(.success)
        let coordinator = makeCoordinator(log: log, uploader: metaUploader,
                                          audioUploader: audioUploader, audioWriter: writer)

        _ = await coordinator.sync()
        XCTAssertEqual(metaUploader.uploadCount, 0, "metadata not sent when audio upload is retryable")
        XCTAssertEqual(log.status(forId: e.id), .pending, "whole record retries next trigger")
        XCTAssertTrue(FileManager.default.fileExists(atPath: clipURL.path), "clip kept for retry")
    }

    // MARK: - Observability + robustness (Phase 7)

    func testStatsRecordLastSyncAndUploadCount() async throws {
        let log = try makeLog()
        log.record(event())
        let coordinator = makeCoordinator(log: log, uploader: FakeUploader(.success))
        XCTAssertNil(coordinator.stats.lastSyncDate)
        _ = await coordinator.sync()
        XCTAssertNotNil(coordinator.stats.lastSyncDate)
        XCTAssertEqual(coordinator.stats.lastUploaded, 1)
        XCTAssertNil(coordinator.stats.lastError)
    }

    func testStatsRecordLastError() async throws {
        let log = try makeLog()
        log.record(event())
        let coordinator = makeCoordinator(log: log, uploader: FakeUploader(.permanent("422 boom")))
        _ = await coordinator.sync()
        XCTAssertEqual(coordinator.stats.lastError, "422 boom")
    }

    func testStatusCountsExposedThroughCoordinator() async throws {
        let log = try makeLog()
        log.record(event())
        let coordinator = makeCoordinator(log: log, uploader: FakeUploader(.success))
        XCTAssertEqual(coordinator.statusCounts().pending, 1)
        _ = await coordinator.sync()
        XCTAssertEqual(coordinator.statusCounts().uploaded, 1)
    }

    func testParksPoisonRecordAtSyncStart() async throws {
        let log = try makeLog()
        let e = event()
        log.record(e)
        // Simulate prior exhausted attempts.
        for _ in 0..<3 { log.markStatus(.uploading, forId: e.id) }
        let uploader = FakeUploader(.success)
        let coordinator = makeCoordinator(log: log, uploader: uploader, maxAttempts: 3)
        let count = await coordinator.sync()
        XCTAssertEqual(count, 0, "exhausted record is parked, not uploaded")
        XCTAssertEqual(uploader.uploadCount, 0)
        XCTAssertEqual(log.status(forId: e.id), .failed)
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
