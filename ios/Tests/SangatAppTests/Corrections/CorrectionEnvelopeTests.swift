//
//  CorrectionEnvelopeTests.swift
//  SangatAppTests — corrections feedback loop Phase 0
//
//  Verifies the upload wire contract: metadata defaults from `AppMetadata`,
//  the envelope id tracks the event id (idempotency key), and the whole thing
//  survives a JSON Codable round-trip (the sync layer will encode it for the
//  Edge Function).

import XCTest
@testable import SangatApp
import GurbaniCaptioning

final class CorrectionEnvelopeTests: XCTestCase {

    private func sampleEvent() -> CorrectionEvent {
        CorrectionEvent(
            sessionId: UUID(),
            kind: .hardNegPos,
            predicted: .init(shabadId: 4377, lineIdx: 2, confidence: 0.61, runnerUps: [1821: 0.55]),
            groundTruth: .init(shabadId: 1821, lineIdx: nil),
            engineStateRaw: "committed(4377)",
            recentChunks: [.init(start: 10, end: 15, text: "test chunk")]
        )
    }

    func testMetadataDefaultsFromAppMetadata() {
        let device = UUID()
        let envelope = CorrectionEnvelope(event: sampleEvent(), deviceId: device)
        XCTAssertEqual(envelope.deviceId, device)
        XCTAssertEqual(envelope.schemaVersion, AppMetadata.schemaVersion)
        XCTAssertEqual(envelope.appVersion, AppMetadata.appVersion)
        XCTAssertEqual(envelope.buildNumber, AppMetadata.buildNumber)
        XCTAssertEqual(envelope.modelVersion, AppMetadata.modelVersion)
    }

    func testEnvelopeIdTracksEventId() {
        let event = sampleEvent()
        let envelope = CorrectionEnvelope(event: event, deviceId: UUID())
        XCTAssertEqual(envelope.id, event.id, "envelope id must equal event id (idempotency key)")
    }

    func testCodableRoundTrip() throws {
        let envelope = CorrectionEnvelope(event: sampleEvent(), deviceId: UUID())
        let data = try JSONEncoder().encode(envelope)
        let decoded = try JSONDecoder().decode(CorrectionEnvelope.self, from: data)
        XCTAssertEqual(envelope, decoded)
    }
}
