//
//  CorrectionRowDTOTests.swift
//  SangatAppTests — corrections feedback loop Phase 4
//
//  Verifies the envelope → snake_case row JSON mapping matches the
//  `public.corrections` columns (validated against the live project in Phase 3).

import XCTest
@testable import SangatApp
import GurbaniCaptioning

final class CorrectionRowDTOTests: XCTestCase {

    private func envelope() -> CorrectionEnvelope {
        let event = CorrectionEvent(
            sessionId: UUID(),
            kind: .hardNegPos,
            predicted: .init(shabadId: 4377, lineIdx: 2, confidence: 0.61, runnerUps: [1821: 0.55]),
            groundTruth: .init(shabadId: 1821, lineIdx: nil),
            engineStateRaw: "committed(4377)",
            recentChunks: [.init(start: 10, end: 15, text: "ਤੇਸਟ")],
            audioBufferPath: "/tmp/clip.m4a",
            audioStart: 10, audioEnd: 15
        )
        return CorrectionEnvelope(event: event, deviceId: UUID())
    }

    /// Encode and decode to a generic JSON object to assert column-name shape.
    private func encodedObject(_ env: CorrectionEnvelope, storageAudioPath: String? = nil) throws -> [String: Any] {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(CorrectionRowDTO(env, storageAudioPath: storageAudioPath))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testKeysAreSnakeCaseAndPresent() throws {
        let obj = try encodedObject(envelope())
        for key in ["id", "device_id", "session_id", "kind", "predicted_shabad_id",
                    "ground_truth_shabad_id", "runner_ups", "recent_chunks",
                    "schema_version", "model_version", "created_at"] {
            XCTAssertNotNil(obj[key], "missing column key \(key)")
        }
        // camelCase must NOT leak.
        XCTAssertNil(obj["shabadId"])
        XCTAssertNil(obj["deviceId"])
    }

    func testValuesMapCorrectly() throws {
        let env = envelope()
        let obj = try encodedObject(env)
        XCTAssertEqual(obj["kind"] as? String, "hardNegPos")
        XCTAssertEqual(obj["predicted_shabad_id"] as? Int, 4377)
        XCTAssertEqual(obj["ground_truth_shabad_id"] as? Int, 1821)
        XCTAssertEqual(obj["schema_version"] as? Int, env.schemaVersion)
        XCTAssertEqual(obj["model_version"] as? String, env.modelVersion)
        // audio_path holds the Storage key (not the device-local path), or is
        // omitted when no clip was uploaded.
        XCTAssertNil(obj["audio_path"], "no storage key → audio_path omitted")
        let withAudio = try encodedObject(env, storageAudioPath: "device/abc.m4a")
        XCTAssertEqual(withAudio["audio_path"] as? String, "device/abc.m4a")
        // runner_ups jsonb object: integer shabad ids become string keys.
        let runners = try XCTUnwrap(obj["runner_ups"] as? [String: Any])
        XCTAssertEqual(runners["1821"] as? Double, 0.55)
        // recent_chunks jsonb array of {start,end,text}.
        let chunks = try XCTUnwrap(obj["recent_chunks"] as? [[String: Any]])
        XCTAssertEqual(chunks.first?["text"] as? String, "ਤੇਸਟ")
    }

    func testNilOptionalsAreOmitted() throws {
        // ground_truth_line_idx is nil here → key omitted (column NULL applies).
        let obj = try encodedObject(envelope())
        XCTAssertNil(obj["ground_truth_line_idx"])
    }
}
