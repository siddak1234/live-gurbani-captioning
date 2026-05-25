//
//  CorrectionRowDTO.swift
//  GurbaniCaptioningApp · SangatApp · Sync
//
//  Created for the Sangat iOS app, corrections feedback loop Phase 4
//  (Sync engine). See docs/corrections_feedback_loop_plan.md.
//
//  Flattens a `CorrectionEnvelope` into the exact JSON shape of the
//  `public.corrections` table (snake_case columns), for the PostgREST insert.
//  Keys here MUST match the Phase 3 migration columns; the column mapping is
//  documented in the plan doc §3b/§8.

import Foundation
import GurbaniCaptioning

/// Snake_case row payload posted to PostgREST. `runner_ups` / `recent_chunks`
/// land in `jsonb` columns; optional fields are omitted when nil (the column
/// default/NULL applies).
struct CorrectionRowDTO: Encodable, Equatable {
    let id: UUID
    let device_id: UUID
    let session_id: UUID?
    let kind: String
    let predicted_shabad_id: Int
    let predicted_line_idx: Int?
    let confidence: Double?
    let runner_ups: [String: Double]
    let ground_truth_shabad_id: Int
    let ground_truth_line_idx: Int?
    let engine_state: String?
    let recent_chunks: [Chunk]
    let audio_path: String?
    let audio_start: Double
    let audio_end: Double
    let app_version: String?
    let build_number: String?
    let model_version: String?
    let schema_version: Int
    let created_at: Date

    struct Chunk: Encodable, Equatable {
        let start: Double
        let end: Double
        let text: String
    }

    init(_ envelope: CorrectionEnvelope) {
        let e = envelope.event
        id = e.id
        device_id = envelope.deviceId
        session_id = e.sessionId
        kind = e.kind.rawValue
        predicted_shabad_id = e.predicted.shabadId
        predicted_line_idx = e.predicted.lineIdx
        confidence = e.predicted.confidence
        // jsonb object keys must be strings.
        runner_ups = Dictionary(uniqueKeysWithValues: e.predicted.runnerUps.map { (String($0.key), $0.value) })
        ground_truth_shabad_id = e.groundTruth.shabadId
        ground_truth_line_idx = e.groundTruth.lineIdx
        engine_state = e.engineStateRaw
        recent_chunks = e.recentChunks.map { Chunk(start: $0.start, end: $0.end, text: $0.text) }
        audio_path = e.audioBufferPath
        audio_start = e.audioStart
        audio_end = e.audioEnd
        app_version = envelope.appVersion
        build_number = envelope.buildNumber
        model_version = envelope.modelVersion
        schema_version = envelope.schemaVersion
        created_at = e.timestamp
    }
}
