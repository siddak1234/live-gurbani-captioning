//
//  CorrectionEnvelope.swift
//  GurbaniCaptioningApp · SangatApp · Corrections
//
//  Created for the Sangat iOS app, corrections feedback loop Phase 0
//  (Contracts & consent scaffolding). See
//  docs/corrections_feedback_loop_plan.md.
//
//  The wire contract for uploading a correction to Supabase. Wraps the existing
//  `CorrectionEvent` (the rich on-device record) with the lineage the training
//  side needs: which device, which app build, which model erred, and which
//  schema version to parse. The column mapping is documented in
//  docs/corrections_feedback_loop_plan.md §3b.
//
//  Phase 0 status: defined and tested, **sent by nobody yet** — the sync
//  engine (Phase 4) builds and uploads these via the Edge Function gateway.
//  No behavior change today.

import Foundation
import GurbaniCaptioning

/// One correction, packaged for upload. `Codable` so the sync layer can encode
/// it to JSON for the Supabase Edge Function. `event.id` is the idempotency key
/// — retried uploads upsert on it server-side, so an offline batch drains
/// exactly-once.
public struct CorrectionEnvelope: Codable, Sendable, Equatable, Identifiable {

    /// Stable per-correction id — mirrors `event.id`. Doubles as the upload
    /// idempotency key (`onConflict=id` server-side).
    public var id: UUID { event.id }

    /// Wire-schema version (`AppMetadata.schemaVersion`).
    public let schemaVersion: Int
    /// Anonymous device id (`DeviceIdentity.deviceId`).
    public let deviceId: UUID
    /// App version string (`AppMetadata.appVersion`).
    public let appVersion: String
    /// Build number (`AppMetadata.buildNumber`).
    public let buildNumber: String
    /// Model that produced the mistake (`AppMetadata.modelVersion`).
    public let modelVersion: String
    /// The on-device correction record.
    public let event: CorrectionEvent

    /// Designated initializer. Metadata fields default from `AppMetadata` so
    /// call sites only supply the `event` and the `deviceId`.
    public init(
        event: CorrectionEvent,
        deviceId: UUID,
        schemaVersion: Int = AppMetadata.schemaVersion,
        appVersion: String = AppMetadata.appVersion,
        buildNumber: String = AppMetadata.buildNumber,
        modelVersion: String = AppMetadata.modelVersion
    ) {
        self.event = event
        self.deviceId = deviceId
        self.schemaVersion = schemaVersion
        self.appVersion = appVersion
        self.buildNumber = buildNumber
        self.modelVersion = modelVersion
    }
}
