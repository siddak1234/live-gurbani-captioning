//
//  CorrectionRecord.swift
//  GurbaniCaptioningApp · SangatApp · Corrections
//
//  Created for the Sangat iOS app, corrections feedback loop Phase 1
//  (Durable local store). See docs/corrections_feedback_loop_plan.md §7 for
//  the documented on-device schema.
//
//  The SwiftData persistence entity backing `DurableCorrectionLog`. We store
//  the rich `CorrectionEvent` as a JSON blob (`payload`) plus a few indexed,
//  queryable columns (id, createdAt, syncStatus). Rationale: the on-device
//  store never *queries* on the inner correction fields — it only needs to
//  fetch "newest N" and "pending for upload" — so a blob keeps the schema
//  small and stable while the wire/Postgres shape (CorrectionEnvelope) stays
//  the single source of truth for the rich fields. This also insulates the
//  store from `CorrectionEvent` evolving: a new event field is a payload
//  change, not a SwiftData migration.
//
//  `id` is `.unique`, so re-recording the same correction id upserts rather
//  than duplicating — the same idempotency guarantee the upload path relies on.

import Foundation
import SwiftData

@Model
final class CorrectionRecord {

    /// Mirrors `CorrectionEvent.id`. Unique → idempotent inserts.
    @Attribute(.unique) var id: UUID

    /// Mirrors `CorrectionEvent.timestamp`. Indexed column for "newest N".
    var createdAt: Date

    /// `CorrectionSyncStatus.rawValue`. Queried by the outbox; stored as a
    /// string so it survives enum changes without a migration.
    var syncStatusRaw: String

    /// Number of upload attempts so far (incremented on each `.uploading`).
    var attemptCount: Int

    /// Last upload error string, if any. Diagnostic only.
    var lastError: String?

    /// Convenience mirror of `CorrectionEvent.audioBufferPath` so the sync /
    /// cleanup layers can find the local audio clip without decoding payload.
    var audioPath: String?

    /// JSON-encoded `CorrectionEvent` — the full rich record.
    var payload: Data

    init(
        id: UUID,
        createdAt: Date,
        syncStatusRaw: String,
        attemptCount: Int = 0,
        lastError: String? = nil,
        audioPath: String? = nil,
        payload: Data
    ) {
        self.id = id
        self.createdAt = createdAt
        self.syncStatusRaw = syncStatusRaw
        self.attemptCount = attemptCount
        self.lastError = lastError
        self.audioPath = audioPath
        self.payload = payload
    }
}
