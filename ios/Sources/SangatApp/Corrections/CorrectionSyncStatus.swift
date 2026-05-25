//
//  CorrectionSyncStatus.swift
//  GurbaniCaptioningApp · SangatApp · Corrections
//
//  Created for the Sangat iOS app, corrections feedback loop Phase 1
//  (Durable local store). See docs/corrections_feedback_loop_plan.md.
//
//  Per-record outbox state. Set to `.pending` on capture; the sync engine
//  (Phase 4) transitions it through `.uploading` → `.uploaded`, or to
//  `.failed` on a non-retryable error. Only `.pending` records are eligible
//  for upload, so the field doubles as the outbox query key.

import Foundation

/// Upload lifecycle of a stored correction.
public enum CorrectionSyncStatus: String, Codable, Sendable, CaseIterable {
    /// Captured on-device, not yet uploaded. Eligible for the outbox.
    case pending
    /// Upload in flight (Phase 4). Excluded from the outbox to avoid double-send.
    case uploading
    /// Confirmed stored server-side. Terminal; local audio may be purged.
    case uploaded
    /// Non-retryable failure (e.g. rejected by the Edge Function). Terminal
    /// until manually retried; carries `lastError` for diagnosis.
    case failed
}

/// Snapshot of how many stored corrections sit in each sync state — the outbox
/// observability surface (Phase 7).
public struct CorrectionSyncCounts: Equatable, Sendable {
    public var pending = 0
    public var uploading = 0
    public var uploaded = 0
    public var failed = 0

    public init(pending: Int = 0, uploading: Int = 0, uploaded: Int = 0, failed: Int = 0) {
        self.pending = pending
        self.uploading = uploading
        self.uploaded = uploaded
        self.failed = failed
    }

    public var total: Int { pending + uploading + uploaded + failed }
}
