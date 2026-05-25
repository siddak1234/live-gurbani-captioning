//
//  DurableCorrectionLog.swift
//  GurbaniCaptioningApp · SangatApp · Corrections
//
//  Created for the Sangat iOS app, corrections feedback loop Phase 1
//  (Durable local store). See docs/corrections_feedback_loop_plan.md.
//
//  SwiftData-backed `CorrectionLog`. Replaces `InMemoryCorrectionLog` (which
//  evaporated on app kill) so opted-in corrections survive launches and form a
//  durable outbox for the Phase 4 sync engine.
//
//  Concurrency
//  -----------
//  `CorrectionLog` is a synchronous, `Sendable`, any-actor protocol, while
//  SwiftData's `ModelContext` is not `Sendable` and must not be touched
//  concurrently. We reconcile the two the same way `InMemoryCorrectionLog`
//  does — confine all state to one serializing primitive. Here it's a private
//  serial `DispatchQueue`: the `ModelContext` is created on it and every
//  read/write runs on it. `record(_:)` dispatches async (non-blocking per the
//  protocol contract); reads use `sync` and, because the queue is serial and
//  FIFO, observe all prior writes. This keeps the class `@unchecked Sendable`
//  without a `@MainActor` hop or the `async` API a `@ModelActor` would force.
//
//  Phase 1 scope: persistence + per-record `syncStatus`. The status-transition
//  and `pending(limit:)` helpers below are concrete (not on the protocol) and
//  exist for the Phase 4 outbox; they are unused by app code today.

import Foundation
import SwiftData

public final class DurableCorrectionLog: CorrectionLog, @unchecked Sendable {

    private let container: ModelContainer
    private let context: ModelContext
    private let queue = DispatchQueue(label: "com.sangat.app.corrections.durable")
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    /// Construct over a caller-provided container (used by factories + tests).
    public init(container: ModelContainer) {
        self.container = container
        // Create the context on the serial queue so it is only ever touched
        // from that one execution context.
        self.context = queue.sync { ModelContext(container) }
    }

    /// On-disk store at SwiftData's default app location. Throws if the store
    /// can't be opened (caller falls back to in-memory so the app survives).
    public static func makeDefault() throws -> DurableCorrectionLog {
        let container = try ModelContainer(for: CorrectionRecord.self)
        return DurableCorrectionLog(container: container)
    }

    /// Ephemeral store (tests / previews). Same code path, no disk.
    public static func inMemory() throws -> DurableCorrectionLog {
        let config = ModelConfiguration(isStoredInMemoryOnly: true)
        let container = try ModelContainer(for: CorrectionRecord.self, configurations: config)
        return DurableCorrectionLog(container: container)
    }

    // MARK: - CorrectionLog

    public func record(_ event: CorrectionEvent) {
        queue.async { [weak self] in
            guard let self else { return }
            do {
                let payload = try self.encoder.encode(event)
                let record = CorrectionRecord(
                    id: event.id,
                    createdAt: event.timestamp,
                    syncStatusRaw: CorrectionSyncStatus.pending.rawValue,
                    audioPath: event.audioBufferPath,
                    payload: payload
                )
                self.context.insert(record)
                try self.context.save()
                AppLogger.corrections.info("Durable correction recorded: kind=\(event.kind.rawValue, privacy: .public) predicted=\(event.predicted.shabadId, privacy: .public) truth=\(event.groundTruth.shabadId, privacy: .public)")
            } catch {
                AppLogger.corrections.error("Durable correction record failed: \(error.localizedDescription, privacy: .public)")
            }
        }
    }

    public func recent(limit: Int) -> [CorrectionEvent] {
        queue.sync {
            var descriptor = FetchDescriptor<CorrectionRecord>(
                sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
            )
            descriptor.fetchLimit = limit
            let records = (try? context.fetch(descriptor)) ?? []
            return records.compactMap { try? decoder.decode(CorrectionEvent.self, from: $0.payload) }
        }
    }

    public func clear() {
        queue.sync {
            do {
                try context.delete(model: CorrectionRecord.self)
                try context.save()
                AppLogger.corrections.info("Durable correction log cleared")
            } catch {
                AppLogger.corrections.error("Durable correction clear failed: \(error.localizedDescription, privacy: .public)")
            }
        }
    }

    public var approximateCount: Int {
        queue.sync {
            (try? context.fetchCount(FetchDescriptor<CorrectionRecord>())) ?? 0
        }
    }

    // MARK: - Outbox helpers (Phase 4) — concrete, not on the protocol

    /// Oldest-first batch of records still awaiting upload (`.pending`).
    public func pending(limit: Int) -> [CorrectionEvent] {
        queue.sync {
            let pendingRaw = CorrectionSyncStatus.pending.rawValue
            var descriptor = FetchDescriptor<CorrectionRecord>(
                predicate: #Predicate { $0.syncStatusRaw == pendingRaw },
                sortBy: [SortDescriptor(\.createdAt, order: .forward)]
            )
            descriptor.fetchLimit = limit
            let records = (try? context.fetch(descriptor)) ?? []
            return records.compactMap { try? decoder.decode(CorrectionEvent.self, from: $0.payload) }
        }
    }

    /// Reclaim records stuck in `.uploading` (an upload interrupted by app
    /// suspension/kill) back to `.pending` so the next sync retries them.
    /// Called at the start of a sync; within a process, overlapping syncs are
    /// already coalesced by the coordinator.
    public func requeueInFlight() {
        queue.async { [weak self] in
            guard let self else { return }
            let uploadingRaw = CorrectionSyncStatus.uploading.rawValue
            let pendingRaw = CorrectionSyncStatus.pending.rawValue
            let descriptor = FetchDescriptor<CorrectionRecord>(
                predicate: #Predicate { $0.syncStatusRaw == uploadingRaw }
            )
            guard let records = try? self.context.fetch(descriptor), !records.isEmpty else { return }
            for record in records { record.syncStatusRaw = pendingRaw }
            try? self.context.save()
        }
    }

    /// Current sync status of a record, or nil if unknown.
    public func status(forId id: UUID) -> CorrectionSyncStatus? {
        queue.sync {
            var descriptor = FetchDescriptor<CorrectionRecord>(
                predicate: #Predicate { $0.id == id }
            )
            descriptor.fetchLimit = 1
            guard let record = (try? context.fetch(descriptor))?.first else { return nil }
            return CorrectionSyncStatus(rawValue: record.syncStatusRaw)
        }
    }

    /// Transition a record's sync status. Moving to `.uploading` increments the
    /// attempt counter. No-op if the id is unknown.
    public func markStatus(_ status: CorrectionSyncStatus, forId id: UUID, error: String? = nil) {
        queue.async { [weak self] in
            guard let self else { return }
            var descriptor = FetchDescriptor<CorrectionRecord>(
                predicate: #Predicate { $0.id == id }
            )
            descriptor.fetchLimit = 1
            guard let record = (try? self.context.fetch(descriptor))?.first else { return }
            record.syncStatusRaw = status.rawValue
            if status == .uploading { record.attemptCount += 1 }
            record.lastError = error
            do {
                try self.context.save()
            } catch {
                AppLogger.corrections.error("Durable correction status update failed: \(error.localizedDescription, privacy: .public)")
            }
        }
    }
}
