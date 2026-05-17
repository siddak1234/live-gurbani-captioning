//
//  SessionHistoryStore.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Append-only store for "shabads heard today". Backs the Sevadar history
//  screen. Real persistence (file or SQLite) lives behind the protocol —
//  M5.1 ships only the in-memory implementation so the UI can demo and we
//  defer the persistence decision (which is partly a trust question — do
//  we save which shabads users listened to?).

import Foundation

// MARK: - Entry

/// One row in the session history.
public struct SessionEntry: Identifiable, Sendable, Equatable {
    public let id: UUID
    public let timestamp: Date
    public let shabadId: Int
    public let firstLineGurmukhi: String
    public let ang: Int
    public let durationSeconds: TimeInterval
    public let endedReason: EndedReason

    public init(
        id: UUID = UUID(),
        timestamp: Date,
        shabadId: Int,
        firstLineGurmukhi: String,
        ang: Int,
        durationSeconds: TimeInterval,
        endedReason: EndedReason
    ) {
        self.id = id
        self.timestamp = timestamp
        self.shabadId = shabadId
        self.firstLineGurmukhi = firstLineGurmukhi
        self.ang = ang
        self.durationSeconds = durationSeconds
        self.endedReason = endedReason
    }

    public enum EndedReason: String, Sendable, Codable {
        case userReset
        case userManualPicked
        case engineCommittedNew
        case appBackgrounded
    }
}

// MARK: - Protocol

/// Append-only store for `SessionEntry` values. Implementations must be
/// thread-safe; `append` may be called from any actor.
public protocol SessionHistoryStore: AnyObject, Sendable {
    func append(_ entry: SessionEntry)
    func recent(limit: Int) -> [SessionEntry]
    func clear()
}

// MARK: - In-memory implementation

public final class InMemorySessionHistoryStore: SessionHistoryStore, @unchecked Sendable {

    private let lock = NSLock()
    private var entries: [SessionEntry] = []

    public init() {}

    public func append(_ entry: SessionEntry) {
        lock.lock()
        entries.append(entry)
        lock.unlock()
        AppLogger.app.info("Session history: appended shabad #\(entry.shabadId, privacy: .public)")
    }

    public func recent(limit: Int) -> [SessionEntry] {
        lock.lock()
        defer { lock.unlock() }
        if entries.count <= limit { return entries.reversed() }
        return Array(entries.suffix(limit).reversed())
    }

    public func clear() {
        lock.lock()
        entries.removeAll()
        lock.unlock()
    }
}
