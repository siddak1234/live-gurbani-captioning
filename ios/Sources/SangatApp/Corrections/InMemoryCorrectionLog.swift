//
//  InMemoryCorrectionLog.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.6.x (carry-forward fix).
//
//  Session-scoped `CorrectionLog`. Stores events in an `NSLock`-guarded
//  array, evaporates on app kill — same trust contract as
//  `InMemorySessionHistoryStore`. Lets the Settings → "Improve detection"
//  count reflect real activity within a sitting; durable persistence
//  across launches remains M5.8 work.
//
//  Why ship this in M5.6.x rather than waiting for M5.8: the M5.6 UI
//  surfaces a count + a Clear button. With `NoopCorrectionLog` both
//  controls are visually broken — count stays at 0 forever and Clear
//  is permanently disabled. That UX dead-end was the original M5.6
//  intent leaking past the implementation. This impl restores
//  feedback without making any new persistence commitments.

import Foundation

public final class InMemoryCorrectionLog: CorrectionLog, @unchecked Sendable {

    private let lock = NSLock()
    private var events: [CorrectionEvent] = []

    public init() {}

    public func record(_ event: CorrectionEvent) {
        lock.lock(); defer { lock.unlock() }
        events.append(event)
        AppLogger.corrections.info("InMemory correction recorded: kind=\(event.kind.rawValue, privacy: .public) predicted=\(event.predicted.shabadId, privacy: .public) truth=\(event.groundTruth.shabadId, privacy: .public) count=\(self.events.count, privacy: .public)")
    }

    public func recent(limit: Int) -> [CorrectionEvent] {
        lock.lock(); defer { lock.unlock() }
        return Array(events.suffix(limit).reversed())
    }

    public func clear() {
        lock.lock(); defer { lock.unlock() }
        events.removeAll()
        AppLogger.corrections.info("InMemory correction log cleared")
    }

    public var approximateCount: Int {
        lock.lock(); defer { lock.unlock() }
        return events.count
    }
}
