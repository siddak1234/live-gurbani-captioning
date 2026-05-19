//
//  CorrectionLogSpy.swift
//  SangatAppTests — M5.6 (Correction loop surfaces)
//
//  Lightweight `CorrectionLog` test double. Captures every recorded
//  event in-memory so wiring tests can assert "exactly one event of
//  kind X" without going through the production no-op.
//
//  Thread-safety mirrors the real protocol (which is `Sendable`): we
//  guard the array with a lock so concurrent records don't tear the
//  state. In practice all M5.6 emissions happen on the main actor.

import Foundation
@testable import SangatApp

final class CorrectionLogSpy: CorrectionLog, @unchecked Sendable {

    private let lock = NSLock()
    private var events: [CorrectionEvent] = []

    init() {}

    func record(_ event: CorrectionEvent) {
        lock.lock(); defer { lock.unlock() }
        events.append(event)
    }

    func recent(limit: Int) -> [CorrectionEvent] {
        lock.lock(); defer { lock.unlock() }
        return Array(events.suffix(limit).reversed())
    }

    func clear() {
        lock.lock(); defer { lock.unlock() }
        events.removeAll()
    }

    var approximateCount: Int {
        lock.lock(); defer { lock.unlock() }
        return events.count
    }

    /// Test helper: all recorded events in record order. Not part of
    /// the protocol — read directly from tests.
    var recorded: [CorrectionEvent] {
        lock.lock(); defer { lock.unlock() }
        return events
    }
}
