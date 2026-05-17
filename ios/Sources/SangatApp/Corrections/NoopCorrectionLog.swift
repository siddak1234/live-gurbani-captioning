//
//  NoopCorrectionLog.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Default `CorrectionLog` implementation: accepts events, discards them.
//  Used until the M5.6 audit gate confirms the persistence + opt-in design.
//
//  Why ship a no-op?
//  -----------------
//  The correction *touchpoints* (long-press menu, picker, dock) ship in
//  M5.6 — earlier than the persistence layer. Wiring them against a stable
//  protocol means we can build, test, and review the UI without making the
//  data-retention decision yet. When the real implementation lands, only
//  one line in `AppEnvironment` changes (the constructor injection).

import Foundation

public final class NoopCorrectionLog: CorrectionLog, @unchecked Sendable {

    public init() {}

    public func record(_ event: CorrectionEvent) {
        AppLogger.corrections.info("Noop correction recorded: kind=\(event.kind.rawValue, privacy: .public) predicted=\(event.predicted.shabadId, privacy: .public) truth=\(event.groundTruth.shabadId, privacy: .public)")
    }

    public func recent(limit: Int) -> [CorrectionEvent] {
        []
    }

    public func clear() {
        // nothing to clear
    }

    public var approximateCount: Int { 0 }
}
