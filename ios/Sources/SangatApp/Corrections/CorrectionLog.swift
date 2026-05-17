//
//  CorrectionLog.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Protocol for the correction event store. The protocol is stable from
//  M5.1 onward — feature work in M5.6 will write through this surface, so
//  swapping the no-op shipped today for a real JSONL implementation later
//  requires zero changes in feature code.

import Foundation

/// Append-only store for `CorrectionEvent`s. Implementations must be safe
/// to call from any actor; the protocol is `Sendable` and methods do not
/// require main-actor isolation.
public protocol CorrectionLog: AnyObject, Sendable {

    /// Record a correction. Returns immediately; persistence may be async
    /// internally.
    func record(_ event: CorrectionEvent)

    /// Most recent N events, ordered newest first.
    func recent(limit: Int) -> [CorrectionEvent]

    /// Wipe all stored events. Reversible-by-the-user only from Settings.
    func clear()

    /// Approximate count of stored events. Cheap; used by Settings to show
    /// "N corrections saved on this device".
    var approximateCount: Int { get }
}
