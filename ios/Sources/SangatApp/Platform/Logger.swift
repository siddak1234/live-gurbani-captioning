//
//  Logger.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Thin namespace over `os.Logger`. We log via categorized loggers so
//  Console.app + log streams can be filtered by feature area. The subsystem
//  is `com.sangat.app` — keep it stable across the codebase; downstream
//  log search depends on it.
//
//  Usage:
//      AppLogger.app.info("Starting")
//      AppLogger.source.error("CaptionSource failed: \(error.localizedDescription, privacy: .public)")
//
//  Architectural note
//  ------------------
//  This is a *static utility namespace*, not a service singleton. It holds
//  no state and no business behavior; it is the standard pattern for `os.Logger`
//  in Apple's frameworks. The "no singletons" invariant (see audit checklist
//  A7) explicitly exempts this file.

import Foundation
import OSLog

/// Namespaced category-keyed loggers for the Sangat iOS app.
public enum AppLogger {
    /// Stable subsystem identifier — keep aligned with the app bundle id.
    public static let subsystem: String = "com.sangat.app"

    /// Application lifecycle, routing, and high-level state transitions.
    public static let app = Logger(subsystem: subsystem, category: "app")

    /// `CaptionSource` activity — start/stop, state transitions, demo timeline.
    public static let source = Logger(subsystem: subsystem, category: "source")

    /// Correction loop events — picks, nudges, retroactive labels.
    public static let corrections = Logger(subsystem: subsystem, category: "corrections")

    /// External display / AirPlay / cast scene lifecycle.
    public static let cast = Logger(subsystem: subsystem, category: "cast")

    /// View-level events that aren't worth a feature-specific category.
    public static let ui = Logger(subsystem: subsystem, category: "ui")
}
