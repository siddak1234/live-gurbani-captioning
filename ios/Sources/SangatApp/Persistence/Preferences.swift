//
//  Preferences.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Thin facade over `UserDefaults` for app-level preferences. Keeps the
//  storage key strings in one place (no scattered `"theme"` literals across
//  the codebase) and gives us a single seam to swap in a different store
//  later (e.g. iCloud key-value, Keychain, or a SQLite-backed settings
//  table) without touching call sites.
//
//  Why not `@AppStorage` directly?
//  -------------------------------
//  `@AppStorage` is convenient for one-off SwiftUI bindings but it spreads
//  raw string keys throughout the view layer and ties storage to view
//  lifecycle. Centralizing keys here keeps the audit trail clean and lets
//  unit tests pass in an in-memory `UserDefaults` instance.

import Foundation

@MainActor
public final class Preferences {

    private let defaults: UserDefaults

    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    // MARK: - Keys (private; never leak into the view layer)

    private enum Key {
        static let theme = "preferences.theme"
        static let mode = "preferences.mode"
        static let hasCompletedOnboarding = "preferences.hasCompletedOnboarding"
        static let translitEnabled = "preferences.translitEnabled"
        static let meaningEnabled = "preferences.meaningEnabled"
        static let readingLayout = "preferences.readingLayout"
        static let useDemoSource = "preferences.useDemoSource"
        static let micPermissionAcknowledged = "preferences.micPermissionAcknowledged"
        static let correctionsOptIn = "preferences.correctionsOptIn"
    }

    // MARK: - Theme

    /// User-selected theme. `nil` means "not yet set" — use `Theme.default`.
    public var theme: Theme? {
        get {
            guard let raw = defaults.string(forKey: Key.theme) else { return nil }
            return Theme(rawValue: raw)
        }
        set { defaults.set(newValue?.rawValue, forKey: Key.theme) }
    }

    // MARK: - App mode (sangat / sevadar)

    public var mode: AppMode? {
        get {
            guard let raw = defaults.string(forKey: Key.mode) else { return nil }
            return AppMode(rawValue: raw)
        }
        set { defaults.set(newValue?.rawValue, forKey: Key.mode) }
    }

    // MARK: - Onboarding

    public var hasCompletedOnboarding: Bool {
        get { defaults.bool(forKey: Key.hasCompletedOnboarding) }
        set { defaults.set(newValue, forKey: Key.hasCompletedOnboarding) }
    }

    public var micPermissionAcknowledged: Bool {
        get { defaults.bool(forKey: Key.micPermissionAcknowledged) }
        set { defaults.set(newValue, forKey: Key.micPermissionAcknowledged) }
    }

    // MARK: - Reading layers

    /// Show transliteration row under Gurmukhi. Default true.
    public var translitEnabled: Bool {
        get { defaults.object(forKey: Key.translitEnabled) as? Bool ?? true }
        set { defaults.set(newValue, forKey: Key.translitEnabled) }
    }

    /// Show English meaning. Default true.
    public var meaningEnabled: Bool {
        get { defaults.object(forKey: Key.meaningEnabled) as? Bool ?? true }
        set { defaults.set(newValue, forKey: Key.meaningEnabled) }
    }

    // MARK: - Reading layout

    /// Hero / karaoke / full. `nil` means "not yet set" — use `.hero`.
    public var readingLayout: ReadingLayout? {
        get {
            guard let raw = defaults.string(forKey: Key.readingLayout) else { return nil }
            return ReadingLayout(rawValue: raw)
        }
        set { defaults.set(newValue?.rawValue, forKey: Key.readingLayout) }
    }

    // MARK: - Developer / feature flags

    /// When true, the app uses `DemoCaptionSource` instead of `LiveCaptionSource`.
    /// Defaults to true in DEBUG builds, false in release. The setter
    /// overrides for either build.
    public var useDemoSource: Bool {
        get {
            #if DEBUG
            return defaults.object(forKey: Key.useDemoSource) as? Bool ?? true
            #else
            return defaults.object(forKey: Key.useDemoSource) as? Bool ?? false
            #endif
        }
        set { defaults.set(newValue, forKey: Key.useDemoSource) }
    }

    // MARK: - Corrections

    /// Whether the user opted in to saving correction events on this device.
    /// Default false — must be an explicit user choice (audio retention is
    /// a trust hinge, see `docs/ios_app_architecture.md`).
    public var correctionsOptIn: Bool {
        get { defaults.bool(forKey: Key.correctionsOptIn) }
        set { defaults.set(newValue, forKey: Key.correctionsOptIn) }
    }
}

// MARK: - Reading layout

/// Reading view variants for the committed state.
public enum ReadingLayout: String, CaseIterable, Identifiable, Codable, Sendable {
    case hero
    case karaoke
    case full

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .hero:    return "Hero"
        case .karaoke: return "3-line karaoke"
        case .full:    return "Full shabad"
        }
    }

    public var subtitle: String {
        switch self {
        case .hero:    return "Single line, biggest"
        case .karaoke: return "Previous · current · next"
        case .full:    return "Whole shabad, auto-scroll"
        }
    }

    public static let `default`: ReadingLayout = .hero
}

// MARK: - Test helpers

extension Preferences {

    /// In-memory `Preferences` backed by a fresh `UserDefaults` suite.
    /// Use only in tests and previews — the suite name is unique per call.
    public static func inMemory() -> Preferences {
        let suiteName = "preferences.inmemory.\(UUID().uuidString)"
        // The `UserDefaults(suiteName:)` initializer can technically return
        // nil if the suite name is invalid; the random UUID we construct is
        // always valid, but we fall back to `.standard` for safety. The
        // tests verifying this path live in `PreferencesTests.swift`.
        let defaults = UserDefaults(suiteName: suiteName) ?? .standard
        return Preferences(defaults: defaults)
    }
}
