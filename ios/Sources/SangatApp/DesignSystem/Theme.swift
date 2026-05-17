//
//  Theme.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  The three visual directions explored in the design canvas:
//    - .paper  ("Reading Room") — warm gutka-paper ivory, saffron accent.
//    - .darbar ("Darbar Night") — calm dark, wheat accent. AirPlay-ready.
//    - .mool   ("Mool")         — editorial minimal, terracotta accent.
//
//  This enum is the *only* place themes are enumerated. Adding a fourth
//  direction means extending this case set and the matching tokens in
//  `ThemeTokens+Presets.swift`; every consumer reads from `tokens` so no
//  view needs to change.
//
//  Theme is `Codable` so it can be persisted via `@AppStorage` on the user's
//  preference, and `CaseIterable` so the Settings screen can list options.

import Foundation

/// Visual direction for the entire app surface.
///
/// Source of truth for color, typography, spacing, and radius tokens.
/// Read `tokens` to get the concrete `ThemeTokens` for the active theme.
public enum Theme: String, CaseIterable, Identifiable, Codable, Sendable {

    /// Warm gutka-paper ivory with deep brown-black ink and a saffron accent.
    /// Default reading mode — feels like a printed gutka.
    case paper

    /// Warm near-black with cream foreground and a wheat-gold accent.
    /// Tuned for low-light darbars and AirPlay onto a projector.
    case darbar

    /// Editorial minimal — near-white with sans Gurmukhi and a terracotta accent.
    /// The most modern of the three.
    case mool

    // MARK: - Metadata

    public var id: String { rawValue }

    /// Human-readable name shown in Settings.
    public var displayName: String {
        switch self {
        case .paper:  return "Reading Room"
        case .darbar: return "Darbar Night"
        case .mool:   return "Mool"
        }
    }

    /// One-line description shown beneath the name in Settings.
    public var subtitle: String {
        switch self {
        case .paper:  return "Warm paper, ivory ink, saffron accent"
        case .darbar: return "Dark, cream foreground, wheat accent"
        case .mool:   return "Editorial minimal, terracotta accent"
        }
    }

    /// Whether this theme uses a dark background — drives the system
    /// `colorScheme` preference and status bar style.
    public var isDark: Bool {
        switch self {
        case .darbar: return true
        case .paper, .mool: return false
        }
    }

    /// Concrete tokens for the theme. The single resolution point — all
    /// views read color/type/spacing/radii via this.
    public var tokens: ThemeTokens {
        switch self {
        case .paper:  return .paper
        case .darbar: return .darbar
        case .mool:   return .mool
        }
    }

    /// The default theme on first launch. Centralized so unit tests and
    /// previews can reference one place.
    public static let `default`: Theme = .paper
}
