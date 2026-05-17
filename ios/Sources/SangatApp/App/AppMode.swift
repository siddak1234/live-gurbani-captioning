//
//  AppMode.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Top-level "who is using this device right now" mode. Drives whether
//  Sevadar-only surfaces (dock, picker, confidence panel) are visible.
//
//  M5.1 ships the enum + persistence facade only. M5.3 will add the
//  mode-toggle UI and unlock semantics. We keep `requiresUnlock` here so
//  future tightening (PIN, biometric) only changes this file.

import Foundation

/// Who is using this device — drives Sevadar-only surfaces.
public enum AppMode: String, CaseIterable, Identifiable, Codable, Sendable {

    /// Sangat. Default. Quiet reading interface, big type, almost no
    /// controls beyond reading-layer choice.
    case sangat

    /// Sevadar. Operator surfaces available: dock, manual picker, engine
    /// confidence, session history.
    case sevadar

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .sangat:  return "Sangat"
        case .sevadar: return "Sevadar"
        }
    }

    public var gurmukhi: String {
        switch self {
        case .sangat:  return "ਸੰਗਤ"
        case .sevadar: return "ਸੇਵਾਦਾਰ"
        }
    }

    public var subtitle: String {
        switch self {
        case .sangat:
            return "Read along during kirtan. Quiet interface, big type."
        case .sevadar:
            return "Operator mode. Pick shabads, nudge lines, cast, see confidence."
        }
    }

    /// Whether switching *into* this mode should require an unlock step.
    /// M5.1: false for both. Tightening later only changes this file.
    public var requiresUnlock: Bool {
        switch self {
        case .sangat:  return false
        case .sevadar: return false
        }
    }

    public static let `default`: AppMode = .sangat
}
