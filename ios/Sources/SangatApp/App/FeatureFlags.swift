//
//  FeatureFlags.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Compile-time + runtime feature gates. Today everything is compile-time;
//  the `Preferences`-backed overrides ride on the same struct so future
//  runtime gates (remote config, A/B tests) plug in without churning call
//  sites.

import Foundation

/// Snapshot of which features are enabled for this build/run.
///
/// Read at app launch in `AppEnvironment` and held immutable for the
/// session. Settings → Developer (M5.6) can flip overrides into
/// `Preferences`; the change applies on next launch.
public struct FeatureFlags: Sendable, Equatable {

    /// Whether the M5.6 correction-loop touchpoints are visible.
    public let correctionLoopEnabled: Bool

    /// Whether cast / AirPlay handling is active.
    public let castEnabled: Bool

    /// Whether the developer menu (Settings → Developer) is visible.
    public let developerMenuEnabled: Bool

    /// Use `DemoCaptionSource` rather than `LiveCaptionSource`.
    /// Default true in DEBUG so the app always boots without the model.
    public let useDemoSource: Bool

    public init(
        correctionLoopEnabled: Bool,
        castEnabled: Bool,
        developerMenuEnabled: Bool,
        useDemoSource: Bool
    ) {
        self.correctionLoopEnabled = correctionLoopEnabled
        self.castEnabled = castEnabled
        self.developerMenuEnabled = developerMenuEnabled
        self.useDemoSource = useDemoSource
    }

    // MARK: - Build-time resolution

    /// Read defaults from the current build configuration.
    @MainActor
    public static func compileTime(preferences: Preferences? = nil) -> FeatureFlags {
        let demoOverride = preferences?.useDemoSource

        #if DEBUG
        return FeatureFlags(
            correctionLoopEnabled: false,   // M5.6 enables
            castEnabled: false,             // M5.5 enables
            developerMenuEnabled: true,
            useDemoSource: demoOverride ?? true
        )
        #else
        return FeatureFlags(
            correctionLoopEnabled: false,
            castEnabled: false,
            developerMenuEnabled: false,
            useDemoSource: demoOverride ?? false
        )
        #endif
    }

    /// Convenience for tests + previews.
    public static let allEnabled = FeatureFlags(
        correctionLoopEnabled: true,
        castEnabled: true,
        developerMenuEnabled: true,
        useDemoSource: true
    )

    public static let allDisabled = FeatureFlags(
        correctionLoopEnabled: false,
        castEnabled: false,
        developerMenuEnabled: false,
        useDemoSource: false
    )
}
