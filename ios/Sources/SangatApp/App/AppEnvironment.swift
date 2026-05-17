//
//  AppEnvironment.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Single composition root. Holds every service the app uses; observable
//  state (theme, mode, onboarding) lives directly on the env so view
//  trees re-render on changes.
//
//  Construction
//  ------------
//  - `production()` — boots from `Preferences` + `Bundle`. Uses
//    `LiveCaptionSource` when wired, falls back to `DemoCaptionSource`
//    on corpus / model failure (so a broken bundle never bricks the app).
//  - `preview(...)` — fully in-memory, scriptable, no I/O. Used by every
//    SwiftUI `#Preview` and by tests.
//
//  Services are read directly off the env via `@Environment(AppEnvironment.self)`
//  in views. Mutable state changes (theme, mode) trigger Observation
//  re-renders without per-property `@Published`.

import Foundation
import Observation
import GurbaniCaptioning

@MainActor
@Observable
public final class AppEnvironment {

    // MARK: - Services (immutable, constructor-injected)

    @ObservationIgnored public let captionSource: any CaptionSource
    @ObservationIgnored public let captionModel: CaptionSourceModel
    @ObservationIgnored public let correctionLog: any CorrectionLog
    @ObservationIgnored public let preferences: Preferences
    @ObservationIgnored public let haptics: any HapticsService
    @ObservationIgnored public let sessionHistory: any SessionHistoryStore
    @ObservationIgnored public let featureFlags: FeatureFlags

    // MARK: - Observable state

    /// Active theme. Persists to `Preferences` on set.
    public var theme: Theme {
        didSet { preferences.theme = theme }
    }

    /// Active mode. Persists to `Preferences` on set.
    public var mode: AppMode {
        didSet { preferences.mode = mode }
    }

    /// Whether the user has completed first-launch onboarding.
    public var hasCompletedOnboarding: Bool {
        didSet { preferences.hasCompletedOnboarding = hasCompletedOnboarding }
    }

    /// Reading layout (hero / karaoke / full). Persists.
    public var readingLayout: ReadingLayout {
        didSet { preferences.readingLayout = readingLayout }
    }

    /// Show transliteration row in reading views. Persists.
    public var translitEnabled: Bool {
        didSet { preferences.translitEnabled = translitEnabled }
    }

    /// Show English meaning in reading views. Persists.
    public var meaningEnabled: Bool {
        didSet { preferences.meaningEnabled = meaningEnabled }
    }

    // MARK: - Init

    public init(
        captionSource: any CaptionSource,
        correctionLog: any CorrectionLog,
        preferences: Preferences,
        haptics: any HapticsService,
        sessionHistory: any SessionHistoryStore,
        featureFlags: FeatureFlags
    ) {
        self.captionSource = captionSource
        self.captionModel = CaptionSourceModel(source: captionSource)
        self.correctionLog = correctionLog
        self.preferences = preferences
        self.haptics = haptics
        self.sessionHistory = sessionHistory
        self.featureFlags = featureFlags

        // Seed observable state from preferences (with safe defaults).
        self.theme = preferences.theme ?? .default
        self.mode = preferences.mode ?? .default
        self.hasCompletedOnboarding = preferences.hasCompletedOnboarding
        self.readingLayout = preferences.readingLayout ?? .default
        self.translitEnabled = preferences.translitEnabled
        self.meaningEnabled = preferences.meaningEnabled

        AppLogger.app.info("AppEnvironment booted — theme=\(self.theme.rawValue, privacy: .public) mode=\(self.mode.rawValue, privacy: .public) flags=\(String(describing: featureFlags), privacy: .public)")
    }

    // MARK: - Convenience composition

    /// Production environment: reads preferences, loads bundled corpus,
    /// uses the live source by default (falls back to demo on failure).
    public static func production() -> AppEnvironment {
        let preferences = Preferences()
        let flags = FeatureFlags.compileTime(preferences: preferences)
        let source: any CaptionSource = Self.makeCaptionSource(flags: flags)

        #if canImport(UIKit)
        let haptics: any HapticsService = UIKitHapticsService()
        #else
        let haptics: any HapticsService = NoopHapticsService()
        #endif

        return AppEnvironment(
            captionSource: source,
            correctionLog: NoopCorrectionLog(),
            preferences: preferences,
            haptics: haptics,
            sessionHistory: InMemorySessionHistoryStore(),
            featureFlags: flags
        )
    }

    /// In-memory environment for tests + previews. No bundle loads.
    public static func preview(
        captionSource: (any CaptionSource)? = nil,
        theme: Theme = .default,
        mode: AppMode = .sangat,
        hasCompletedOnboarding: Bool = true,
        flags: FeatureFlags = .allEnabled
    ) -> AppEnvironment {
        let prefs = Preferences.inMemory()
        prefs.theme = theme
        prefs.mode = mode
        prefs.hasCompletedOnboarding = hasCompletedOnboarding

        let source = captionSource ?? DemoCaptionSource(script: .quickCommit)
        return AppEnvironment(
            captionSource: source,
            correctionLog: NoopCorrectionLog(),
            preferences: prefs,
            haptics: NoopHapticsService(),
            sessionHistory: InMemorySessionHistoryStore(),
            featureFlags: flags
        )
    }

    // MARK: - Internal factory

    private static func makeCaptionSource(flags: FeatureFlags) -> any CaptionSource {
        if flags.useDemoSource {
            AppLogger.app.info("AppEnvironment using DemoCaptionSource")
            return DemoCaptionSource()
        }
        do {
            let corpus = try ShabadCorpus.loadFromBundle()
            let config = CaptionEngine.Config(
                modelPath: "surt-small-v3-kirtan",
                language: "punjabi",
                chunkSeconds: 5.0
            )
            return LiveCaptionSource(corpus: corpus, config: config)
        } catch {
            AppLogger.app.error("AppEnvironment failed to construct LiveCaptionSource — \(error.localizedDescription, privacy: .public). Falling back to DemoCaptionSource so the app remains usable.")
            return DemoCaptionSource()
        }
    }
}
