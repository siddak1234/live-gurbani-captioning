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

    /// Writer for correction audio clips (corrections feedback loop Phase 2).
    /// `nil` if the clip store couldn't be created or in previews/tests.
    /// Used by the correction-emission sites when `audioCaptureOptIn` is on.
    @ObservationIgnored public let audioClipWriter: AudioClipWriter?

    /// Outbox sync coordinator (corrections feedback loop Phase 4). Drains
    /// pending durable corrections to Supabase when `uploadOptIn` is on and the
    /// device is online. `nil` when there is no durable store (in-memory
    /// fallback) or in previews/tests.
    @ObservationIgnored public let syncCoordinator: SyncCoordinator?

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
    ///
    /// Persisted to `Preferences` so subsequent cold starts skip the
    /// 4-card onboarding flow. The brief session-only window between
    /// commits `b63bd38` and the M5.4 landing was a stopgap when Welcome
    /// was a content-free placeholder; once onboarding asks for mic
    /// permission and picks a role, re-asking every launch is hostile.
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

    // MARK: - Session identity (M5.6)

    /// Identifier for the current "session" — the span between a
    /// Let's Begin tap and either the next Let's Begin or app exit.
    /// Stamped onto every `CorrectionEvent` so the off-device fine-
    /// tune pipeline can group user corrections by sitting.
    ///
    /// Seeded at init so the field is always populated (the type is
    /// non-optional `UUID`), and re-rolled by `startNewSession()`
    /// when RootView observes the Let's Begin transition.
    ///
    /// Not persisted to `Preferences` — a session is by definition
    /// in-memory; cold launches always start a fresh one.
    public private(set) var sessionId: UUID = UUID()

    /// Roll a fresh `sessionId`. Called by RootView when the user
    /// taps Let's Begin (the `didStartSession` false→true transition).
    /// Idempotent for callers — calling twice in a row generates two
    /// different ids, which is the intended semantics.
    public func startNewSession() {
        sessionId = UUID()
        AppLogger.app.info("AppEnvironment: started new session \(self.sessionId.uuidString, privacy: .public)")
    }

    // MARK: - Init

    public init(
        captionSource: any CaptionSource,
        correctionLog: any CorrectionLog,
        preferences: Preferences,
        haptics: any HapticsService,
        sessionHistory: any SessionHistoryStore,
        featureFlags: FeatureFlags,
        audioClipWriter: AudioClipWriter? = nil,
        syncCoordinator: SyncCoordinator? = nil
    ) {
        self.captionSource = captionSource
        self.captionModel = CaptionSourceModel(source: captionSource)
        self.correctionLog = correctionLog
        self.preferences = preferences
        self.haptics = haptics
        self.sessionHistory = sessionHistory
        self.featureFlags = featureFlags
        self.audioClipWriter = audioClipWriter
        self.syncCoordinator = syncCoordinator

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

        // Phase 1 (corrections feedback loop): durable SwiftData store so
        // opted-in corrections survive app launches and form the upload outbox.
        // Falls back to the session-only in-memory log if the store can't open,
        // so a broken store never bricks the app. Nothing is written unless the
        // user opts in (`correctionsOptIn`).
        let correctionLog = makeCorrectionLog()
        // Phase 2: clip store for opted-in correction audio (empty dir until a
        // capture happens under `audioCaptureOptIn`).
        let audioClipWriter = try? AudioClipWriter.makeDefault()
        // Phase 4: outbox sync coordinator (nil if no durable store). Nothing
        // uploads unless `uploadOptIn` is on and the device is online. Phase 6a:
        // also uploads the audio clip (when present) to Storage.
        let syncCoordinator = makeSyncCoordinator(
            correctionLog: correctionLog,
            preferences: preferences,
            audioClipWriter: audioClipWriter
        )

        return AppEnvironment(
            captionSource: source,
            correctionLog: correctionLog,
            preferences: preferences,
            haptics: haptics,
            sessionHistory: InMemorySessionHistoryStore(),
            featureFlags: flags,
            audioClipWriter: audioClipWriter,
            syncCoordinator: syncCoordinator
        )
    }

    /// Build the outbox sync coordinator when a durable store is available.
    /// Returns nil for the in-memory fallback (nothing durable to sync) or in
    /// previews/tests, so those paths never reach the network.
    private static func makeSyncCoordinator(
        correctionLog: any CorrectionLog,
        preferences: Preferences,
        audioClipWriter: AudioClipWriter?
    ) -> SyncCoordinator? {
        guard let durable = correctionLog as? DurableCorrectionLog else { return nil }
        return SyncCoordinator(
            log: durable,
            uploader: SupabaseRESTUploader(),
            reachability: NetworkReachability(),
            preferences: preferences,
            deviceId: DeviceIdentity().deviceId,
            audioUploader: SupabaseStorageUploader(),
            audioWriter: audioClipWriter
        )
    }

    /// In-memory environment for tests + previews. No bundle loads.
    ///
    /// `correctionLog` defaults to `NoopCorrectionLog`; tests can pass
    /// a `CorrectionLogSpy` (defined in `SangatAppTests`) to inspect
    /// emissions through the M5.6 wiring without touching the
    /// production seam.
    public static func preview(
        captionSource: (any CaptionSource)? = nil,
        correctionLog: (any CorrectionLog)? = nil,
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
            // M5.6.x: preview default mirrors production. Tests that
            // want to inspect emissions inject `CorrectionLogSpy`
            // explicitly via the `correctionLog:` parameter; tests
            // that want absolute discard can pass `NoopCorrectionLog()`.
            correctionLog: correctionLog ?? InMemoryCorrectionLog(),
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
            // M5.6.x: pass the PreviewData-backed line count so the
            // demo source's synthetic auto-advance (post-manualCommit)
            // wraps at the correct line count per shabad. The
            // production call goes through PreviewData rather than
            // the env's `totalLines(forShabadId:)` to avoid the env-
            // is-not-yet-built chicken-and-egg.
            return DemoCaptionSource(
                totalLinesProvider: { PreviewData.lineCount(forShabadId: $0) }
            )
        }
        do {
            let corpus = try ShabadCorpus.loadFromBundle()
            let config = CaptionEngine.Config(
                modelPath: "surt-small-v3-kirtan",
                // ISO-639-1 short code only — `"punjabi"` silently falls
                // back to English in WhisperKit (see CaptionEngine.Config
                // doc-comment for why).
                language: "pa",
                chunkSeconds: 5.0
            )
            return LiveCaptionSource(corpus: corpus, config: config)
        } catch {
            AppLogger.app.error("AppEnvironment failed to construct LiveCaptionSource — \(error.localizedDescription, privacy: .public). Falling back to DemoCaptionSource so the app remains usable.")
            return DemoCaptionSource(
                totalLinesProvider: { PreviewData.lineCount(forShabadId: $0) }
            )
        }
    }

    /// Durable on-device correction store (Phase 1). Falls back to the
    /// session-only in-memory log if SwiftData can't open the store, mirroring
    /// the caption-source fallback philosophy — a broken store never bricks the
    /// app. Behavior-neutral by default because emission is gated on
    /// `correctionsOptIn` (default off).
    private static func makeCorrectionLog() -> any CorrectionLog {
        do {
            let log = try DurableCorrectionLog.makeDefault()
            AppLogger.corrections.info("AppEnvironment using DurableCorrectionLog (SwiftData)")
            return log
        } catch {
            AppLogger.corrections.error("AppEnvironment: durable correction store unavailable — \(error.localizedDescription, privacy: .public). Falling back to InMemoryCorrectionLog.")
            return InMemoryCorrectionLog()
        }
    }
}
