//
//  RootView.swift
//  GurbaniCaptioningApp · SangatApp · App
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//  Updated in M5.2 to route into the real Sangat reading surface.
//
//  Top-level route. Three branches:
//
//    1. First launch — `!hasCompletedOnboarding`  →  OnboardingPlaceholder
//       (M5.4 will replace with the real onboarding flow.)
//
//    2. Idle — no listening session yet           →  IdleView
//       Shown after onboarding completes and after `captionModel.stop()`.
//       The big "Listen" button is the only affordance.
//
//    3. Running — listening / tentative / committed  →  ReadingHost
//       Dispatches by engine state + user reading-layout preference.
//
//  M5.2 removes the M5.1 auto-start: caption source is `prepare()`-d on
//  appear but no longer auto-`start()`-ed. The user taps Listen in
//  `IdleView` to begin a session.

import SwiftUI
import GurbaniCaptioning

public struct RootView: View {

    @State private var env: AppEnvironment
    @State private var showSettings: Bool = false
    @State private var showShabadPicker: Bool = false
    @State private var showCastHint: Bool = false

    /// M5.6: predicted shabad id captured when the user long-presses
    /// a reading view. Non-nil drives the `WrongShabadSheet`
    /// presentation; nil means no sheet. Wrapped in an `Identifiable`
    /// shim because SwiftUI's `.sheet(item:)` requires it.
    @State private var pendingWrongShabadPrediction: PendingShabad?

    /// Per-session gate flipped by the user tapping Begin on
    /// `SessionStartView`. Resets to false on every cold start since
    /// it's plain `@State`, which is exactly the behavior we want:
    /// every cold start lands on Let's Begin before the Listen page.
    @State private var didStartSession: Bool = false

    /// Phase 7: drives an outbox drain when the app returns to the foreground.
    @Environment(\.scenePhase) private var scenePhase

    /// Owns the external-display UIWindow + observation. `@State` so
    /// it survives RootView body re-renders; lazily initialized in
    /// `.task` once the env is fully wired (a UIScreen may already be
    /// connected at app launch in the simulator).
    @State private var castCoordinator: CastSceneCoordinator?

    public init(env: AppEnvironment? = nil) {
        _env = State(initialValue: env ?? AppEnvironment.production())
    }

    public var body: some View {
        ZStack {
            env.theme.tokens.colors.bg.ignoresSafeArea()

            currentScreen
                .transition(.opacity)
        }
        .preferredColorScheme(env.theme.isDark ? .dark : .light)
        .safeAreaInset(edge: .top, spacing: 0) {
            // Top chrome row — Back (when running) + Settings gear. Lives
            // in the safe-area inset so the reading view's content always
            // sits below the chrome and the meta header never collides
            // with the corner icons. Onboarding + Let's Begin are
            // deliberately chrome-free to keep those moments gentle.
            if env.hasCompletedOnboarding && didStartSession {
                topChromeRow
            }
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            // Sevadar dock — overlays the reading view from below when
            // the user is in sevadar mode AND a session is running. The
            // safe-area inset means the reading view content lays out
            // above the dock automatically; no Sangat view is modified.
            if shouldShowSevadarDock {
                sevadarDock
            }
        }
        // Environment modifiers MUST come after the safeAreaInset calls.
        // SwiftUI wraps the modified view with each modifier in order;
        // the inset's content closure is a sibling of the wrapped ZStack,
        // so .environment placed BEFORE the insets never reached the
        // dock's `@Environment(\.themeTokens)` lookup — the dock fell
        // back to Theme.default.tokens (paper) even when env.theme was
        // darbar or mool. Moving environment modifiers downstream of the
        // insets wraps everything including the inset content.
        .environment(\.theme, env.theme)
        .environment(\.themeTokens, env.theme.tokens)
        .environment(env)
        .sheet(isPresented: $showSettings) {
            SettingsView()
                .environment(env)
                .environment(\.theme, env.theme)
                .environment(\.themeTokens, env.theme.tokens)
                .preferredColorScheme(env.theme.isDark ? .dark : .light)
        }
        .sheet(isPresented: $showShabadPicker) {
            ShabadPickerView(
                viewModel: ShabadPickerViewModel(
                    nowPlayingShabadId: env.captionModel.committedShabadId,
                    recents: Array<ShabadPickerEntry>.demoRecents
                ),
                onPick: { shabadId in
                    env.haptics.play(.success)
                    // M5.6: emit hardNegPos when the manual pick
                    // disagrees with the current commit. Same picker
                    // also handles the Sangat-side wrong-shabad sheet
                    // via `WrongShabadSheet`, but THAT path goes
                    // through its own emission below (different
                    // predicted/runner-ups snapshot semantics).
                    recordSevadarPickerCorrectionIfMismatch(pickedShabadId: shabadId)
                    env.captionModel.manuallyCommit(shabadId: shabadId)
                }
            )
            .environment(env)
            .environment(\.theme, env.theme)
            .environment(\.themeTokens, env.theme.tokens)
            .preferredColorScheme(env.theme.isDark ? .dark : .light)
        }
        .sheet(item: $pendingWrongShabadPrediction) { pending in
            WrongShabadSheet(
                predictedShabadId: pending.shabadId,
                onCorrect: { correctedShabadId in
                    recordSangatWrongShabad(
                        predictedShabadId: pending.shabadId,
                        correctedShabadId: correctedShabadId
                    )
                    env.haptics.play(.success)
                    env.captionModel.manuallyCommit(shabadId: correctedShabadId)
                    pendingWrongShabadPrediction = nil
                }
            )
            .environment(env)
            .environment(\.theme, env.theme)
            .environment(\.themeTokens, env.theme.tokens)
            .preferredColorScheme(env.theme.isDark ? .dark : .light)
        }
        .sheet(isPresented: $showCastHint) {
            CastHintSheet(
                isConnected: castCoordinator?.isExternalScreenConnected ?? false
            )
            .environment(env)
            .environment(\.theme, env.theme)
            .environment(\.themeTokens, env.theme.tokens)
            .preferredColorScheme(env.theme.isDark ? .dark : .light)
            .presentationDetents([.medium])
            .presentationDragIndicator(.visible)
        }
        .task {
            await prepareCaptionSource()
            // Lazy-init the cast coordinator with the live env. Holds
            // a weak reference inside, so RootView retains ownership.
            if castCoordinator == nil {
                castCoordinator = CastSceneCoordinator(env: env)
            }
            // Phase 4: drain the corrections outbox on launch. No-op unless the
            // user opted in to upload and the device is online; safe to call
            // when `syncCoordinator` is nil (demo / in-memory store).
            await env.syncCoordinator?.sync()
        }
        .onChange(of: scenePhase) { _, newPhase in
            // Phase 7: drain the outbox when the app returns to the foreground.
            if newPhase == .active {
                Task { await env.syncCoordinator?.sync() }
            }
        }
        .onChange(of: env.mode) { _, newMode in
            // Cast is a Sevadar-only surface (per onboarding role copy
            // and SevadarDock's sole `onCast` affordance). When a user
            // flips role mid-session, the projector window must follow:
            // detach on flip to Sangat, re-attach on flip to Sevadar if
            // an external screen is still physically connected.
            castCoordinator?.modeDidChange(to: newMode)
        }
        .animation(
            .easeInOut(duration: 0.25),
            value: env.hasCompletedOnboarding
        )
        .animation(
            .easeInOut(duration: 0.25),
            value: env.captionModel.isRunning
        )
    }

    @ViewBuilder
    private var currentScreen: some View {
        if !env.hasCompletedOnboarding {
            OnboardingFlow()
        } else if !didStartSession {
            // "Let's Begin" — per-session entry screen shown on every
            // cold start once first-time onboarding has been completed.
            // Tapping Begin flips `didStartSession` to true for the
            // rest of this app process lifetime AND rolls a fresh
            // `env.sessionId` so M5.6 correction events stamp the
            // current sitting consistently.
            SessionStartView {
                env.startNewSession()
                didStartSession = true
            }
        } else if !env.captionModel.isRunning {
            IdleView()
        } else {
            ReadingHost(
                onRequestPicker: { showShabadPicker = true },
                onRequestWrongShabad: { predicted in
                    pendingWrongShabadPrediction = PendingShabad(shabadId: predicted)
                }
            )
        }
    }

    private var topChromeRow: some View {
        HStack(spacing: 0) {
            if env.captionModel.isRunning {
                backButton
            } else {
                // Reserve symmetric space so the gear stays anchored to
                // the right edge whether or not back is shown.
                Color.clear.frame(width: 44, height: 44)
            }

            Spacer()

            // Current-mode tag in the center of the chrome row so it
            // is visible on every post-onboarding screen — matches the
            // design canvas's `V1MetaHeader` strip (sangat dot in
            // saffron, sevadar in indigo). Without this the only place
            // the mode is surfaced is Let's Begin, which is invisible
            // mid-session.
            modeChip

            // "Casting" indicator appears immediately to the right of
            // the mode chip when the cast coordinator has attached to
            // an external screen. Subtle by design — it's a status
            // confirmation, not an action.
            if castCoordinator?.isExternalScreenConnected == true {
                castingIndicator
                    .padding(.leading, env.theme.tokens.spacing.sm)
            }

            Spacer()

            settingsGearButton
        }
        .padding(.horizontal, env.theme.tokens.spacing.edge - 10)
        .padding(.top, env.theme.tokens.spacing.xs)
        .padding(.bottom, env.theme.tokens.spacing.xs)
    }

    private var modeChip: some View {
        HStack(spacing: env.theme.tokens.spacing.xs) {
            Circle()
                .fill(env.mode == .sevadar
                      ? env.theme.tokens.colors.sevadar
                      : env.theme.tokens.colors.accent)
                .frame(width: 6, height: 6)
            Text(env.mode == .sevadar ? "Sevadar" : "Sangat")
                .font(env.theme.tokens.type.sansCaps)
                .tracking(0.6)
                .foregroundStyle(env.theme.tokens.colors.ink3)
        }
        .accessibilityIdentifier("root.modeChip")
    }

    /// Visible only while `castCoordinator.isExternalScreenConnected`.
    /// Uses the theme's `accent` (saffron / wheat-gold / terracotta)
    /// to match the cast view's own "Live · Casting from iPhone"
    /// indicator — same dot color in both surfaces.
    private var castingIndicator: some View {
        HStack(spacing: env.theme.tokens.spacing.xs) {
            Circle()
                .fill(env.theme.tokens.colors.accent)
                .frame(width: 6, height: 6)
            Text("Casting")
                .font(env.theme.tokens.type.sansCaps)
                .tracking(0.6)
                .foregroundStyle(env.theme.tokens.colors.ink3)
        }
        .accessibilityIdentifier("root.castingIndicator")
        .transition(.opacity)
    }

    private var backButton: some View {
        Button {
            env.haptics.play(.selection)
            env.captionModel.stop()
        } label: {
            Image(systemName: "chevron.left")
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(env.theme.tokens.colors.ink3)
                .frame(width: 44, height: 44)
                .contentShape(Rectangle())
        }
        .accessibilityLabel("Stop listening")
        .accessibilityIdentifier("root.back")
    }

    private var settingsGearButton: some View {
        Button {
            env.haptics.play(.selection)
            showSettings = true
        } label: {
            Image(systemName: "gearshape")
                .font(.system(size: 22, weight: .regular))
                .foregroundStyle(env.theme.tokens.colors.ink3)
                .frame(width: 44, height: 44)
                .contentShape(Rectangle())
        }
        .accessibilityLabel("Settings")
        .accessibilityIdentifier("root.settings")
    }

    // MARK: - Sevadar dock

    private var shouldShowSevadarDock: Bool {
        // Listening + Tentative look identical for Sangat and Sevadar
        // by design — the dock is reserved for the committed state so
        // it doesn't add chrome to moments that haven't earned it yet.
        // Tentative's "Pick manually" affordance is the picker entry
        // before commit; the dock takes over after.
        env.hasCompletedOnboarding
            && didStartSession
            && env.mode == .sevadar
            && env.captionModel.isRunning
            && env.captionModel.isCommitted
    }

    private var sevadarDock: some View {
        SevadarDock(
            isPaused: env.captionModel.isPaused,
            layersSummary: layersSummary(
                layout: env.readingLayout,
                translit: env.translitEnabled,
                meaning: env.meaningEnabled
            ),
            onNudgeBack: {
                env.haptics.play(.selection)
                nudgeLine(by: -1)
            },
            onTogglePause: {
                env.haptics.play(.selection)
                if env.captionModel.isPaused {
                    env.captionModel.resume()
                } else {
                    env.captionModel.pause()
                }
            },
            onNudgeForward: {
                env.haptics.play(.selection)
                nudgeLine(by: 1)
            },
            onPick: {
                env.haptics.play(.selection)
                showShabadPicker = true
            },
            onCast: {
                // M5.5: Cast routing is owned by the OS (Control Center
                // → Screen Mirroring). Tapping this button can't *start*
                // casting; instead we surface a hint sheet so the user
                // knows where to find the system control. Once an
                // external screen connects, CastSceneCoordinator takes
                // over and the chrome row shows a "Casting" indicator.
                env.haptics.play(.selection)
                showCastHint = true
            },
            onEditLayers: {
                env.haptics.play(.selection)
                showSettings = true
            }
        )
    }

    /// Clamp the nudge target to the current shabad's line count, then
    /// hand the absolute index to the source via the existing
    /// `nudge(by:)` protocol method. `env.totalLines(forShabadId:)`
    /// owns the upper bound — the source clamps only at zero.
    ///
    /// M5.6 side effect: when the nudge actually moves the line, emit
    /// a `lineNudge` correction event so the smoother / loop-aligner
    /// has training signal. Gated on `correctionsOptIn` so the trust
    /// hinge is honored at every emission site.
    private func nudgeLine(by delta: Int) {
        guard let guess = env.captionModel.currentGuess else { return }
        let total = env.totalLines(forShabadId: guess.shabadId)
        let target = max(0, min(guess.lineIdx + delta, total - 1))
        let actualDelta = target - guess.lineIdx
        guard actualDelta != 0 else { return }
        recordLineNudge(
            shabadId: guess.shabadId,
            predictedLineIdx: guess.lineIdx,
            delta: actualDelta
        )
        env.captionModel.nudge(by: actualDelta)
    }

    // MARK: - Correction emission helpers (M5.6)

    /// Single source of truth for the "shall we record?" gate. Every
    /// emission site funnels through this so the toggle in
    /// `CorrectionsSettingsView` has one binding contract.
    private func recordIfOptedIn(_ build: () -> CorrectionEvent) {
        guard env.preferences.correctionsOptIn else {
            AppLogger.corrections.debug("Correction skipped — user has not opted in")
            return
        }
        env.correctionLog.record(build())
    }

    /// Phase 2b: snapshot recent mic audio for a correction when the user has
    /// opted in to audio capture *and* the live engine can provide it. Returns
    /// the local clip path, or nil (demo source / opted-out / no audio). Only
    /// called from inside `recordIfOptedIn`, so it never produces an orphan clip
    /// for a correction we don't record. Encode is synchronous here (rare,
    /// user-initiated tap); can move off-main later if it ever hitches.
    private func captureCorrectionClip(forId id: UUID) -> String? {
        CorrectionAudioCapture.capture(
            id: id,
            optedIn: env.preferences.audioCaptureOptIn,
            capturer: env.captionSource as? AudioClipCapturing,
            writer: env.audioClipWriter
        )
    }

    /// Sevadar picker → manualCommit path. Only emits when the picked
    /// shabad differs from whatever the engine currently has — a same-
    /// shabad pick is a no-op confirmation, not a correction.
    private func recordSevadarPickerCorrectionIfMismatch(pickedShabadId: Int) {
        guard let predictedId = env.captionModel.committedShabadId,
              predictedId != pickedShabadId else { return }
        let currentGuess = env.captionModel.currentGuess
        let runnerUps = Dictionary(
            uniqueKeysWithValues: env.captionModel.runnerUps.map { ($0.shabadId, $0.confidence) }
        )
        recordIfOptedIn {
            let eventId = UUID()
            let audioPath = captureCorrectionClip(forId: eventId)
            return CorrectionEventBuilder.makeHardNegPos(
                sessionId: env.sessionId,
                predictedShabadId: predictedId,
                predictedLineIdx: currentGuess?.lineIdx,
                predictedConfidence: currentGuess?.confidence,
                runnerUps: runnerUps,
                correctedShabadId: pickedShabadId,
                engineStateRaw: CorrectionEventBuilder.engineStateRaw(env.captionModel.state),
                audioBufferPath: audioPath,
                id: eventId
            )
        }
    }

    /// Sangat long-press → WrongShabadSheet path. Predicted snapshot
    /// is captured at long-press time, supplied by the sheet's host.
    private func recordSangatWrongShabad(predictedShabadId: Int, correctedShabadId: Int) {
        guard predictedShabadId != correctedShabadId else { return }
        let currentGuess = env.captionModel.currentGuess
        let runnerUps = Dictionary(
            uniqueKeysWithValues: env.captionModel.runnerUps.map { ($0.shabadId, $0.confidence) }
        )
        recordIfOptedIn {
            let eventId = UUID()
            let audioPath = captureCorrectionClip(forId: eventId)
            return CorrectionEventBuilder.makeHardNegPos(
                sessionId: env.sessionId,
                predictedShabadId: predictedShabadId,
                predictedLineIdx: currentGuess?.lineIdx,
                predictedConfidence: currentGuess?.confidence,
                runnerUps: runnerUps,
                correctedShabadId: correctedShabadId,
                engineStateRaw: CorrectionEventBuilder.engineStateRaw(env.captionModel.state),
                audioBufferPath: audioPath,
                id: eventId
            )
        }
    }

    /// Sevadar dock ± → smoother training signal.
    private func recordLineNudge(shabadId: Int, predictedLineIdx: Int, delta: Int) {
        recordIfOptedIn {
            CorrectionEventBuilder.makeLineNudge(
                sessionId: env.sessionId,
                shabadId: shabadId,
                predictedLineIdx: predictedLineIdx,
                delta: delta,
                engineStateRaw: CorrectionEventBuilder.engineStateRaw(env.captionModel.state)
            )
        }
    }

    private func layersSummary(layout: ReadingLayout, translit: Bool, meaning: Bool) -> String {
        let layoutName: String
        switch layout {
        case .hero: layoutName = "Hero"
        case .karaoke: layoutName = "Karaoke"
        case .full: layoutName = "Full"
        }
        let t = translit ? "Translit on" : "Translit off"
        let m = meaning ? "Meaning on" : "Meaning off"
        return "\(layoutName) · \(t) · \(m)"
    }

    /// Prepare the caption source on appear — no automatic `start()` in
    /// M5.2; the user initiates a session by tapping Listen in `IdleView`.
    private func prepareCaptionSource() async {
        do {
            try await env.captionModel.prepare()
        } catch {
            AppLogger.app.error(
                "RootView: caption source prepare failed — \(error.localizedDescription, privacy: .public)"
            )
        }
    }
}

#Preview("RootView · idle (paper)") {
    let env = AppEnvironment.preview()
    RootView(env: env)
}

#Preview("RootView · idle (darbar)") {
    let env = AppEnvironment.preview(theme: .darbar)
    RootView(env: env)
}

#Preview("RootView · onboarding (mool)") {
    let env = AppEnvironment.preview(
        theme: .mool,
        hasCompletedOnboarding: false
    )
    RootView(env: env)
}

// MARK: - Identifiable shim for SwiftUI .sheet(item:)

/// `.sheet(item:)` needs an `Identifiable`; an `Optional<Int>` won't
/// do because two different `Int` values that happen to be equal
/// should re-present the sheet. Wrapping in a struct with a fresh
/// `UUID` per construction sidesteps that and keeps the call site
/// readable.
private struct PendingShabad: Identifiable {
    let id: UUID
    let shabadId: Int

    init(shabadId: Int) {
        self.id = UUID()
        self.shabadId = shabadId
    }
}
