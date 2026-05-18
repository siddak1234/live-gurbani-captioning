//
//  IdleView.swift
//  GurbaniCaptioningApp · SangatApp · Features/Sangat
//
//  Created for the Sangat iOS app, M5.2 (Sangat reading).
//
//  Pre-listening screen. Shown when the caption source is not yet running
//  and not yet committed. Big saffron "Listen" disc + reassuring copy
//  about on-device processing. Tap fires `captionModel.start()` which
//  kicks the source into `.listening`; RootView then routes to
//  `ListeningView`.

import SwiftUI

public struct IdleView: View {

    @Environment(AppEnvironment.self) private var env
    @Environment(\.themeTokens) private var tokens
    @Environment(\.scenePhase) private var scenePhase

    /// Live mic permission status. Seeded on appear, re-read whenever
    /// the app returns to the foreground (so flipping mic on/off in
    /// iOS Settings → Sangat reflects immediately), and after the
    /// user taps the enable pill. Drives the M5.4.1 "Enable
    /// microphone" affordance — visible only when status is
    /// `.notDetermined` (user tapped "Not now" in onboarding) or
    /// `.denied` (granted then revoked via Settings).
    @State private var micStatus: MicPermissionStatus = .notDetermined

    /// Set after a tap on the enable pill while a request is in-flight,
    /// so the pill can disable itself + show a transient state.
    @State private var requestInFlight: Bool = false

    public init() {}

    public var body: some View {
        // The Settings gear is hosted by RootView so it persists across
        // Idle / Listening / Tentative / Committed without each view
        // having to wire its own sheet.
        VStack(spacing: tokens.spacing.xxl) {
            Spacer()

            VStack(spacing: tokens.spacing.md) {
                Text("ਸ੍ਰਵਣ ਕਰੋ")
                    .font(tokens.type.gurmukhiHero)
                    .foregroundStyle(tokens.colors.ink)
                    .accessibilityAddTraits(.isHeader)

                Text("Listen along with the live kirtan.")
                    .font(tokens.type.serif)
                    .foregroundStyle(tokens.colors.ink2)
                    .multilineTextAlignment(.center)
            }

            Spacer()

            ListenButton {
                startListening()
            }

            VStack(spacing: tokens.spacing.xs) {
                Text("Identifies the shabad in ~20 seconds.")
                    .font(tokens.type.sansSmall)
                    .foregroundStyle(tokens.colors.ink3)
                Text("Runs entirely on this device.")
                    .font(tokens.type.sansSmall)
                    .foregroundStyle(tokens.colors.ink3)
            }

            if shouldShowMicPrompt {
                micPermissionPrompt
                    .padding(.top, tokens.spacing.sm)
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
            }

            Spacer()
        }
        .padding(.horizontal, tokens.spacing.edge)
        .accessibilityElement(children: .contain)
        .task {
            micStatus = AudioPermissions.status
        }
        .onChange(of: scenePhase) { _, newPhase in
            // iOS revoke flow: user toggles mic off in Settings →
            // Sangat returns to foreground → status flips to .denied
            // → pill must re-appear. Without this observer the
            // `.task` modifier only fires on initial mount and never
            // notices changes made outside the app.
            if newPhase == .active {
                micStatus = AudioPermissions.status
            }
        }
        .animation(.easeInOut(duration: 0.25), value: shouldShowMicPrompt)
    }

    // MARK: - Mic permission affordance (M5.4.1)

    /// Decision tree for what tapping the Listen disc means in each
    /// permission state. Extracted as a pure mapping so unit tests
    /// cover the contract directly without needing to render or
    /// drive the view. Mirrored by `shouldShowMicPrompt` for the
    /// pill-visibility side of the same gate.
    enum ListenAction: Equatable {
        /// Mic granted — start the caption source.
        case startCapture
        /// Mic not yet asked — fire the system prompt, auto-start on grant.
        case requestThenStart
        /// Mic denied or restricted — bounce to Settings.app.
        case openSettings
        /// Platform without a microphone (macOS test host). Log and stay put.
        case noop
    }

    /// Pure mapping for the Listen-disc tap, given a mic permission
    /// state. Kept `internal` (default) so unit tests can call it; not
    /// exposed beyond the module.
    static func listenAction(for status: MicPermissionStatus) -> ListenAction {
        switch status {
        case .granted:                  return .startCapture
        case .notDetermined:            return .requestThenStart
        case .denied, .restricted:      return .openSettings
        case .unavailable:              return .noop
        }
    }

    /// Pure mapping for the mic-permission pill's visibility. The
    /// pill exists to recover from `.notDetermined` (user tapped
    /// "Not now" in onboarding) or `.denied` (user granted then
    /// revoked via Settings). All other states hide it.
    static func shouldShowMicPrompt(for status: MicPermissionStatus) -> Bool {
        switch status {
        case .notDetermined, .denied:               return true
        case .granted, .restricted, .unavailable:   return false
        }
    }

    /// Title rendered on the pill. `.denied` routes to Settings; the
    /// other surfaced state still prompts in-app.
    static func micPromptTitle(for status: MicPermissionStatus) -> String {
        switch status {
        case .denied:   return "Enable microphone in Settings"
        default:        return "Enable microphone"
        }
    }

    /// Secondary copy on the pill.
    static func micPromptSubtitle(for status: MicPermissionStatus) -> String {
        switch status {
        case .denied:   return "Open iOS Settings → Sangat → Microphone"
        default:        return "Required to caption live kirtan"
        }
    }

    /// SF Symbol name for the pill's leading icon. `.denied` shows
    /// the gear (Settings affordance); everything else shows the mic.
    static func micPromptIconName(for status: MicPermissionStatus) -> String {
        switch status {
        case .denied:   return "gearshape.fill"
        default:        return "mic.fill"
        }
    }

    private var shouldShowMicPrompt: Bool {
        Self.shouldShowMicPrompt(for: micStatus)
    }

    @ViewBuilder
    private var micPermissionPrompt: some View {
        Button {
            handleMicPromptTap()
        } label: {
            HStack(spacing: tokens.spacing.sm) {
                Image(systemName: micPromptIconName)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(tokens.colors.accent)

                VStack(alignment: .leading, spacing: 1) {
                    Text(micPromptTitle)
                        .font(tokens.type.sans.weight(.semibold))
                        .foregroundStyle(tokens.colors.ink)
                    Text(micPromptSubtitle)
                        .font(tokens.type.sansSmall)
                        .foregroundStyle(tokens.colors.ink3)
                }

                Spacer(minLength: tokens.spacing.xs)

                Image(systemName: "chevron.right")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(tokens.colors.ink3)
            }
            .padding(.horizontal, tokens.spacing.md)
            .padding(.vertical, tokens.spacing.sm + 2)
            .background(tokens.colors.bgSoft, in: Capsule())
            .overlay(
                Capsule().stroke(tokens.colors.rule, lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
        .disabled(requestInFlight)
        .opacity(requestInFlight ? 0.55 : 1)
        .accessibilityIdentifier("idle.micPrompt")
    }

    private var micPromptIconName: String  { Self.micPromptIconName(for: micStatus) }
    private var micPromptTitle:    String  { Self.micPromptTitle(for: micStatus) }
    private var micPromptSubtitle: String  { Self.micPromptSubtitle(for: micStatus) }

    private func handleMicPromptTap() {
        env.haptics.play(.selection)
        // The pill's tap is a subset of the Listen disc's tap: it
        // never auto-starts capture even on a grant — the user
        // explicitly asked for the permission step, not the capture
        // step. So we filter the shared decision table down to the
        // permission-recovery actions.
        switch Self.listenAction(for: micStatus) {
        case .openSettings:
            // iOS won't re-prompt once denied; only Settings.app can
            // re-grant. Hand the user there and re-read status when
            // they return via the scenePhase observer.
            _ = AudioPermissions.openAppSettings()
        case .requestThenStart:
            // Fire the system dialog but don't start capture on grant
            // from the pill — see the comment above.
            requestPermissionFromPill()
        case .startCapture, .noop:
            break  // pill is hidden in these states (`shouldShowMicPrompt == false`)
        }
    }

    private func startListening() {
        env.haptics.play(.impactMedium)
        // Listen must gate on mic permission. Without this guard the
        // caption source would either fail silently at AVAudioSession
        // setup (denied) or fire the system prompt mid-start in an
        // un-reentrant code path (notDetermined). One decision table
        // (`Self.listenAction`) keeps the disc and pill in sync.
        switch Self.listenAction(for: micStatus) {
        case .startCapture:
            beginCaptionSource()

        case .requestThenStart:
            requestInFlight = true
            Task {
                let result = await AudioPermissions.request()
                requestInFlight = false
                micStatus = result
                env.preferences.micPermissionAcknowledged = (result != .notDetermined)
                AppLogger.ui.info("IdleView listen-tap mic request resolved → \(String(describing: result), privacy: .public)")
                if result == .granted {
                    beginCaptionSource()
                }
            }

        case .openSettings:
            // No re-prompt is possible once the user has chosen Don't
            // Allow (or for restricted profiles); only Settings can
            // re-grant. Mirror the pill's denied-path: bounce there.
            AppLogger.ui.info("IdleView listen-tap blocked (status=\(String(describing: micStatus), privacy: .public)) — routing to Settings")
            _ = AudioPermissions.openAppSettings()

        case .noop:
            // Platforms without a microphone (macOS test host). No
            // useful action — log and stay put.
            AppLogger.ui.info("IdleView listen-tap ignored — mic unavailable on this platform")
        }
    }

    private func requestPermissionFromPill() {
        requestInFlight = true
        Task {
            let result = await AudioPermissions.request()
            requestInFlight = false
            micStatus = result
            env.preferences.micPermissionAcknowledged = (result != .notDetermined)
            AppLogger.ui.info("IdleView mic re-request resolved → \(String(describing: result), privacy: .public)")
        }
    }

    private func beginCaptionSource() {
        Task {
            do {
                try await env.captionModel.start()
            } catch {
                AppLogger.ui.error(
                    "IdleView: caption source start failed — \(error.localizedDescription, privacy: .public)"
                )
            }
        }
    }
}

#Preview("IdleView · paper") {
    IdleView()
        .environment(AppEnvironment.preview())
        .previewTheme(.paper)
}

#Preview("IdleView · darbar") {
    IdleView()
        .environment(AppEnvironment.preview(theme: .darbar))
        .previewTheme(.darbar)
}

#Preview("IdleView · mool") {
    IdleView()
        .environment(AppEnvironment.preview(theme: .mool))
        .previewTheme(.mool)
}
