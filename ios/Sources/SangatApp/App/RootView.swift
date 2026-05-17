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
        .environment(\.theme, env.theme)
        .environment(\.themeTokens, env.theme.tokens)
        .environment(env)
        .safeAreaInset(edge: .top, spacing: 0) {
            // Top chrome row — Back (when running) + Settings gear. Lives
            // in the safe-area inset so the reading view's content always
            // sits below the chrome and the meta header never collides
            // with the corner icons. The Welcome screen has no chrome.
            if env.hasCompletedOnboarding {
                topChromeRow
            }
        }
        .sheet(isPresented: $showSettings) {
            SettingsView()
                .environment(env)
                .environment(\.theme, env.theme)
                .environment(\.themeTokens, env.theme.tokens)
                .preferredColorScheme(env.theme.isDark ? .dark : .light)
        }
        .task {
            await prepareCaptionSource()
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
            OnboardingPlaceholderView()
        } else if !env.captionModel.isRunning {
            IdleView()
        } else {
            ReadingHost()
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

            settingsGearButton
        }
        .padding(.horizontal, env.theme.tokens.spacing.edge - 10)
        .padding(.top, env.theme.tokens.spacing.xs)
        .padding(.bottom, env.theme.tokens.spacing.xs)
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

// MARK: - Onboarding placeholder (replaced by M5.4)

private struct OnboardingPlaceholderView: View {

    @Environment(AppEnvironment.self) private var env
    @Environment(\.themeTokens) private var tokens

    var body: some View {
        VStack(spacing: tokens.spacing.lg) {
            Spacer()

            Text("ਸ੍ਰਵਣ ਕਰੋ")
                .font(tokens.type.gurmukhiHero)
                .foregroundStyle(tokens.colors.ink)
                .accessibilityAddTraits(.isHeader)

            Text("Live Gurbani")
                .font(tokens.type.sansCaps)
                .tracking(0.8)
                .foregroundStyle(tokens.colors.ink3)

            Spacer()

            Button {
                env.haptics.play(.success)
                env.hasCompletedOnboarding = true
            } label: {
                Text("Begin")
                    .font(tokens.type.sans.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, tokens.spacing.md)
                    .background(tokens.colors.ink, in: Capsule())
                    .foregroundStyle(tokens.colors.bg)
            }
            .padding(.horizontal, tokens.spacing.edge)
            .padding(.bottom, tokens.spacing.lg)
            .accessibilityIdentifier("onboarding.begin")
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
