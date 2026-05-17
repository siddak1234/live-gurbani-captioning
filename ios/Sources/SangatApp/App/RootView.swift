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
