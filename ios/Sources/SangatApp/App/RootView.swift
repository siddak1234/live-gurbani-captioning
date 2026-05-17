//
//  RootView.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Top-level route. Switches between onboarding (first launch) and the
//  reading host (every launch after). M5.1 ships a *foundation preview*
//  reading host — small functional shell that exercises every seam and
//  proves the demo source drives the UI. M5.2 replaces it with the real
//  Sangat reading views.

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

            if !env.hasCompletedOnboarding {
                OnboardingPlaceholderView()
                    .transition(.opacity)
            } else {
                ReadingHostPlaceholderView()
                    .transition(.opacity)
            }
        }
        .preferredColorScheme(env.theme.isDark ? .dark : .light)
        .environment(\.theme, env.theme)
        .environment(\.themeTokens, env.theme.tokens)
        .environment(env)
        .task {
            await bootCaptionSource()
        }
        .animation(.easeInOut(duration: 0.25), value: env.hasCompletedOnboarding)
    }

    private func bootCaptionSource() async {
        do {
            try await env.captionModel.prepare()
            try await env.captionModel.start()
        } catch {
            AppLogger.app.error("RootView: caption source boot failed — \(error.localizedDescription, privacy: .public)")
        }
    }
}

// MARK: - Foundation preview screens (M5.1 only — replaced by M5.2 / M5.4)

/// Minimal onboarding placeholder. M5.4 will replace with the real flow.
private struct OnboardingPlaceholderView: View {

    @Environment(AppEnvironment.self) private var env
    @Environment(\.themeTokens) private var tokens

    var body: some View {
        VStack(spacing: tokens.spacing.lg) {
            Spacer()
            Text("ਸ੍ਰਵਣ ਕਰੋ")
                .font(tokens.type.gurmukhiHero)
                .foregroundStyle(tokens.colors.ink)
            Text("Live Gurbani")
                .font(tokens.type.sansCaps)
                .tracking(0.8)
                .foregroundStyle(tokens.colors.ink3)
            Spacer()
            Button {
                env.hasCompletedOnboarding = true
                env.haptics.play(.success)
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

/// Minimal reading host placeholder. Exercises every seam so M5.1 is
/// independently demoable. M5.2 will replace with the real reading views.
private struct ReadingHostPlaceholderView: View {

    @Environment(AppEnvironment.self) private var env
    @Environment(\.themeTokens) private var tokens

    var body: some View {
        VStack(spacing: tokens.spacing.lg) {
            header

            Spacer()

            StatePill(state: env.captionModel.state)

            if let guess = env.captionModel.currentGuess {
                Text(lineGurmukhi(for: guess))
                    .font(tokens.type.gurmukhiHero)
                    .foregroundStyle(tokens.colors.ink)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, tokens.spacing.edge)

                ConfidenceText(value: guess.confidence)
            } else {
                Text("ਸ੍ਰਵਣ ਕਰ ਰਿਹਾ ਹੈ…")
                    .font(tokens.type.gurmukhiLarge)
                    .foregroundStyle(tokens.colors.ink2)
            }

            Spacer()

            footer
        }
        .padding(.top, tokens.spacing.xl)
    }

    private var header: some View {
        HStack {
            Text("Live Gurbani")
                .font(tokens.type.sansCaps)
                .tracking(0.6)
                .foregroundStyle(tokens.colors.ink3)
            Spacer()
            Text(env.mode.displayName)
                .font(tokens.type.sansCaps)
                .tracking(0.6)
                .foregroundStyle(tokens.colors.ink3)
        }
        .padding(.horizontal, tokens.spacing.edge)
    }

    private var footer: some View {
        VStack(spacing: tokens.spacing.sm) {
            Text("M5.1 · Foundation preview")
                .font(tokens.type.sansSmall)
                .foregroundStyle(tokens.colors.ink3)
            Text("Real reading views land in M5.2.")
                .font(tokens.type.sansSmall)
                .foregroundStyle(tokens.colors.ink3)
        }
        .padding(.bottom, tokens.spacing.lg)
    }

    /// Look up the Gurmukhi line for a guess. M5.1 uses static preview data
    /// (`DemoCaptionSource` ships fixture line indexes). M5.2 wires this to
    /// the real `ShabadCorpus` lookup through the environment.
    private func lineGurmukhi(for guess: LineGuess) -> String {
        PreviewData.line(forIndex: guess.lineIdx)?.gurmukhi ?? "—"
    }
}

#Preview("RootView · paper · committed") {
    RootView(env: .preview(captionSource: DemoCaptionSource(script: .quickCommit)))
}
