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
    @State private var showSettings: Bool = false

    public init() {}

    public var body: some View {
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

            Spacer()
        }
        .padding(.horizontal, tokens.spacing.edge)
        .accessibilityElement(children: .contain)
        .overlay(alignment: .topTrailing) {
            settingsButton
                .padding(.top, tokens.spacing.sm)
                .padding(.trailing, tokens.spacing.edge)
        }
        .sheet(isPresented: $showSettings) {
            SettingsView()
                .environment(env)
                .environment(\.theme, env.theme)
                .environment(\.themeTokens, env.theme.tokens)
                .preferredColorScheme(env.theme.isDark ? .dark : .light)
        }
    }

    private var settingsButton: some View {
        Button {
            env.haptics.play(.selection)
            showSettings = true
        } label: {
            Image(systemName: "gearshape")
                .font(.system(size: 22, weight: .regular))
                .foregroundStyle(tokens.colors.ink3)
                .frame(width: 44, height: 44)
                .contentShape(Rectangle())
        }
        .accessibilityLabel("Settings")
        .accessibilityIdentifier("idle.settings")
    }

    private func startListening() {
        env.haptics.play(.impactMedium)
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
