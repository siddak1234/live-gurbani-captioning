//
//  SessionStartView.swift
//  GurbaniCaptioningApp · SangatApp · Features/Onboarding
//
//  The "Let's Begin" per-session entry screen. Sits between onboarding
//  completion / cold start and the IdleView (Listen page). On every
//  cold start (after the first-time 4-card onboarding has been
//  completed) the app lands here first.
//
//  Layout (matches the minimal Welcome-style placeholder we briefly
//  had after M5.1 + adds a tappable role pill):
//
//    [meta header — Live Gurbani]
//
//    ਸ੍ਰਵਣ ਕਰੋ          (display Gurmukhi, centered)
//    Live Gurbani       (caps tag, centered)
//
//    [ Continuing as Sangat ▾ ]   ← tap → role picker sheet
//
//    [        Begin        ]      ← full-width ink pill
//
//  Role-pill behavior: shows the env's current mode + chevron. Tapping
//  opens `RolePickerSheet` as a bottom sheet. Picking a role updates
//  env.mode immediately (the env's didSet writes to Preferences) and
//  dismisses the sheet. No "Continue" step inside the sheet; selection
//  is the commit.

import SwiftUI

public struct SessionStartView: View {

    @Environment(AppEnvironment.self) private var env
    @Environment(\.themeTokens) private var tokens

    @State private var showRolePicker: Bool = false

    public let onBegin: () -> Void

    public init(onBegin: @escaping () -> Void) {
        self.onBegin = onBegin
    }

    public var body: some View {
        VStack(spacing: 0) {
            OnboardingMetaHeader(mode: nil)

            Spacer()

            VStack(spacing: tokens.spacing.lg) {
                Text("ਸ੍ਰਵਣ ਕਰੋ")
                    .font(tokens.type.gurmukhiDisplay)
                    .foregroundStyle(tokens.colors.ink)
                    .accessibilityAddTraits(.isHeader)
                    .accessibilityLabel("Sravan karo. Listen.")

                Text("Live Gurbani")
                    .font(tokens.type.sansCaps)
                    .tracking(0.8)
                    .foregroundStyle(tokens.colors.ink3)
            }

            Spacer().frame(height: tokens.spacing.xl)

            rolePill

            Spacer()

            beginButton
                .padding(.horizontal, tokens.spacing.edge)
                .padding(.bottom, tokens.spacing.md)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .sheet(isPresented: $showRolePicker) {
            RolePickerSheet(
                selection: Binding(
                    get: { env.mode },
                    set: { newMode in
                        env.haptics.play(.selection)
                        env.mode = newMode
                        showRolePicker = false
                    }
                )
            )
            .environment(env)
            .environment(\.theme, env.theme)
            .environment(\.themeTokens, env.theme.tokens)
            .preferredColorScheme(env.theme.isDark ? .dark : .light)
            .presentationDetents([.height(320), .medium])
            .presentationDragIndicator(.visible)
        }
    }

    private var rolePill: some View {
        Button {
            env.haptics.play(.selection)
            showRolePicker = true
        } label: {
            HStack(spacing: tokens.spacing.xs) {
                Circle()
                    .fill(env.mode == .sevadar ? tokens.colors.sevadar : tokens.colors.accent)
                    .frame(width: 7, height: 7)

                Text("Continuing as \(env.mode == .sevadar ? "Sevadar" : "Sangat")")
                    .font(tokens.type.sans)
                    .foregroundStyle(tokens.colors.ink2)

                Image(systemName: "chevron.down")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(tokens.colors.ink3)
            }
            .padding(.horizontal, tokens.spacing.md)
            .padding(.vertical, tokens.spacing.sm)
            .background(tokens.colors.bgSoft, in: Capsule())
            .overlay(
                Capsule().stroke(tokens.colors.rule, lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("session.rolePill")
    }

    private var beginButton: some View {
        Button {
            env.haptics.play(.success)
            onBegin()
        } label: {
            Text("Begin")
                .font(tokens.type.sans.weight(.semibold))
                .tracking(0.4)
                .frame(maxWidth: .infinity)
                .padding(.vertical, tokens.spacing.md + 2)
                .background(tokens.colors.ink, in: Capsule())
                .foregroundStyle(tokens.colors.bg)
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("session.begin")
    }
}

#Preview("SessionStartView · paper sangat") {
    SessionStartView(onBegin: {})
        .environment(AppEnvironment.preview())
        .previewTheme(.paper)
}

#Preview("SessionStartView · darbar sevadar") {
    SessionStartView(onBegin: {})
        .environment(AppEnvironment.preview(theme: .darbar, mode: .sevadar))
        .previewTheme(.darbar)
}

#Preview("SessionStartView · mool sangat") {
    SessionStartView(onBegin: {})
        .environment(AppEnvironment.preview(theme: .mool))
        .previewTheme(.mool)
}
