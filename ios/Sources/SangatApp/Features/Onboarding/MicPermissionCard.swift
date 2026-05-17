//
//  MicPermissionCard.swift
//  GurbaniCaptioningApp · SangatApp · Features/Onboarding
//
//  Card 3 of 4 — design canvas section 03, `V1Onb03_Mic`
//  (`assets/v1-paper.jsx` lines 238-278).
//
//  Visual: meta header; centered 96pt accentSoft disc with mic glyph
//  + concentric ring at 40% opacity; "Hear the kirtan" title; body
//  copy reassuring on-device privacy. Bottom: "Allow microphone"
//  (ink bg) + "Not now" (plain text, ink3).
//
//  Behavior: "Allow microphone" calls `OnboardingViewModel.requestMic
//  Permission()`, which routes through `AudioPermissions.request()` —
//  the system mic dialog appears, the user's answer is recorded, and
//  the flow advances. "Not now" calls `skipMic()` which records
//  unresolved and advances without prompting.

import SwiftUI

public struct MicPermissionCard: View {

    @Environment(\.themeTokens) private var tokens

    public let onAllow: () -> Void
    public let onSkip: () -> Void
    public let isRequestInFlight: Bool

    public init(
        isRequestInFlight: Bool = false,
        onAllow: @escaping () -> Void,
        onSkip: @escaping () -> Void
    ) {
        self.isRequestInFlight = isRequestInFlight
        self.onAllow = onAllow
        self.onSkip = onSkip
    }

    public var body: some View {
        VStack(spacing: 0) {
            OnboardingMetaHeader(mode: nil)

            Spacer()

            VStack(spacing: tokens.spacing.lg) {
                micDisc

                VStack(spacing: tokens.spacing.sm) {
                    Text("Hear the kirtan")
                        .font(tokens.type.serifTitle)
                        .foregroundStyle(tokens.colors.ink)
                        .multilineTextAlignment(.center)
                        .accessibilityAddTraits(.isHeader)

                    Text("We need microphone access to transcribe the kirtan. Audio is processed on-device and never leaves your phone.")
                        .font(tokens.type.sans)
                        .foregroundStyle(tokens.colors.ink2)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: 280)
                }
            }
            .padding(.horizontal, tokens.spacing.edge + tokens.spacing.md)

            Spacer()

            buttonStack
                .padding(.horizontal, tokens.spacing.edge)
                .padding(.bottom, tokens.spacing.md)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var micDisc: some View {
        ZStack {
            Circle()
                .fill(tokens.colors.accentSoft)
                .frame(width: 96, height: 96)

            Circle()
                .stroke(tokens.colors.accent.opacity(0.4), lineWidth: 1.5)
                .frame(width: 56, height: 56)

            // Simple mic glyph: capsule body + small base bar. Matches
            // the "simple shape, not branded" intent of the JSX.
            VStack(spacing: 3) {
                Capsule()
                    .fill(tokens.colors.accent)
                    .frame(width: 16, height: 28)
                Rectangle()
                    .fill(tokens.colors.accent)
                    .frame(width: 22, height: 1.5)
            }
        }
        .accessibilityHidden(true)
    }

    private var buttonStack: some View {
        VStack(spacing: tokens.spacing.sm) {
            Button(action: onAllow) {
                HStack(spacing: tokens.spacing.sm) {
                    if isRequestInFlight {
                        ProgressView()
                            .progressViewStyle(.circular)
                            .tint(tokens.colors.bg)
                    }
                    Text("Allow microphone")
                        .font(tokens.type.sans.weight(.semibold))
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, tokens.spacing.md + 2)
                .background(tokens.colors.ink, in: Capsule())
                .foregroundStyle(tokens.colors.bg)
            }
            .buttonStyle(.plain)
            .disabled(isRequestInFlight)
            .accessibilityIdentifier("onboarding.mic.allow")

            Button(action: onSkip) {
                Text("Not now")
                    .font(tokens.type.sans)
                    .foregroundStyle(tokens.colors.ink3)
                    .padding(.vertical, tokens.spacing.sm)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.plain)
            .disabled(isRequestInFlight)
            .accessibilityIdentifier("onboarding.mic.skip")
        }
    }
}

#Preview("MicPermissionCard · paper") {
    MicPermissionCard(onAllow: {}, onSkip: {})
        .previewTheme(.paper)
}

#Preview("MicPermissionCard · darbar") {
    MicPermissionCard(onAllow: {}, onSkip: {})
        .previewTheme(.darbar)
}

#Preview("MicPermissionCard · mool") {
    MicPermissionCard(onAllow: {}, onSkip: {})
        .previewTheme(.mool)
}

#Preview("MicPermissionCard · in-flight") {
    MicPermissionCard(isRequestInFlight: true, onAllow: {}, onSkip: {})
        .previewTheme(.paper)
}
