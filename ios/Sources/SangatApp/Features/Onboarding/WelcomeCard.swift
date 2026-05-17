//
//  WelcomeCard.swift
//  GurbaniCaptioningApp · SangatApp · Features/Onboarding
//
//  Card 1 of 4 — design canvas section 03, `V1Onb01_Welcome`
//  (`assets/v1-paper.jsx` lines 150-184).
//
//  Visual: meta header strip; centered vertically, "ਸ੍ਰਵਣ\nਕਰੋ" in display
//  Gurmukhi, two serif body lines on left-aligned 280pt max width, and a
//  full-width "Begin" pill button at the bottom (ink bg, bg text).

import SwiftUI

public struct WelcomeCard: View {

    @Environment(\.themeTokens) private var tokens

    public let onBegin: () -> Void

    public init(onBegin: @escaping () -> Void) {
        self.onBegin = onBegin
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            OnboardingMetaHeader(mode: nil)

            Spacer()

            VStack(alignment: .leading, spacing: tokens.spacing.lg) {
                Text("ਸ੍ਰਵਣ\nਕਰੋ")
                    .font(tokens.type.gurmukhiDisplay)
                    .foregroundStyle(tokens.colors.ink)
                    .lineSpacing(-4)
                    .accessibilityAddTraits(.isHeader)
                    .accessibilityLabel("Sravan karo. Listen.")

                Text("Live captions for kirtan, snapped to the canonical text of Sri Guru Granth Sahib.")
                    .font(tokens.type.serifTitle.weight(.regular))
                    .foregroundStyle(tokens.colors.ink2)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: 280, alignment: .leading)

                Text("Runs entirely on this device. Your microphone audio never leaves the phone.")
                    .font(tokens.type.sans)
                    .foregroundStyle(tokens.colors.ink3)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: 280, alignment: .leading)
            }
            .padding(.horizontal, tokens.spacing.edge + tokens.spacing.md)

            Spacer()

            beginButton
                .padding(.horizontal, tokens.spacing.edge)
                .padding(.bottom, tokens.spacing.md)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var beginButton: some View {
        Button(action: onBegin) {
            Text("Begin")
                .font(tokens.type.sans.weight(.semibold))
                .tracking(0.4)
                .frame(maxWidth: .infinity)
                .padding(.vertical, tokens.spacing.md + 2)
                .background(tokens.colors.ink, in: Capsule())
                .foregroundStyle(tokens.colors.bg)
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("onboarding.welcome.begin")
    }
}

#Preview("WelcomeCard · paper") {
    WelcomeCard(onBegin: {})
        .previewTheme(.paper)
}

#Preview("WelcomeCard · darbar") {
    WelcomeCard(onBegin: {})
        .previewTheme(.darbar)
}

#Preview("WelcomeCard · mool") {
    WelcomeCard(onBegin: {})
        .previewTheme(.mool)
}
