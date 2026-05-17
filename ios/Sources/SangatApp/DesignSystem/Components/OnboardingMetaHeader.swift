//
//  OnboardingMetaHeader.swift
//  GurbaniCaptioningApp · SangatApp · DesignSystem/Components
//
//  Top strip for every onboarding card. Mirrors `V1MetaHeader` from
//  `assets/v1-paper.jsx` (lines 46-67): a small left-aligned "Live
//  Gurbani" wordmark plus a right-aligned mode dot + label. Used only
//  by the onboarding flow today, but kept as a shared atom because the
//  same strip appears on Sevadar surfaces in the design canvas.

import SwiftUI

public struct OnboardingMetaHeader: View {

    @Environment(\.themeTokens) private var tokens

    /// Mode dot color follows mode — saffron accent for Sangat (default),
    /// indigo for Sevadar. `nil` hides the right-side mode label entirely
    /// (e.g. for the Welcome card before the user has picked a role).
    public let mode: AppMode?

    public init(mode: AppMode? = .sangat) {
        self.mode = mode
    }

    public var body: some View {
        HStack {
            Text("Live Gurbani")
                .font(tokens.type.sansCaps)
                .tracking(0.6)
                .foregroundStyle(tokens.colors.ink3)

            Spacer()

            if let mode {
                HStack(spacing: tokens.spacing.xs) {
                    Circle()
                        .fill(dotColor(for: mode))
                        .frame(width: 6, height: 6)
                    Text(mode == .sevadar ? "Sevadar" : "Sangat")
                        .font(tokens.type.sansCaps)
                        .tracking(0.6)
                        .foregroundStyle(tokens.colors.ink3)
                }
            }
        }
        .padding(.horizontal, tokens.spacing.edge)
        .padding(.top, tokens.spacing.md)
    }

    private func dotColor(for mode: AppMode) -> Color {
        switch mode {
        case .sevadar: return tokens.colors.sevadar
        case .sangat:  return tokens.colors.accent
        }
    }
}

#Preview("OnboardingMetaHeader · paper sangat") {
    OnboardingMetaHeader(mode: .sangat)
        .previewTheme(.paper)
}

#Preview("OnboardingMetaHeader · darbar sevadar") {
    OnboardingMetaHeader(mode: .sevadar)
        .previewTheme(.darbar)
}

#Preview("OnboardingMetaHeader · mool, no mode") {
    OnboardingMetaHeader(mode: nil)
        .previewTheme(.mool)
}
