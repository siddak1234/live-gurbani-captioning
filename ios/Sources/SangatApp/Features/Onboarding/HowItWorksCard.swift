//
//  HowItWorksCard.swift
//  GurbaniCaptioningApp · SangatApp · Features/Onboarding
//
//  Card 2 of 4 — design canvas section 03, `V1Onb02_HowItWorks`
//  (`assets/v1-paper.jsx` lines 186-236).
//
//  Visual: meta header; "How it follows along" title; vertical timeline
//  of 3 steps with dot+rule connectors: Listening (amber), Tentative
//  (amber), Committed (saffron/accent). Bottom: Back (1× flex,
//  transparent border) + Continue (2× flex, ink bg).

import SwiftUI

public struct HowItWorksCard: View {

    @Environment(\.themeTokens) private var tokens

    public let onBack: () -> Void
    public let onContinue: () -> Void

    public init(onBack: @escaping () -> Void, onContinue: @escaping () -> Void) {
        self.onBack = onBack
        self.onContinue = onContinue
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            OnboardingMetaHeader(mode: nil)

            VStack(alignment: .leading, spacing: tokens.spacing.sm) {
                Text("How it follows along")
                    .font(tokens.type.serifTitle)
                    .foregroundStyle(tokens.colors.ink)
                    .accessibilityAddTraits(.isHeader)
            }
            .padding(.horizontal, tokens.spacing.edge + tokens.spacing.md)
            .padding(.top, tokens.spacing.xxl)

            VStack(alignment: .leading, spacing: tokens.spacing.lg) {
                ForEach(Array(steps.enumerated()), id: \.offset) { idx, step in
                    StepRow(
                        label: step.label,
                        caption: step.caption,
                        dotColor: step.dotColor(tokens: tokens),
                        showConnector: idx < steps.count - 1
                    )
                }
            }
            .padding(.horizontal, tokens.spacing.edge + tokens.spacing.md)
            .padding(.top, tokens.spacing.xl)

            Spacer()

            buttonRow
                .padding(.horizontal, tokens.spacing.edge)
                .padding(.bottom, tokens.spacing.md)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var buttonRow: some View {
        HStack(spacing: tokens.spacing.sm) {
            Button(action: onBack) {
                Text("Back")
                    .font(tokens.type.sans.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, tokens.spacing.md + 2)
                    .foregroundStyle(tokens.colors.ink2)
                    .overlay(
                        Capsule().stroke(tokens.colors.rule, lineWidth: 1)
                    )
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("onboarding.howItWorks.back")

            Button(action: onContinue) {
                Text("Continue")
                    .font(tokens.type.sans.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, tokens.spacing.md + 2)
                    .background(tokens.colors.ink, in: Capsule())
                    .foregroundStyle(tokens.colors.bg)
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("onboarding.howItWorks.continue")
            .layoutPriority(1)
        }
        // 1:2 flex per the design: Back (1) + Continue (2). With matching
        // padding the layout reads correctly because layoutPriority makes
        // Continue absorb extra width.
    }

    // MARK: - Step data (matches V1Onb02_HowItWorks steps array, jsx:187-190)

    private struct StepData {
        let label: String
        let caption: String
        let useAccentDot: Bool

        func dotColor(tokens: ThemeTokens) -> Color {
            useAccentDot ? tokens.colors.accent : tokens.colors.amber
        }
    }

    private var steps: [StepData] {
        [
            StepData(
                label: "Listening",
                caption: "Your phone listens to the kirtan and forms a transcript on-device.",
                useAccentDot: false
            ),
            StepData(
                label: "Tentative",
                caption: "It compares the transcript against every shabad in SGGS to find the best match.",
                useAccentDot: false
            ),
            StepData(
                label: "Committed",
                caption: "Once confident, captions snap to the canonical Gurmukhi line — and follow along.",
                useAccentDot: true
            )
        ]
    }
}

// MARK: - StepRow

private struct StepRow: View {

    @Environment(\.themeTokens) private var tokens

    let label: String
    let caption: String
    let dotColor: Color
    let showConnector: Bool

    var body: some View {
        HStack(alignment: .top, spacing: tokens.spacing.md) {
            VStack(spacing: tokens.spacing.xs) {
                Circle()
                    .fill(dotColor)
                    .frame(width: 10, height: 10)
                    .padding(.top, 6)

                if showConnector {
                    Rectangle()
                        .fill(tokens.colors.rule)
                        .frame(width: 1)
                }
            }
            .frame(width: 10)

            VStack(alignment: .leading, spacing: tokens.spacing.xs) {
                Text(label.uppercased())
                    .font(tokens.type.sansCaps)
                    .tracking(0.8)
                    .foregroundStyle(tokens.colors.ink)

                Text(caption)
                    .font(tokens.type.serif)
                    .foregroundStyle(tokens.colors.ink2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.bottom, tokens.spacing.sm)
        }
    }
}

#Preview("HowItWorksCard · paper") {
    HowItWorksCard(onBack: {}, onContinue: {})
        .previewTheme(.paper)
}

#Preview("HowItWorksCard · darbar") {
    HowItWorksCard(onBack: {}, onContinue: {})
        .previewTheme(.darbar)
}

#Preview("HowItWorksCard · mool") {
    HowItWorksCard(onBack: {}, onContinue: {})
        .previewTheme(.mool)
}
