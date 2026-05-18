//
//  StatCard.swift
//  GurbaniCaptioningApp · SangatApp · DesignSystem/Components
//
//  Small bgSoft tile with an uppercase label and a large value. Mirrors
//  the `Stat` helper in `assets/v1-paper.jsx` (lines 1026-1040) — the
//  4-card grid in `ConfidenceView` (State / Confidence / Margin / RTF)
//  uses this atom four times. Value font is monospaced + tabular so
//  digits don't jiggle as the live engine updates.

import SwiftUI

public struct StatCard: View {

    @Environment(\.themeTokens) private var tokens

    public let label: String
    public let value: String
    public let valueColor: Color?
    public let useMono: Bool

    public init(
        label: String,
        value: String,
        valueColor: Color? = nil,
        useMono: Bool = true
    ) {
        self.label = label
        self.value = value
        self.valueColor = valueColor
        self.useMono = useMono
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label.uppercased())
                .font(tokens.type.sansCaps)
                .tracking(0.6)
                .foregroundStyle(tokens.colors.ink3)

            Text(value)
                .font(valueFont)
                .monospacedDigit()
                .foregroundStyle(valueColor ?? tokens.colors.ink)
        }
        .padding(.horizontal, tokens.spacing.md - 2)
        .padding(.vertical, tokens.spacing.sm + 2)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(tokens.colors.bgSoft, in: RoundedRectangle(cornerRadius: tokens.radii.sm, style: .continuous))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(label) \(value)")
    }

    private var valueFont: Font {
        useMono
            ? tokens.type.mono.weight(.semibold)
            : tokens.type.serif.weight(.semibold)
    }
}

#Preview("StatCard grid · paper") {
    LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 2), spacing: 8) {
        StatCard(label: "State", value: "Committed", valueColor: ThemeTokens.paper.colors.accent, useMono: false)
        StatCard(label: "Confidence", value: "87.2")
        StatCard(label: "Margin", value: "+12.4")
        StatCard(label: "RTF", value: "0.07")
    }
    .padding()
    .previewTheme(.paper)
}

#Preview("StatCard grid · darbar") {
    LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 2), spacing: 8) {
        StatCard(label: "State", value: "Listening", valueColor: ThemeTokens.darbar.colors.amber, useMono: false)
        StatCard(label: "Confidence", value: "—")
        StatCard(label: "Margin", value: "—")
        StatCard(label: "RTF", value: "0.08")
    }
    .padding()
    .previewTheme(.darbar)
}
