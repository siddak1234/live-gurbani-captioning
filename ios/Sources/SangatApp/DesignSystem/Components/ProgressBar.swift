//
//  ProgressBar.swift
//  GurbaniCaptioningApp · SangatApp · DesignSystem/Components
//
//  Thin horizontal fill bar — `tokens.colors.rule` background +
//  `tokens.colors.accent` foreground. Used by:
//    - `HeroLineView` progress strip (Line N of M filled bar)
//    - `CastReadingView` projector progress strip
//
//  Promoted from a private struct in `HeroLineView` to a shared atom
//  during M5.5 — the cast view needed the same primitive, so the
//  "no duplicates" invariant required extraction. Both call sites
//  read identically: the surrounding text + spacing is per-view, the
//  bar primitive is shared.

import SwiftUI

public struct ProgressBar: View {

    @Environment(\.themeTokens) private var tokens

    /// Fill fraction in `[0, 1]`. Values outside that range are clamped.
    public let value: Double

    /// Bar thickness. Default matches the in-app reading strip; cast
    /// view passes a slightly thicker value for projector legibility.
    public let height: CGFloat

    public init(value: Double, height: CGFloat = 3) {
        self.value = value
        self.height = height
    }

    public var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(tokens.colors.rule)
                Capsule()
                    .fill(tokens.colors.accent)
                    .frame(width: max(0, geo.size.width * clampedValue))
            }
        }
        .frame(height: height)
        .accessibilityHidden(true)
    }

    private var clampedValue: Double {
        min(max(value, 0), 1)
    }
}

#Preview("ProgressBar · all themes") {
    VStack(spacing: 24) {
        ForEach(Theme.allCases) { theme in
            VStack(alignment: .leading, spacing: 8) {
                Text(theme.displayName)
                    .font(theme.tokens.type.sansCaps)
                ProgressBar(value: 0.5)
                    .frame(width: 240)
            }
            .padding()
            .background(theme.tokens.colors.bg)
            .environment(\.themeTokens, theme.tokens)
        }
    }
}
