//
//  ConfidenceText.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Renders a numeric confidence value in monospace + tabular figures so the
//  digit width stays stable as the value changes. Used by the Sevadar
//  confidence view, the state pill subtitle, and engine debug overlays.

import SwiftUI

public struct ConfidenceText: View {

    @Environment(\.themeTokens) private var tokens

    public let value: Double
    public let prefix: String?
    public let highlight: Bool

    public init(value: Double, prefix: String? = nil, highlight: Bool = false) {
        self.value = value
        self.prefix = prefix
        self.highlight = highlight
    }

    public var body: some View {
        HStack(spacing: 2) {
            if let prefix {
                Text(prefix)
                    .font(tokens.type.mono)
                    .foregroundStyle(tokens.colors.ink3)
            }
            Text(String(format: "%.1f", value))
                .font(tokens.type.mono)
                .monospacedDigit()
                .foregroundStyle(highlight ? tokens.colors.accent : tokens.colors.ink2)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(Text("Confidence \(String(format: "%.0f", value))"))
    }
}

#Preview("ConfidenceText · all themes") {
    VStack(spacing: 24) {
        ForEach(Theme.allCases) { theme in
            HStack(spacing: 16) {
                ConfidenceText(value: 87.2)
                ConfidenceText(value: 87.2, prefix: "conf")
                ConfidenceText(value: 87.2, highlight: true)
            }
            .padding(16)
            .background(theme.tokens.colors.bg)
            .environment(\.themeTokens, theme.tokens)
        }
    }
    .padding(20)
}
