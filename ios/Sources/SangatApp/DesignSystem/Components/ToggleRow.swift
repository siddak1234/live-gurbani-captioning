//
//  ToggleRow.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Settings row with a title, optional subtitle, and a `Toggle`. The
//  reusable form-row primitive — used by every Settings screen.

import SwiftUI

public struct ToggleRow: View {

    @Environment(\.themeTokens) private var tokens

    public let title: String
    public let subtitle: String?
    @Binding public var isOn: Bool

    public init(title: String, subtitle: String? = nil, isOn: Binding<Bool>) {
        self.title = title
        self.subtitle = subtitle
        self._isOn = isOn
    }

    public var body: some View {
        HStack(alignment: .center, spacing: tokens.spacing.md) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(tokens.type.serif)
                    .foregroundStyle(tokens.colors.ink)
                if let subtitle {
                    Text(subtitle)
                        .font(tokens.type.sansSmall)
                        .foregroundStyle(tokens.colors.ink3)
                }
            }
            Spacer()
            Toggle("", isOn: $isOn)
                .labelsHidden()
                .tint(tokens.colors.accent)
        }
        .padding(.horizontal, tokens.spacing.md)
        .padding(.vertical, tokens.spacing.md)
        .background(
            Rectangle()
                .fill(tokens.colors.ruleSoft)
                .frame(height: 0.5),
            alignment: .bottom
        )
        .accessibilityElement(children: .combine)
        .accessibilityValue(Text(isOn ? "On" : "Off"))
    }
}

#Preview("ToggleRow · all themes") {
    @Previewable @State var on1 = true
    @Previewable @State var on2 = false
    VStack(spacing: 32) {
        ForEach(Theme.allCases) { theme in
            VStack(spacing: 0) {
                ToggleRow(title: "Transliteration", subtitle: "Latin script", isOn: $on1)
                ToggleRow(title: "English meaning", subtitle: "Sentence translation", isOn: $on2)
            }
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity)
            .background(theme.tokens.colors.bg)
            .environment(\.themeTokens, theme.tokens)
        }
    }
    .padding(20)
}
