//
//  SectionLabel.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Settings/list section header. Uppercase, tracked, tertiary ink.

import SwiftUI

public struct SectionLabel: View {

    @Environment(\.themeTokens) private var tokens

    public let title: String

    public init(_ title: String) {
        self.title = title
    }

    public var body: some View {
        Text(title.uppercased())
            .font(tokens.type.sansCaps)
            .tracking(0.8)
            .foregroundStyle(tokens.colors.ink3)
            .accessibilityAddTraits(.isHeader)
    }
}

#Preview("SectionLabel · all themes") {
    VStack(alignment: .leading, spacing: 32) {
        ForEach(Theme.allCases) { theme in
            VStack(alignment: .leading, spacing: 8) {
                SectionLabel("Layers")
                Text("Settings rows go here.")
                    .foregroundStyle(theme.tokens.colors.ink2)
            }
            .padding(20)
            .background(theme.tokens.colors.bg)
            .environment(\.themeTokens, theme.tokens)
        }
    }
    .padding(20)
}
