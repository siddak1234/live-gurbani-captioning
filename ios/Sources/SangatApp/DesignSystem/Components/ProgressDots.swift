//
//  ProgressDots.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Vote-build indicator: "n of N chunks agree". Used in the Tentative
//  view to show the user how close the engine is to committing.

import SwiftUI

public struct ProgressDots: View {

    @Environment(\.themeTokens) private var tokens

    public let filled: Int
    public let total: Int
    public let label: String?

    public init(filled: Int, total: Int, label: String? = nil) {
        self.filled = filled
        self.total = total
        self.label = label
    }

    public var body: some View {
        VStack(spacing: tokens.spacing.sm) {
            HStack(spacing: 4) {
                ForEach(0..<total, id: \.self) { idx in
                    Capsule()
                        .fill(idx < filled ? tokens.colors.amber : tokens.colors.rule)
                        .frame(width: 22, height: 4)
                }
            }
            if let label {
                Text(label.uppercased())
                    .font(tokens.type.sansCaps)
                    .tracking(0.4)
                    .foregroundStyle(tokens.colors.ink3)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(label ?? "\(filled) of \(total) chunks agree"))
        .accessibilityValue(Text("\(filled) of \(total)"))
    }
}

#Preview("ProgressDots · all themes") {
    VStack(spacing: 32) {
        ForEach(Theme.allCases) { theme in
            ProgressDots(filled: 3, total: 5, label: "3 of 5 chunks agree")
                .padding(20)
                .frame(maxWidth: .infinity)
                .background(theme.tokens.colors.bg)
                .environment(\.themeTokens, theme.tokens)
        }
    }
    .padding(20)
}
