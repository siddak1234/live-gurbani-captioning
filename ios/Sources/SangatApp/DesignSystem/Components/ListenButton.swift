//
//  ListenButton.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Primary call-to-action on the idle screen. Big saffron disc; haptic on
//  tap. Action is passed in — view is presentation-only.

import SwiftUI

public struct ListenButton: View {

    @Environment(\.themeTokens) private var tokens

    public let label: String
    public let size: CGFloat
    public let action: () -> Void

    public init(label: String = "Listen", size: CGFloat = 132, action: @escaping () -> Void) {
        self.label = label
        self.size = size
        self.action = action
    }

    public var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(tokens.colors.accent)
                    .frame(width: size, height: size)
                    .shadow(
                        color: tokens.colors.accent.opacity(0.35),
                        radius: 24, x: 0, y: 8
                    )
                Text(label.uppercased())
                    .font(tokens.type.sans.weight(.semibold))
                    .tracking(0.6)
                    .foregroundStyle(tokens.colors.bg)
            }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(Text(label))
        .accessibilityAddTraits(.isButton)
        .accessibilityIdentifier("listenButton")
    }
}

#Preview("ListenButton · all themes") {
    VStack(spacing: 32) {
        ForEach(Theme.allCases) { theme in
            ListenButton(action: {})
                .padding(40)
                .frame(maxWidth: .infinity)
                .background(theme.tokens.colors.bg)
                .environment(\.themeTokens, theme.tokens)
        }
    }
}
