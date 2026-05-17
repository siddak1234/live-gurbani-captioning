//
//  StatePill.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Visual indicator for the engine state. Cases mirror `ShabadState` from
//  the library. Every reading view headers this so the user sees what
//  state the engine is in at a glance.

import SwiftUI
import GurbaniCaptioning

public struct StatePill: View {

    @Environment(\.themeTokens) private var tokens

    public let state: ShabadState

    public init(state: ShabadState) {
        self.state = state
    }

    public var body: some View {
        HStack(spacing: tokens.spacing.sm) {
            Circle()
                .fill(dotColor)
                .frame(width: 7, height: 7)
                .modifier(PulseIfListening(isListening: isListening, color: dotColor))
            Text(label)
                .font(tokens.type.sansCaps)
                .tracking(0.4)
                .foregroundStyle(tokens.colors.ink2)
        }
        .padding(.horizontal, tokens.spacing.md)
        .padding(.vertical, 6)
        .background(background, in: Capsule())
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(label)
    }

    // MARK: - Resolution

    private var label: String {
        switch state {
        case .listening:        return "Listening"
        case .tentative:        return "Tentative match"
        case .committed:        return "Committed"
        }
    }

    private var dotColor: Color {
        switch state {
        case .listening, .tentative: return tokens.colors.amber
        case .committed:             return tokens.colors.accent
        }
    }

    private var background: Color {
        switch state {
        case .listening, .tentative: return tokens.colors.bgSoft
        case .committed:             return tokens.colors.accentSoft
        }
    }

    private var isListening: Bool {
        if case .listening = state { return true }
        return false
    }
}

// MARK: - Pulse animation

private struct PulseIfListening: ViewModifier {
    let isListening: Bool
    let color: Color
    @State private var pulsing = false

    func body(content: Content) -> some View {
        content
            .overlay(
                Circle()
                    .stroke(color, lineWidth: 1)
                    .opacity(pulsing ? 0 : 0.6)
                    .scaleEffect(pulsing ? 2.4 : 1.0)
                    .animation(
                        isListening
                            ? .easeOut(duration: 1.6).repeatForever(autoreverses: false)
                            : .default,
                        value: pulsing
                    )
            )
            .onAppear { if isListening { pulsing = true } }
            .onChange(of: isListening) { _, newValue in pulsing = newValue }
    }
}

#Preview("StatePill · all themes") {
    VStack(spacing: 24) {
        ForEach(Theme.allCases) { theme in
            VStack(alignment: .leading, spacing: 12) {
                Text(theme.displayName.uppercased())
                    .font(.system(size: 10, weight: .semibold))
                    .tracking(0.6)
                    .foregroundStyle(.secondary)
                HStack(spacing: 12) {
                    StatePill(state: .listening)
                    StatePill(state: .tentative(shabadId: 1789))
                    StatePill(state: .committed(shabadId: 1789))
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(theme.tokens.colors.bg)
            .environment(\.themeTokens, theme.tokens)
            .preferredColorScheme(theme.isDark ? .dark : .light)
        }
    }
    .padding(20)
    .background(Color.gray.opacity(0.1))
}
