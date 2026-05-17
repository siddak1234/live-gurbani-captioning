//
//  PulseRings.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Concentric pulsing rings — used in the Listening view to indicate the
//  engine is hearing audio, and in the mic-permission onboarding card.
//  Three rings, staggered, looping outward.

import SwiftUI

public struct PulseRings: View {

    @Environment(\.themeTokens) private var tokens
    @State private var phase: Double = 0

    public let size: CGFloat
    public let ringCount: Int
    public let cycleDuration: Double

    public init(size: CGFloat = 200, ringCount: Int = 3, cycleDuration: Double = 2.4) {
        self.size = size
        self.ringCount = ringCount
        self.cycleDuration = cycleDuration
    }

    public var body: some View {
        ZStack {
            ForEach(0..<ringCount, id: \.self) { idx in
                Circle()
                    .stroke(tokens.colors.amber, lineWidth: 1)
                    .frame(width: size, height: size)
                    .scaleEffect(scale(for: idx))
                    .opacity(opacity(for: idx))
            }
        }
        .frame(width: size, height: size)
        .onAppear {
            withAnimation(
                .easeOut(duration: cycleDuration).repeatForever(autoreverses: false)
            ) {
                phase = 1
            }
        }
        .accessibilityHidden(true) // decoration
    }

    private func scale(for idx: Int) -> Double {
        let offset = Double(idx) / Double(ringCount)
        let p = (phase + offset).truncatingRemainder(dividingBy: 1.0)
        return 0.4 + p * 1.0
    }

    private func opacity(for idx: Int) -> Double {
        let offset = Double(idx) / Double(ringCount)
        let p = (phase + offset).truncatingRemainder(dividingBy: 1.0)
        return max(0, 0.5 - p * 0.5)
    }
}

#Preview("PulseRings · all themes") {
    VStack(spacing: 32) {
        ForEach(Theme.allCases) { theme in
            PulseRings()
                .frame(height: 220)
                .frame(maxWidth: .infinity)
                .background(theme.tokens.colors.bg)
                .environment(\.themeTokens, theme.tokens)
        }
    }
}
