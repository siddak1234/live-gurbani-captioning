//
//  WaveformView.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Stylized live waveform. M5.1 ships an animated decorative waveform —
//  not driven by real audio levels. M5.6 can wire it to the audio session's
//  meter readings; the view's `levels` API is already in place for that.

import SwiftUI

public struct WaveformView: View {

    @Environment(\.themeTokens) private var tokens
    @State private var phase: Double = 0

    public let levels: [CGFloat]?
    public let barCount: Int
    public let maxHeight: CGFloat
    public let isAnimating: Bool

    public init(
        levels: [CGFloat]? = nil,
        barCount: Int = 32,
        maxHeight: CGFloat = 40,
        isAnimating: Bool = true
    ) {
        self.levels = levels
        self.barCount = barCount
        self.maxHeight = maxHeight
        self.isAnimating = isAnimating
    }

    public var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0)) { context in
            HStack(alignment: .center, spacing: 3) {
                ForEach(0..<barCount, id: \.self) { idx in
                    Capsule()
                        .fill(tokens.colors.ink2)
                        .frame(width: 2, height: barHeight(at: idx, t: context.date.timeIntervalSinceReferenceDate))
                }
            }
            .frame(height: maxHeight)
        }
        .accessibilityHidden(true)
    }

    private func barHeight(at idx: Int, t: TimeInterval) -> CGFloat {
        if let levels, idx < levels.count {
            return max(4, levels[idx] * maxHeight)
        }
        // Decorative: layered sines + sample-jitter.
        let i = Double(idx)
        let v = abs(sin(i * 0.7 + t * 3.0) + 0.6 * sin(i * 1.3 + t * 5.0))
        let normalized = min(1.0, v / 1.6)
        return max(4, CGFloat(normalized) * maxHeight)
    }
}

#Preview("WaveformView · all themes") {
    VStack(spacing: 32) {
        ForEach(Theme.allCases) { theme in
            WaveformView()
                .padding(.vertical, 20)
                .frame(maxWidth: .infinity)
                .background(theme.tokens.colors.bg)
                .environment(\.themeTokens, theme.tokens)
        }
    }
    .padding(20)
}
