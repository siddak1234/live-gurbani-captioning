//
//  ListeningView.swift
//  GurbaniCaptioningApp · SangatApp · Features/Sangat
//
//  Created for the Sangat iOS app, M5.2 (Sangat reading).
//
//  Shown while the engine is in `.listening` state — audio coming in,
//  no shabad candidate locked yet. Pulse rings + elapsed timer + small
//  waveform + stop button.
//
//  The elapsed timer is driven by `TimelineView(.periodic)` on a local
//  `startTime` so it counts seconds since the view appeared. Real engine
//  uptime tracking lands when LiveCaptionSource is wired (M5.7+).

import SwiftUI

public struct ListeningView: View {

    @Environment(AppEnvironment.self) private var env
    @Environment(\.themeTokens) private var tokens
    @State private var startTime: Date = Date()

    public init() {}

    public var body: some View {
        VStack(spacing: tokens.spacing.xl) {
            StatePill(state: .listening)
                .padding(.top, tokens.spacing.lg)

            Spacer()

            ZStack {
                PulseRings(size: 200)
                ElapsedTimeBadge(startTime: startTime)
            }
            .frame(width: 200, height: 200)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Listening for the shabad")

            VStack(spacing: tokens.spacing.sm) {
                Text("ਸ੍ਰਵਣ ਕਰ ਰਿਹਾ ਹੈ…")
                    .font(tokens.type.gurmukhiLarge)
                    .foregroundStyle(tokens.colors.ink)
                Text("Identifying the shabad in SGGS")
                    .font(tokens.type.sansSmall)
                    .foregroundStyle(tokens.colors.ink3)
            }

            WaveformView(maxHeight: 32)
                .padding(.horizontal, tokens.spacing.xl)

            Spacer()

            stopButton
                .padding(.bottom, tokens.spacing.lg)
        }
        .padding(.horizontal, tokens.spacing.edge)
        .onAppear {
            startTime = Date()
        }
    }

    private var stopButton: some View {
        Button {
            env.haptics.play(.selection)
            env.captionModel.stop()
        } label: {
            Text("Stop")
                .font(tokens.type.sans.weight(.semibold))
                .foregroundStyle(tokens.colors.ink2)
                .padding(.horizontal, tokens.spacing.lg)
                .padding(.vertical, tokens.spacing.sm)
                .background(
                    Capsule().stroke(tokens.colors.rule, lineWidth: 1)
                )
        }
        .accessibilityLabel("Stop listening")
    }
}

private struct ElapsedTimeBadge: View {

    @Environment(\.themeTokens) private var tokens
    let startTime: Date

    var body: some View {
        TimelineView(.periodic(from: startTime, by: 0.5)) { context in
            Text(formatted(elapsed: context.date.timeIntervalSince(startTime)))
                .font(tokens.type.mono)
                .foregroundStyle(tokens.colors.ink2)
                .padding(.horizontal, tokens.spacing.md)
                .padding(.vertical, tokens.spacing.sm)
                .background(
                    Circle()
                        .fill(tokens.colors.bg)
                        .frame(width: 88, height: 88)
                        .overlay(Circle().stroke(tokens.colors.amber, lineWidth: 1.5))
                )
        }
    }

    private func formatted(elapsed: TimeInterval) -> String {
        let total = max(0, Int(elapsed))
        let minutes = total / 60
        let seconds = total % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

#Preview("ListeningView · paper") {
    ListeningView()
        .environment(AppEnvironment.preview())
        .previewTheme(.paper)
}

#Preview("ListeningView · darbar") {
    ListeningView()
        .environment(AppEnvironment.preview(theme: .darbar))
        .previewTheme(.darbar)
}

#Preview("ListeningView · mool") {
    ListeningView()
        .environment(AppEnvironment.preview(theme: .mool))
        .previewTheme(.mool)
}
