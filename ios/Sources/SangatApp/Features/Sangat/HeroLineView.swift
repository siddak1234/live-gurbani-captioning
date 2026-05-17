//
//  HeroLineView.swift
//  GurbaniCaptioningApp · SangatApp · Features/Sangat
//
//  Created for the Sangat iOS app, M5.2 (Sangat reading).
//
//  Committed-state reading layout #1 — single big line. Default reading
//  view per the design system; ideal when the device is held in hand.
//
//  Gurmukhi at hero size; translit and english are gated by user prefs
//  (translit on by default, meaning on by default — see Preferences).
//  Progress strip at the bottom shows line N of M + verseId.

import SwiftUI
import GurbaniCaptioning

public struct HeroLineView: View {

    @Environment(AppEnvironment.self) private var env
    @Environment(\.themeTokens) private var tokens

    public let guess: LineGuess

    public init(guess: LineGuess) {
        self.guess = guess
    }

    public var body: some View {
        VStack(spacing: tokens.spacing.lg) {
            metaHeader
                .padding(.top, tokens.spacing.lg)

            StatePill(state: .committed(shabadId: guess.shabadId))

            Spacer()

            lineDisplay
                .padding(.horizontal, tokens.spacing.edge)
                .id(guess.lineIdx)
                .transition(.opacity)

            Spacer()

            progressStrip
                .padding(.horizontal, tokens.spacing.edge)
                .padding(.bottom, tokens.spacing.lg)
        }
        .animation(.easeInOut(duration: 0.3), value: guess.lineIdx)
    }

    @ViewBuilder
    private var metaHeader: some View {
        let meta = env.shabadMeta(forShabadId: guess.shabadId)
        ShabadMetaHeader(
            raag: meta.raag,
            ang: meta.ang,
            author: meta.author,
            authorGurmukhi: meta.authorGurmukhi
        )
    }

    @ViewBuilder
    private var lineDisplay: some View {
        if let resolved = env.resolveLine(shabadId: guess.shabadId, lineIdx: guess.lineIdx) {
            GurbaniText(
                gurmukhi: resolved.gurmukhi,
                translit: env.translitEnabled ? resolved.transliteration : nil,
                english: env.meaningEnabled ? resolved.english : nil,
                size: .hero
            )
        } else {
            Text("—")
                .font(tokens.type.gurmukhiHero)
                .foregroundStyle(tokens.colors.ink3)
        }
    }

    private var progressStrip: some View {
        let total = env.totalLines(forShabadId: guess.shabadId)
        let verseId = env.resolveLine(shabadId: guess.shabadId, lineIdx: guess.lineIdx)?.verseId

        return VStack(spacing: tokens.spacing.sm) {
            HStack(alignment: .firstTextBaseline) {
                Text("Line \(guess.lineIdx + 1) of \(total)")
                    .font(tokens.type.mono)
                    .foregroundStyle(tokens.colors.ink3)
                Spacer()
                if let verseId {
                    Text(verseId)
                        .font(tokens.type.mono)
                        .foregroundStyle(tokens.colors.ink3)
                }
                ConfidenceText(value: guess.confidence)
            }
            ProgressBar(value: Double(guess.lineIdx + 1) / Double(max(total, 1)))
        }
    }
}

private struct ProgressBar: View {

    @Environment(\.themeTokens) private var tokens
    let value: Double

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(tokens.colors.rule)
                Capsule()
                    .fill(tokens.colors.accent)
                    .frame(width: max(0, geo.size.width * value))
            }
        }
        .frame(height: 3)
        .accessibilityHidden(true)
    }
}

#Preview("HeroLineView · paper") {
    HeroLineView(guess: PreviewSample.guess(lineIdx: 1))
        .environment(AppEnvironment.preview())
        .previewTheme(.paper)
}

#Preview("HeroLineView · darbar") {
    HeroLineView(guess: PreviewSample.guess(lineIdx: 2))
        .environment(AppEnvironment.preview(theme: .darbar))
        .previewTheme(.darbar)
}

#Preview("HeroLineView · mool") {
    HeroLineView(guess: PreviewSample.guess(lineIdx: 3))
        .environment(AppEnvironment.preview(theme: .mool))
        .previewTheme(.mool)
}

// MARK: - Preview helpers

private enum PreviewSample {
    static func guess(lineIdx: Int) -> LineGuess {
        LineGuess(
            chunk: AsrChunk(start: 0, end: 5, text: ""),
            shabadId: 1789,
            lineIdx: lineIdx,
            confidence: 88.4,
            isCommitted: true
        )
    }
}
