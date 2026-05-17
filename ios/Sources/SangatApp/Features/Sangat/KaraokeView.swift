//
//  KaraokeView.swift
//  GurbaniCaptioningApp · SangatApp · Features/Sangat
//
//  Created for the Sangat iOS app, M5.2 (Sangat reading).
//
//  Committed-state reading layout #2 — three lines stacked: previous
//  (faded), current (accented in a soft-accent card), next (faded).
//  Mimics karaoke-style follow-along reading.

import SwiftUI
import GurbaniCaptioning

public struct KaraokeView: View {

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

            VStack(spacing: tokens.spacing.lg) {
                neighborLine(at: guess.lineIdx - 1)
                currentLine
                neighborLine(at: guess.lineIdx + 1)
            }
            .padding(.horizontal, tokens.spacing.md)

            Spacer()

            lineTicks
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
            authorGurmukhi: meta.authorGurmukhi,
            compact: true
        )
    }

    @ViewBuilder
    private func neighborLine(at idx: Int) -> some View {
        let total = env.totalLines(forShabadId: guess.shabadId)
        if idx >= 0, idx < total,
           let resolved = env.resolveLine(shabadId: guess.shabadId, lineIdx: idx) {
            Text(resolved.gurmukhi)
                .font(tokens.type.gurmukhiLarge)
                .foregroundStyle(tokens.colors.ink3)
                .multilineTextAlignment(.center)
                .opacity(0.4)
                .frame(maxWidth: .infinity)
                .padding(.horizontal, tokens.spacing.edge)
                .accessibilityLabel(idx < guess.lineIdx ? "Previous line" : "Next line")
        } else {
            Color.clear.frame(height: 36)
        }
    }

    @ViewBuilder
    private var currentLine: some View {
        if let resolved = env.resolveLine(shabadId: guess.shabadId, lineIdx: guess.lineIdx) {
            VStack(spacing: tokens.spacing.sm) {
                Text(resolved.gurmukhi)
                    .font(tokens.type.gurmukhiHero)
                    .foregroundStyle(tokens.colors.ink)
                    .multilineTextAlignment(.center)
                if let translit = resolved.transliteration, env.translitEnabled {
                    Text(translit)
                        .font(tokens.type.serifItalic)
                        .foregroundStyle(tokens.colors.ink2)
                        .multilineTextAlignment(.center)
                }
            }
            .padding(.vertical, tokens.spacing.md)
            .padding(.horizontal, tokens.spacing.md)
            .frame(maxWidth: .infinity)
            .background(
                RoundedRectangle(cornerRadius: tokens.radii.lg)
                    .fill(tokens.colors.accentSoft)
            )
            .id(guess.lineIdx)
            .accessibilityAddTraits(.isHeader)
        }
    }

    private var lineTicks: some View {
        let total = env.totalLines(forShabadId: guess.shabadId)
        return HStack(spacing: 4) {
            ForEach(0..<total, id: \.self) { idx in
                tick(for: idx)
            }
        }
        .accessibilityHidden(true)
    }

    private func tick(for idx: Int) -> some View {
        let color: Color
        if idx == guess.lineIdx {
            color = tokens.colors.accent
        } else if idx < guess.lineIdx {
            color = tokens.colors.ink3
        } else {
            color = tokens.colors.rule
        }
        return Capsule()
            .fill(color)
            .frame(height: 3)
    }
}

#Preview("KaraokeView · paper") {
    KaraokeView(guess: PreviewSample.guess(lineIdx: 2))
        .environment(AppEnvironment.preview())
        .previewTheme(.paper)
}

#Preview("KaraokeView · darbar") {
    KaraokeView(guess: PreviewSample.guess(lineIdx: 2))
        .environment(AppEnvironment.preview(theme: .darbar))
        .previewTheme(.darbar)
}

#Preview("KaraokeView · mool") {
    KaraokeView(guess: PreviewSample.guess(lineIdx: 2))
        .environment(AppEnvironment.preview(theme: .mool))
        .previewTheme(.mool)
}

private enum PreviewSample {
    static func guess(lineIdx: Int) -> LineGuess {
        LineGuess(
            chunk: AsrChunk(start: 0, end: 5, text: ""),
            shabadId: 1789,
            lineIdx: lineIdx,
            confidence: 89.1,
            isCommitted: true
        )
    }
}
