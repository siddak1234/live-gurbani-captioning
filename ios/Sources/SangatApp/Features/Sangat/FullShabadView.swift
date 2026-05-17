//
//  FullShabadView.swift
//  GurbaniCaptioningApp · SangatApp · Features/Sangat
//
//  Created for the Sangat iOS app, M5.2 (Sangat reading).
//
//  Committed-state reading layout #3 — full shabad scrolled, active line
//  highlighted with the soft-accent card, past lines faded. Auto-scrolls
//  to keep the active line in view as the engine advances.

import SwiftUI
import GurbaniCaptioning

public struct FullShabadView: View {

    @Environment(AppEnvironment.self) private var env
    @Environment(\.themeTokens) private var tokens

    public let guess: LineGuess

    public init(guess: LineGuess) {
        self.guess = guess
    }

    public var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: tokens.spacing.md) {
                    header
                        .padding(.bottom, tokens.spacing.sm)

                    let total = env.totalLines(forShabadId: guess.shabadId)
                    ForEach(0..<total, id: \.self) { idx in
                        lineRow(at: idx)
                            .id(idx)
                    }
                }
                .padding(.horizontal, tokens.spacing.edge)
                .padding(.vertical, tokens.spacing.lg)
            }
            .onAppear {
                proxy.scrollTo(guess.lineIdx, anchor: .center)
            }
            .onChange(of: guess.lineIdx) { _, newIdx in
                withAnimation(.easeInOut(duration: 0.3)) {
                    proxy.scrollTo(newIdx, anchor: .center)
                }
            }
        }
    }

    @ViewBuilder
    private var header: some View {
        let meta = env.shabadMeta(forShabadId: guess.shabadId)
        VStack(alignment: .leading, spacing: tokens.spacing.sm) {
            ShabadMetaHeader(
                raag: meta.raag,
                ang: meta.ang,
                author: meta.author,
                authorGurmukhi: meta.authorGurmukhi,
                compact: true
            )
            HStack {
                StatePill(state: .committed(shabadId: guess.shabadId))
                Spacer()
                Text("Auto-scroll")
                    .font(tokens.type.sansCaps)
                    .tracking(0.4)
                    .foregroundStyle(tokens.colors.ink3)
            }
        }
    }

    @ViewBuilder
    private func lineRow(at idx: Int) -> some View {
        if let resolved = env.resolveLine(shabadId: guess.shabadId, lineIdx: idx) {
            let isActive = idx == guess.lineIdx
            let isPast = idx < guess.lineIdx

            HStack(alignment: .top, spacing: tokens.spacing.sm) {
                Rectangle()
                    .fill(isActive ? tokens.colors.accent : Color.clear)
                    .frame(width: 3)
                    .clipShape(Capsule())

                VStack(alignment: .leading, spacing: tokens.spacing.xs) {
                    Text(resolved.gurmukhi)
                        .font(isActive ? tokens.type.gurmukhiLarge : tokens.type.gurmukhi)
                        .foregroundStyle(tokens.colors.ink)
                        .multilineTextAlignment(.leading)

                    if isActive, let translit = resolved.transliteration, env.translitEnabled {
                        Text(translit)
                            .font(tokens.type.serifItalic)
                            .foregroundStyle(tokens.colors.ink2)
                    }
                }
                .opacity(isPast ? 0.42 : 1.0)

                Spacer(minLength: 0)
            }
            .padding(.vertical, isActive ? tokens.spacing.md : tokens.spacing.sm)
            .padding(.horizontal, tokens.spacing.sm)
            .background(
                RoundedRectangle(cornerRadius: tokens.radii.md)
                    .fill(isActive ? tokens.colors.accentSoft : Color.clear)
            )
            .accessibilityElement(children: .combine)
            .accessibilityAddTraits(isActive ? .isHeader : [])
        }
    }
}

#Preview("FullShabadView · paper") {
    FullShabadView(guess: PreviewSample.guess(lineIdx: 3))
        .environment(AppEnvironment.preview())
        .previewTheme(.paper)
}

#Preview("FullShabadView · darbar") {
    FullShabadView(guess: PreviewSample.guess(lineIdx: 3))
        .environment(AppEnvironment.preview(theme: .darbar))
        .previewTheme(.darbar)
}

#Preview("FullShabadView · mool") {
    FullShabadView(guess: PreviewSample.guess(lineIdx: 3))
        .environment(AppEnvironment.preview(theme: .mool))
        .previewTheme(.mool)
}

private enum PreviewSample {
    static func guess(lineIdx: Int) -> LineGuess {
        LineGuess(
            chunk: AsrChunk(start: 0, end: 5, text: ""),
            shabadId: 1789,
            lineIdx: lineIdx,
            confidence: 86.7,
            isCommitted: true
        )
    }
}
