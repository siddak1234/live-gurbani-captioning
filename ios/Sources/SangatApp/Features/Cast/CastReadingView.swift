//
//  CastReadingView.swift
//  GurbaniCaptioningApp · SangatApp · Features/Cast
//
//  16:9 chrome-free projector view — design canvas section 09,
//  V1CastView (assets/v1-paper.jsx 1046-1108). Hosted on an
//  external UIWindow attached to the connected UIScreen by
//  `CastSceneCoordinator`. Reads `currentGuess` from the same
//  `AppEnvironment` instance the phone uses, so phone + projector
//  stay in sync without manual mirroring.
//
//  Layout (four anchored regions, no chrome):
//    - top-leading:   Raag · Ang · Author  (sansCaps, ink3)
//    - top-trailing:  • Live · Casting from iPhone  (sans, ink3 + accent dot)
//    - center:        Gurmukhi @ gurmukhiCast (72pt) + English italic
//    - bottom strip:  Line N of M (mono) + ProgressBar + verseId (mono)
//
//  Sangat / Sevadar role does not change the cast view — the projector
//  surface is the same regardless. Theme propagates via env.

import SwiftUI
import GurbaniCaptioning

public struct CastReadingView: View {

    @Environment(AppEnvironment.self) private var env
    @Environment(\.themeTokens) private var tokens

    public init() {}

    public var body: some View {
        ZStack {
            tokens.colors.bg.ignoresSafeArea()

            cornerMeta

            heroLine

            bottomStrip
        }
        .accessibilityIdentifier("cast.root")
    }

    // MARK: - Top corners

    @ViewBuilder
    private var cornerMeta: some View {
        // Meta top-leading + casting status top-trailing. Both at the
        // same vertical offset so the corners read as a balanced row.
        VStack {
            HStack(alignment: .firstTextBaseline) {
                Text(metaLine)
                    .font(tokens.type.sansCaps)
                    .tracking(0.8)
                    .foregroundStyle(tokens.colors.ink3)
                    .accessibilityIdentifier("cast.meta")

                Spacer(minLength: tokens.spacing.xl)

                HStack(spacing: tokens.spacing.xs + 2) {
                    Circle()
                        .fill(tokens.colors.accent)
                        .frame(width: 8, height: 8)
                    Text("Live · Casting from iPhone")
                        .font(tokens.type.sans)
                        .foregroundStyle(tokens.colors.ink3)
                }
                .accessibilityIdentifier("cast.status")
            }
            .padding(.horizontal, 32)
            .padding(.top, 28)

            Spacer()
        }
    }

    // MARK: - Center hero

    @ViewBuilder
    private var heroLine: some View {
        VStack(spacing: 22) {
            if let resolved = currentResolvedLine {
                Text(resolved.gurmukhi)
                    .font(tokens.type.gurmukhiCast)
                    .foregroundStyle(tokens.colors.ink)
                    .multilineTextAlignment(.center)
                    .lineSpacing(8)
                    .accessibilityIdentifier("cast.gurmukhi")

                // Translit row gated on env.translitEnabled — matches
                // the phone's Reading Layers toggle so the projector
                // shows whatever the Sevadar configured.
                if env.translitEnabled,
                   let translit = resolved.transliteration,
                   !translit.isEmpty {
                    Text(translit)
                        .font(tokens.type.serifCastBody)
                        .foregroundStyle(tokens.colors.ink3)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: .infinity)
                        .accessibilityIdentifier("cast.translit")
                }

                // English meaning gated on env.meaningEnabled.
                if env.meaningEnabled,
                   let english = resolved.english,
                   !english.isEmpty {
                    Text(english)
                        .font(tokens.type.serifCastBody)
                        .foregroundStyle(tokens.colors.ink2)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: .infinity)
                        .accessibilityIdentifier("cast.english")
                }
            } else {
                Text("—")
                    .font(tokens.type.gurmukhiCast)
                    .foregroundStyle(tokens.colors.ink3)
            }
        }
        .padding(.horizontal, 80)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        // Center the hero block in the 16:9 view; corners and bottom
        // strip overlay on top via the parent ZStack.
    }

    // MARK: - Bottom progress strip

    @ViewBuilder
    private var bottomStrip: some View {
        VStack {
            Spacer()
            HStack(spacing: 14) {
                Text(lineLabel)
                    .font(tokens.type.mono)
                    .foregroundStyle(tokens.colors.ink3)

                ProgressBar(value: progressValue, height: 2)
                    .frame(maxWidth: .infinity)

                Text(verseIdLabel)
                    .font(tokens.type.mono)
                    .foregroundStyle(tokens.colors.ink3)
            }
            .padding(.horizontal, 32)
            .padding(.bottom, 32)
        }
    }

    // MARK: - Derived display values

    private var currentShabadId: Int? {
        env.captionModel.currentGuess?.shabadId ?? env.captionModel.committedShabadId
    }

    private var currentResolvedLine: ResolvedLine? {
        guard let guess = env.captionModel.currentGuess else { return nil }
        return env.resolveLine(shabadId: guess.shabadId, lineIdx: guess.lineIdx)
    }

    private var metaLine: String {
        guard let sid = currentShabadId else {
            return "Waiting for shabad…".uppercased()
        }
        let m = env.shabadMeta(forShabadId: sid)
        var parts: [String] = ["Raag \(m.raag)"]
        if m.ang > 0 {
            parts.append("Ang \(m.ang)")
        }
        if let author = m.author {
            parts.append(author)
        }
        return parts.joined(separator: " · ")
    }

    private var lineLabel: String {
        guard let guess = env.captionModel.currentGuess else { return "—" }
        let total = env.totalLines(forShabadId: guess.shabadId)
        return "Line \(guess.lineIdx + 1) of \(total)"
    }

    private var verseIdLabel: String {
        currentResolvedLine?.verseId ?? "—"
    }

    private var progressValue: Double {
        guard let guess = env.captionModel.currentGuess else { return 0 }
        let total = env.totalLines(forShabadId: guess.shabadId)
        return Double(guess.lineIdx + 1) / Double(max(total, 1))
    }
}

// MARK: - Previews

#Preview("CastReadingView · paper · 16:9") {
    CastReadingView()
        .environment(AppEnvironment.preview())
        .previewTheme(.paper)
        .frame(width: 960, height: 540)
}

#Preview("CastReadingView · darbar · 16:9") {
    CastReadingView()
        .environment(AppEnvironment.preview(theme: .darbar))
        .previewTheme(.darbar)
        .frame(width: 960, height: 540)
}

#Preview("CastReadingView · mool · 16:9") {
    CastReadingView()
        .environment(AppEnvironment.preview(theme: .mool))
        .previewTheme(.mool)
        .frame(width: 960, height: 540)
}
