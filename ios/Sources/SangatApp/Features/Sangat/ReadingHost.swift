//
//  ReadingHost.swift
//  GurbaniCaptioningApp · SangatApp · Features/Sangat
//
//  Created for the Sangat iOS app, M5.2 (Sangat reading).
//
//  Dispatcher for the Sangat reading surface. Branches on engine state
//  and (when committed) the user's reading-layout preference.
//
//   state                                    →   view
//   ──────────────────────────────────────────────────────────────
//   .listening                                   ListeningView
//   .tentative(sid)                              TentativeView
//   .committed(sid) + guess.isCommitted          {Hero, Karaoke, Full}
//   .committed(sid) without guess                Waiting-for-line stub
//
//  Idle (before the user has tapped Listen) is handled one level up in
//  RootView, since the dispatch needs `isRunning` too.

import SwiftUI
import GurbaniCaptioning

public struct ReadingHost: View {

    @Environment(AppEnvironment.self) private var env
    @Environment(\.themeTokens) private var tokens

    public init() {}

    public var body: some View {
        switch env.captionModel.state {
        case .listening:
            ListeningView()

        case .tentative(let shabadId):
            TentativeView(shabadId: shabadId)

        case .committed(let shabadId):
            committedView(shabadId: shabadId)
        }
    }

    @ViewBuilder
    private func committedView(shabadId: Int) -> some View {
        if let guess = env.captionModel.currentGuess, guess.isCommitted {
            switch env.readingLayout {
            case .hero:
                HeroLineView(guess: guess)
            case .karaoke:
                KaraokeView(guess: guess)
            case .full:
                FullShabadView(guess: guess)
            }
        } else {
            waitingForLine(shabadId: shabadId)
        }
    }

    private func waitingForLine(shabadId: Int) -> some View {
        let meta = env.shabadMeta(forShabadId: shabadId)
        return VStack(spacing: tokens.spacing.lg) {
            ShabadMetaHeader(
                raag: meta.raag,
                ang: meta.ang,
                author: meta.author,
                authorGurmukhi: meta.authorGurmukhi
            )
            .padding(.top, tokens.spacing.lg)

            StatePill(state: .committed(shabadId: shabadId))

            Spacer()

            ProgressView()
                .scaleEffect(1.2)
                .tint(tokens.colors.ink2)

            Text("Locking onto the current line…")
                .font(tokens.type.sansSmall)
                .foregroundStyle(tokens.colors.ink3)

            Spacer()
        }
    }
}

#Preview("ReadingHost · paper hero") {
    ReadingHost()
        .environment(AppEnvironment.preview())
        .previewTheme(.paper)
}
