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

    /// Forwarded to `TentativeView.onPickManually`. RootView owns the
    /// single `ShabadPickerView` sheet; ReadingHost is the conduit.
    public let onRequestPicker: () -> Void

    /// M5.6: long-press on a committed reading view → "this isn't
    /// right" surface. RootView owns the `WrongShabadSheet`
    /// presentation and the correction-event emission; ReadingHost
    /// just relays the gesture with the currently-committed shabad.
    public let onRequestWrongShabad: (Int) -> Void

    public init(
        onRequestPicker: @escaping () -> Void = {},
        onRequestWrongShabad: @escaping (Int) -> Void = { _ in }
    ) {
        self.onRequestPicker = onRequestPicker
        self.onRequestWrongShabad = onRequestWrongShabad
    }

    public var body: some View {
        switch env.captionModel.state {
        case .listening:
            ListeningView(onPickManually: onRequestPicker)

        case .tentative(let shabadId):
            TentativeView(shabadId: shabadId, onPickManually: onRequestPicker)

        case .committed(let shabadId):
            committedView(shabadId: shabadId)
        }
    }

    @ViewBuilder
    private func committedView(shabadId: Int) -> some View {
        if let guess = env.captionModel.currentGuess, guess.isCommitted {
            // Long-press on the active reading view in any layout
            // surfaces the wrong-shabad sheet. Minimum duration is
            // 0.55s — short enough to feel responsive, long enough
            // not to fire on momentary touches while reading.
            // `HeroLineView`, `KaraokeView`, and `FullShabadView` have
            // no competing gestures (verified at M5.6 audit), so this
            // attaches cleanly.
            Group {
                switch env.readingLayout {
                case .hero:
                    HeroLineView(guess: guess)
                case .karaoke:
                    KaraokeView(guess: guess)
                case .full:
                    FullShabadView(guess: guess)
                }
            }
            .contentShape(Rectangle())
            .onLongPressGesture(minimumDuration: 0.55) {
                env.haptics.play(.warning)
                onRequestWrongShabad(guess.shabadId)
            }
            .accessibilityAction(named: Text("Flag wrong shabad")) {
                onRequestWrongShabad(guess.shabadId)
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
