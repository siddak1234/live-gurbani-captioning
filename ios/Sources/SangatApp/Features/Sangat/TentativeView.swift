//
//  TentativeView.swift
//  GurbaniCaptioningApp · SangatApp · Features/Sangat
//
//  Created for the Sangat iOS app, M5.2 (Sangat reading).
//
//  Shown while the engine is in `.tentative(shabadId:)` — it has a
//  candidate but hasn't yet seen enough consecutive votes to commit.
//  Surface: faded first line of the candidate shabad + a "3 of 5 chunks
//  agree" progress strip + a "Pick manually" affordance (wired in M5.3).
//
//  Vote build is hardcoded to 3/5 in M5.2; M5.3 will read real vote
//  buffer state when ShabadStateMachine exposes it through CaptionSource.

import SwiftUI
import GurbaniCaptioning

public struct TentativeView: View {

    @Environment(AppEnvironment.self) private var env
    @Environment(\.themeTokens) private var tokens

    public let shabadId: Int

    /// Opens the same `ShabadPickerView` the Sevadar dock uses. Passed
    /// down from `ReadingHost` (which gets it from `RootView`) so there
    /// is a single source of truth for the picker sheet — no duplicate
    /// `.sheet` modifiers, one shared picker for both Sangat and Sevadar.
    public let onPickManually: () -> Void

    public init(shabadId: Int, onPickManually: @escaping () -> Void = {}) {
        self.shabadId = shabadId
        self.onPickManually = onPickManually
    }

    public var body: some View {
        VStack(spacing: tokens.spacing.lg) {
            StatePill(state: .tentative(shabadId: shabadId))
                .padding(.top, tokens.spacing.lg)

            if let meta = headerMeta {
                ShabadMetaHeader(
                    raag: meta.raag,
                    ang: meta.ang,
                    author: meta.author,
                    authorGurmukhi: meta.authorGurmukhi,
                    compact: true
                )
                .opacity(0.7)
            }

            Spacer()

            candidateLine
                .padding(.horizontal, tokens.spacing.edge)

            Spacer()

            ProgressDots(filled: 3, total: 5, label: "3 of 5 chunks agree")

            pickManuallyButton
                .padding(.bottom, tokens.spacing.lg)
        }
        .padding(.horizontal, tokens.spacing.edge)
    }

    @ViewBuilder
    private var candidateLine: some View {
        if let resolved = env.resolveLine(shabadId: shabadId, lineIdx: 0) {
            GurbaniText(
                gurmukhi: resolved.gurmukhi,
                translit: env.translitEnabled ? resolved.transliteration : nil,
                english: nil,
                size: .large
            )
            .opacity(0.55)
            .accessibilityLabel(
                Text("Tentative match: \(resolved.gurmukhi)")
            )
        } else {
            Text("—")
                .font(tokens.type.gurmukhiLarge)
                .foregroundStyle(tokens.colors.ink3)
        }
    }

    private var pickManuallyButton: some View {
        Button {
            env.haptics.play(.selection)
            onPickManually()
        } label: {
            HStack(spacing: tokens.spacing.xs) {
                Image(systemName: "magnifyingglass")
                Text("Pick manually")
            }
            .font(tokens.type.sans.weight(.semibold))
            .foregroundStyle(tokens.colors.ink2)
            .padding(.horizontal, tokens.spacing.md)
            .padding(.vertical, tokens.spacing.sm)
            .background(Capsule().stroke(tokens.colors.rule, lineWidth: 1))
        }
        .accessibilityLabel("Pick the shabad manually")
    }

    private var headerMeta: ShabadMeta? {
        env.shabadMeta(forShabadId: shabadId)
    }
}

#Preview("TentativeView · paper") {
    TentativeView(shabadId: 1789)
        .environment(AppEnvironment.preview())
        .previewTheme(.paper)
}

#Preview("TentativeView · darbar") {
    TentativeView(shabadId: 1789)
        .environment(AppEnvironment.preview(theme: .darbar))
        .previewTheme(.darbar)
}

#Preview("TentativeView · mool") {
    TentativeView(shabadId: 1789)
        .environment(AppEnvironment.preview(theme: .mool))
        .previewTheme(.mool)
}
