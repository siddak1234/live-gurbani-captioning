//
//  CastHintSheet.swift
//  GurbaniCaptioningApp · SangatApp · Features/Cast
//
//  Bottom sheet shown when the Sevadar taps the dock's "Cast" button.
//
//  Why a hint sheet and not a real action button: AirPlay routing is
//  owned by the OS (Control Center → Screen Mirroring), not by an
//  app-level affordance. Our Cast button can't *start* mirroring;
//  what it can do is tell the user where to find the system control,
//  AND confirm whether an external display is already attached so
//  the user knows the projector will pick up the cast view
//  automatically.
//
//  Two states:
//    - `isConnected == false`: instruct the user how to start
//      mirroring (Control Center, or Simulator's Device menu in
//      development).
//    - `isConnected == true`: confirm the projector is already
//      receiving the cast view; no action needed.

import SwiftUI

public struct CastHintSheet: View {

    @Environment(\.themeTokens) private var tokens
    @Environment(\.dismiss) private var dismiss

    public let isConnected: Bool

    public init(isConnected: Bool) {
        self.isConnected = isConnected
    }

    public var body: some View {
        ZStack(alignment: .top) {
            tokens.colors.bg.ignoresSafeArea()

            VStack(alignment: .leading, spacing: tokens.spacing.lg) {
                header

                if isConnected {
                    connectedBody
                } else {
                    instructionsBody
                }

                Spacer()

                Button {
                    dismiss()
                } label: {
                    Text(isConnected ? "Got it" : "OK")
                        .font(tokens.type.sans.weight(.semibold))
                        .tracking(0.4)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, tokens.spacing.md + 2)
                        .background(tokens.colors.ink, in: Capsule())
                        .foregroundStyle(tokens.colors.bg)
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("cast.hint.dismiss")
            }
            .padding(.horizontal, tokens.spacing.edge)
            .padding(.top, tokens.spacing.xl)
            .padding(.bottom, tokens.spacing.lg)
        }
        .accessibilityIdentifier("cast.hint.root")
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: tokens.spacing.xs) {
            HStack(spacing: tokens.spacing.sm) {
                Image(systemName: isConnected ? "checkmark.circle.fill" : "tv.and.hifispeaker.fill")
                    .font(.system(size: 22, weight: .regular))
                    .foregroundStyle(tokens.colors.accent)
                Text(isConnected ? "Casting active" : "Cast to a projector")
                    .font(tokens.type.serifTitle)
                    .foregroundStyle(tokens.colors.ink)
                    .accessibilityAddTraits(.isHeader)
            }
            Text(isConnected
                 ? "Your phone is already mirroring the kirtan line to the connected display."
                 : "Mirror just the current line to a TV or projector for the sangat to read.")
                .font(tokens.type.sansSmall)
                .foregroundStyle(tokens.colors.ink3)
        }
    }

    private var connectedBody: some View {
        VStack(alignment: .leading, spacing: tokens.spacing.md) {
            HintRow(
                number: "•",
                text: "The projector shows only the current line — no chrome, no controls."
            )
            HintRow(
                number: "•",
                text: "The phone keeps the dock and reading view. You stay in control."
            )
            HintRow(
                number: "•",
                text: "Disconnect from Control Center → Screen Mirroring to stop casting."
            )
        }
    }

    private var instructionsBody: some View {
        VStack(alignment: .leading, spacing: tokens.spacing.md) {
            HintRow(
                number: "1",
                text: "Swipe down from the top-right of your phone to open Control Center."
            )
            HintRow(
                number: "2",
                text: "Tap Screen Mirroring and pick an Apple TV or AirPlay-compatible display on the same network."
            )
            HintRow(
                number: "3",
                text: "The projector switches to the cast view automatically — no extra steps in the app."
            )
        }
    }
}

// MARK: - HintRow (inline — single use site)

private struct HintRow: View {

    @Environment(\.themeTokens) private var tokens

    let number: String
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: tokens.spacing.md) {
            Text(number)
                .font(tokens.type.sansCaps)
                .tracking(0.6)
                .foregroundStyle(tokens.colors.accent)
                .frame(width: 18, alignment: .leading)

            Text(text)
                .font(tokens.type.serif)
                .foregroundStyle(tokens.colors.ink2)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

#Preview("CastHintSheet · not connected · paper") {
    CastHintSheet(isConnected: false)
        .environment(AppEnvironment.preview())
        .previewTheme(.paper)
}

#Preview("CastHintSheet · connected · darbar") {
    CastHintSheet(isConnected: true)
        .environment(AppEnvironment.preview(theme: .darbar))
        .previewTheme(.darbar)
}

#Preview("CastHintSheet · not connected · mool") {
    CastHintSheet(isConnected: false)
        .environment(AppEnvironment.preview(theme: .mool))
        .previewTheme(.mool)
}
