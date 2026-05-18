//
//  SevadarDock.swift
//  GurbaniCaptioningApp · SangatApp · Features/Sevadar
//
//  Bottom-anchored floating control deck for Sevadar mode. Composed at
//  RootView level via `.safeAreaInset(edge: .bottom)` so the reading
//  view content (HeroLineView / KaraokeView / FullShabadView) auto-
//  lays out above the dock — no state loss, no remount, single state
//  machine. `Features/Sevadar/` never imports `Features/Sangat/`.
//
//  Six controls, in design-canvas order (assets/v1-paper.jsx 786-797):
//    Row 1: Line −1  |  Pause auto (primary)  |  Line +1
//    Row 2: Pick shabad (flex 1.4)  |  Cast  |  Lock
//  Plus a "Sangat sees" preview strip at the bottom.
//
//  The dock takes its actions as callbacks rather than reading
//  AppEnvironment directly because the actions are owned by RootView
//  (so it can drive `.sheet(isPresented:)` for the picker).

import SwiftUI

public struct SevadarDock: View {

    @Environment(\.themeTokens) private var tokens

    public let isPaused: Bool
    public let onNudgeBack: () -> Void
    public let onTogglePause: () -> Void
    public let onNudgeForward: () -> Void
    public let onPick: () -> Void
    public let onCast: () -> Void
    public let onEditLayers: () -> Void

    public let layersSummary: String

    public init(
        isPaused: Bool,
        layersSummary: String,
        onNudgeBack: @escaping () -> Void,
        onTogglePause: @escaping () -> Void,
        onNudgeForward: @escaping () -> Void,
        onPick: @escaping () -> Void,
        onCast: @escaping () -> Void,
        onEditLayers: @escaping () -> Void
    ) {
        self.isPaused = isPaused
        self.layersSummary = layersSummary
        self.onNudgeBack = onNudgeBack
        self.onTogglePause = onTogglePause
        self.onNudgeForward = onNudgeForward
        self.onPick = onPick
        self.onCast = onCast
        self.onEditLayers = onEditLayers
    }

    public var body: some View {
        VStack(spacing: tokens.spacing.sm + 2) {
            // Row 1: arrow icons flank a wider primary Pause/Resume.
            // Arrows are icon-only — same nudge wiring as before, less
            // visual noise. Equal vertical padding to the line-2 row.
            HStack(spacing: tokens.spacing.sm) {
                SevadarIconButton(systemName: "arrow.left", action: onNudgeBack)
                SevadarButton(
                    isPaused ? "Resume" : "Pause auto",
                    style: .primary,
                    action: onTogglePause
                )
                SevadarIconButton(systemName: "arrow.right", action: onNudgeForward)
            }

            // Row 2: two equal-width action buttons. Lock was removed
            // (it had no real surface to lock — the reading view's tap
            // gating lands with a future milestone).
            HStack(spacing: tokens.spacing.sm) {
                SevadarButton("Pick shabad", action: onPick)
                SevadarButton("Cast", action: onCast)
            }

            sangatSeesStrip
                .padding(.top, tokens.spacing.xs)
        }
        .padding(tokens.spacing.md - 2)
        .background(tokens.colors.surface, in: RoundedRectangle(cornerRadius: tokens.radii.xl, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: tokens.radii.xl, style: .continuous)
                .stroke(tokens.colors.rule, lineWidth: 0.5)
        )
        .shadow(
            color: Color.black.opacity(0.18),
            radius: 18, x: 0, y: 10
        )
        .padding(.horizontal, tokens.spacing.md)
        .padding(.bottom, tokens.spacing.sm)
        .accessibilityIdentifier("sevadar.dock")
    }

    private var sangatSeesStrip: some View {
        Button(action: onEditLayers) {
            HStack(spacing: tokens.spacing.sm) {
                RoundedRectangle(cornerRadius: tokens.radii.xs, style: .continuous)
                    .fill(tokens.colors.accentSoft)
                    .overlay(
                        RoundedRectangle(cornerRadius: tokens.radii.xs, style: .continuous)
                            .stroke(tokens.colors.rule, lineWidth: 0.5)
                    )
                    .frame(width: 28, height: 28)

                VStack(alignment: .leading, spacing: 1) {
                    Text("Sangat sees")
                        .font(tokens.type.sansCaps)
                        .tracking(0.6)
                        .foregroundStyle(tokens.colors.ink3)
                    Text(layersSummary)
                        .font(tokens.type.serif)
                        .foregroundStyle(tokens.colors.ink2)
                        .lineLimit(1)
                }

                Spacer(minLength: tokens.spacing.xs)

                Text("Edit ›")
                    .font(tokens.type.sans.weight(.semibold))
                    .foregroundStyle(tokens.colors.accent)
            }
            .padding(.horizontal, tokens.spacing.sm + 2)
            .padding(.vertical, tokens.spacing.sm)
            .background(tokens.colors.bgSoft, in: RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier("sevadar.dock.sangatSees")
    }
}

#Preview("SevadarDock · paper · running") {
    VStack {
        Spacer()
        SevadarDock(
            isPaused: false,
            layersSummary: "Hero · Translit on · Meaning off",
            onNudgeBack: {}, onTogglePause: {}, onNudgeForward: {},
            onPick: {}, onCast: {}, onEditLayers: {}
        )
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .background(ThemeTokens.paper.colors.bg)
    .previewTheme(.paper)
}

#Preview("SevadarDock · paper · paused") {
    VStack {
        Spacer()
        SevadarDock(
            isPaused: true,
            layersSummary: "Karaoke · Translit on · Meaning on",
            onNudgeBack: {}, onTogglePause: {}, onNudgeForward: {},
            onPick: {}, onCast: {}, onEditLayers: {}
        )
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .background(ThemeTokens.paper.colors.bg)
    .previewTheme(.paper)
}

#Preview("SevadarDock · darbar") {
    VStack {
        Spacer()
        SevadarDock(
            isPaused: false,
            layersSummary: "Hero · Translit on · Meaning off",
            onNudgeBack: {}, onTogglePause: {}, onNudgeForward: {},
            onPick: {}, onCast: {}, onEditLayers: {}
        )
    }
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .background(ThemeTokens.darbar.colors.bg)
    .previewTheme(.darbar)
}
