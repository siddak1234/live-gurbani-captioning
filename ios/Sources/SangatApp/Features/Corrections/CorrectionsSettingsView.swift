//
//  CorrectionsSettingsView.swift
//  GurbaniCaptioningApp · SangatApp · Features/Corrections
//
//  Created for the Sangat iOS app, M5.6 (Correction loop surfaces).
//
//  The trust-hinge surface. Controls whether the app records the user's
//  shabad / line corrections to the on-device `CorrectionLog`. Default
//  is OFF — explicit user opt-in is required.
//
//  Wired against `env.preferences.correctionsOptIn` (existing M5.1 key)
//  and `env.correctionLog`. Today the log is `NoopCorrectionLog`, so
//  the on-disk implications of the toggle are zero; the toggle still
//  matters because emission sites in RootView gate their `record(...)`
//  calls on this preference, so a future durable log (M5.8) cannot
//  silently start collecting without the user having flipped this.

import SwiftUI

public struct CorrectionsSettingsView: View {

    @Environment(AppEnvironment.self) private var env
    @Environment(\.themeTokens) private var tokens
    @Environment(\.dismiss) private var dismiss

    /// Mirror of the persisted preference. Reads on appear; writes
    /// flow through to `preferences.correctionsOptIn` on `didSet`.
    @State private var optedIn: Bool = false

    /// Display-only snapshot of the log's count. Refreshed on appear
    /// and on Clear. With `NoopCorrectionLog` this stays at 0; when
    /// a durable log lands (M5.8) the same view shows real counts.
    @State private var savedCount: Int = 0

    public init() {}

    public var body: some View {
        ZStack(alignment: .top) {
            tokens.colors.bg.ignoresSafeArea()

            ScrollView {
                VStack(alignment: .leading, spacing: tokens.spacing.xl) {
                    header

                    explainerSection

                    optInSection

                    storageSection

                    Spacer(minLength: tokens.spacing.xxl)
                }
                .padding(.horizontal, tokens.spacing.edge)
                .padding(.top, tokens.spacing.xxl)
                .padding(.bottom, tokens.spacing.xxl)
            }
        }
        .onAppear {
            optedIn = env.preferences.correctionsOptIn
            savedCount = env.correctionLog.approximateCount
        }
        .onChange(of: optedIn) { _, newValue in
            env.preferences.correctionsOptIn = newValue
            AppLogger.app.info("Corrections opt-in toggled → \(newValue, privacy: .public)")
        }
        .accessibilityIdentifier("correctionsSettings.root")
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: tokens.spacing.xs) {
                Text("Improve detection")
                    .font(tokens.type.gurmukhiLarge.weight(.semibold))
                    .foregroundStyle(tokens.colors.ink)
                    .accessibilityAddTraits(.isHeader)

                Text("Your corrections help us train the engine.")
                    .font(tokens.type.sansSmall)
                    .foregroundStyle(tokens.colors.ink3)
            }

            Spacer()

            Button {
                env.haptics.play(.selection)
                dismiss()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 26))
                    .foregroundStyle(tokens.colors.ink3)
            }
            .accessibilityLabel("Close")
            .accessibilityIdentifier("correctionsSettings.close")
        }
    }

    private var explainerSection: some View {
        VStack(alignment: .leading, spacing: tokens.spacing.sm) {
            SectionLabel("How it works")

            VStack(alignment: .leading, spacing: tokens.spacing.sm + 2) {
                explainerRow(
                    icon: "hand.tap.fill",
                    title: "You correct a mistake",
                    body: "Long-press a wrong line, pick a runner-up, or flag a past session."
                )
                explainerRow(
                    icon: "lock.shield.fill",
                    title: "It stays on this device",
                    body: "Corrections are saved locally. We do not upload anything in this build."
                )
                explainerRow(
                    icon: "sparkles",
                    title: "Future training",
                    body: "When the next engine is fine-tuned, your saved corrections improve it."
                )
            }
            .padding(.horizontal, tokens.spacing.md)
            .padding(.vertical, tokens.spacing.md)
            .background(tokens.colors.surface)
            .clipShape(RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous)
                    .stroke(tokens.colors.rule, lineWidth: 0.5)
            )
        }
    }

    private func explainerRow(icon: String, title: String, body: String) -> some View {
        HStack(alignment: .top, spacing: tokens.spacing.md) {
            Image(systemName: icon)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(tokens.colors.accent)
                .frame(width: 22, alignment: .center)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(tokens.type.sans.weight(.semibold))
                    .foregroundStyle(tokens.colors.ink)
                Text(body)
                    .font(tokens.type.sansSmall)
                    .foregroundStyle(tokens.colors.ink3)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var optInSection: some View {
        VStack(alignment: .leading, spacing: tokens.spacing.sm) {
            SectionLabel("Save my corrections")

            VStack(spacing: 0) {
                ToggleRow(
                    title: "Help improve detection",
                    subtitle: "Save shabad / line corrections on this device.",
                    isOn: $optedIn
                )
                .accessibilityIdentifier("correctionsSettings.optIn")
            }
            .background(tokens.colors.surface)
            .clipShape(RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous)
                    .stroke(tokens.colors.rule, lineWidth: 0.5)
            )
        }
    }

    private var storageSection: some View {
        VStack(alignment: .leading, spacing: tokens.spacing.sm) {
            SectionLabel("Storage")

            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Saved corrections")
                            .font(tokens.type.serif)
                            .foregroundStyle(tokens.colors.ink)
                        Text(storageSubtitle)
                            .font(tokens.type.sansSmall)
                            .foregroundStyle(tokens.colors.ink3)
                    }
                    Spacer()
                    Text("\(savedCount)")
                        .font(tokens.type.mono)
                        .foregroundStyle(tokens.colors.ink2)
                }
                .padding(.horizontal, tokens.spacing.md)
                .padding(.vertical, tokens.spacing.md)
                .background(
                    Rectangle()
                        .fill(tokens.colors.ruleSoft)
                        .frame(height: 0.5),
                    alignment: .bottom
                )
                .accessibilityIdentifier("correctionsSettings.count")

                Button {
                    env.haptics.play(.selection)
                    env.correctionLog.clear()
                    savedCount = env.correctionLog.approximateCount
                    AppLogger.app.info("Corrections cleared")
                } label: {
                    HStack {
                        Text("Clear corrections")
                            .font(tokens.type.serif)
                            .foregroundStyle(savedCount == 0 ? tokens.colors.ink3 : tokens.colors.accent)
                        Spacer()
                    }
                    .padding(.horizontal, tokens.spacing.md)
                    .padding(.vertical, tokens.spacing.md)
                }
                .buttonStyle(.plain)
                .disabled(savedCount == 0)
                .accessibilityIdentifier("correctionsSettings.clear")
            }
            .background(tokens.colors.surface)
            .clipShape(RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous)
                    .stroke(tokens.colors.rule, lineWidth: 0.5)
            )
        }
    }

    private var storageSubtitle: String {
        savedCount == 0
            ? "Nothing saved yet on this device."
            : "Stored locally. Audio never leaves your phone."
    }
}

#Preview("CorrectionsSettingsView · paper") {
    CorrectionsSettingsView()
        .environment(AppEnvironment.preview())
        .previewTheme(.paper)
}

#Preview("CorrectionsSettingsView · darbar") {
    CorrectionsSettingsView()
        .environment(AppEnvironment.preview(theme: .darbar))
        .previewTheme(.darbar)
}
