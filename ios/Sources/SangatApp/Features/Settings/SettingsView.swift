//
//  SettingsView.swift
//  GurbaniCaptioningApp · SangatApp · Features/Settings
//
//  Created for the Sangat iOS app, M5.4 slice — appearance + reading
//  layout + reading layers. Onboarding/mode/corrections sections are
//  intentionally deferred to their own milestones.
//
//  Bindings go straight to `AppEnvironment`'s observable properties; the
//  env's `didSet` hooks persist to `Preferences` so changes survive a
//  relaunch. The screen has no view-model of its own — it is a thin
//  binding surface, and that simplicity is the architecture.

import SwiftUI

/// Bottom-sheet style settings panel for theme, reading layout, and
/// reading layers. Presented as a `.sheet` from `IdleView` (the
/// natural "stop and configure" moment). Reading screens don't show
/// chrome by design — section 05 of the design canvas treats settings
/// as a separate surface, not a reading overlay.
public struct SettingsView: View {

    @Environment(AppEnvironment.self) private var env
    @Environment(\.themeTokens) private var tokens
    @Environment(\.dismiss) private var dismiss

    public init() {}

    public var body: some View {
        @Bindable var env = env

        ZStack(alignment: .top) {
            tokens.colors.bg.ignoresSafeArea()

            ScrollView {
                VStack(alignment: .leading, spacing: tokens.spacing.xl) {
                    header

                    appearanceSection(themeBinding: $env.theme)

                    layoutSection(layoutBinding: $env.readingLayout)

                    layersSection(
                        translitBinding: $env.translitEnabled,
                        meaningBinding: $env.meaningEnabled
                    )
                }
                .padding(.horizontal, tokens.spacing.edge)
                .padding(.top, tokens.spacing.xxl)
                .padding(.bottom, tokens.spacing.xxl)
            }
        }
        .accessibilityIdentifier("settings.root")
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: tokens.spacing.xs) {
                Text("Settings")
                    .font(tokens.type.gurmukhiLarge.weight(.semibold))
                    .foregroundStyle(tokens.colors.ink)
                    .accessibilityAddTraits(.isHeader)

                Text("Theme, reading layout, and which layers show under each line.")
                    .font(tokens.type.sansSmall)
                    .foregroundStyle(tokens.colors.ink3)
            }

            Spacer()

            Button {
                env.haptics.play(.selection)
                dismiss()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 28, weight: .regular))
                    .foregroundStyle(tokens.colors.ink3)
            }
            .accessibilityLabel("Close settings")
            .accessibilityIdentifier("settings.close")
        }
    }

    // MARK: - Appearance

    private func appearanceSection(themeBinding: Binding<Theme>) -> some View {
        VStack(alignment: .leading, spacing: tokens.spacing.md) {
            SectionLabel("Appearance")

            VStack(spacing: 0) {
                ForEach(Theme.allCases) { option in
                    ThemeOptionRow(
                        theme: option,
                        isSelected: themeBinding.wrappedValue == option
                    ) {
                        env.haptics.play(.selection)
                        themeBinding.wrappedValue = option
                    }
                    if option != Theme.allCases.last {
                        Divider().background(tokens.colors.ruleSoft)
                    }
                }
            }
            .background(tokens.colors.surface)
            .clipShape(RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous)
                    .stroke(tokens.colors.rule, lineWidth: 0.5)
            )
        }
    }

    // MARK: - Reading layout

    private func layoutSection(layoutBinding: Binding<ReadingLayout>) -> some View {
        VStack(alignment: .leading, spacing: tokens.spacing.md) {
            SectionLabel("Reading layout")

            VStack(spacing: 0) {
                ForEach(ReadingLayout.allCases) { option in
                    LayoutOptionRow(
                        layout: option,
                        isSelected: layoutBinding.wrappedValue == option
                    ) {
                        env.haptics.play(.selection)
                        layoutBinding.wrappedValue = option
                    }
                    if option != ReadingLayout.allCases.last {
                        Divider().background(tokens.colors.ruleSoft)
                    }
                }
            }
            .background(tokens.colors.surface)
            .clipShape(RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous)
                    .stroke(tokens.colors.rule, lineWidth: 0.5)
            )
        }
    }

    // MARK: - Reading layers

    private func layersSection(
        translitBinding: Binding<Bool>,
        meaningBinding: Binding<Bool>
    ) -> some View {
        VStack(alignment: .leading, spacing: tokens.spacing.md) {
            SectionLabel("Reading layers")

            VStack(spacing: 0) {
                ToggleRow(
                    title: "Transliteration",
                    subtitle: "Latin spelling under each Gurmukhi line",
                    isOn: translitBinding
                )
                .padding(.horizontal, tokens.spacing.md)
                .padding(.vertical, tokens.spacing.sm)

                Divider().background(tokens.colors.ruleSoft)

                ToggleRow(
                    title: "English meaning",
                    subtitle: "Translation under each line",
                    isOn: meaningBinding
                )
                .padding(.horizontal, tokens.spacing.md)
                .padding(.vertical, tokens.spacing.sm)
            }
            .background(tokens.colors.surface)
            .clipShape(RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous)
                    .stroke(tokens.colors.rule, lineWidth: 0.5)
            )
        }
    }
}

// MARK: - Theme option row

private struct ThemeOptionRow: View {

    @Environment(\.themeTokens) private var tokens

    let theme: Theme
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: tokens.spacing.md) {
                // Preview swatch — show the theme's bg + accent so users see
                // what they're picking without applying it first.
                ZStack {
                    RoundedRectangle(cornerRadius: tokens.radii.sm, style: .continuous)
                        .fill(theme.tokens.colors.bg)
                    Circle()
                        .fill(theme.tokens.colors.accent)
                        .frame(width: 18, height: 18)
                }
                .frame(width: 40, height: 40)
                .overlay(
                    RoundedRectangle(cornerRadius: tokens.radii.sm, style: .continuous)
                        .stroke(tokens.colors.rule, lineWidth: 0.5)
                )

                VStack(alignment: .leading, spacing: 2) {
                    Text(theme.displayName)
                        .font(tokens.type.serif.weight(.semibold))
                        .foregroundStyle(tokens.colors.ink)
                    Text(theme.subtitle)
                        .font(tokens.type.sansSmall)
                        .foregroundStyle(tokens.colors.ink3)
                }

                Spacer()

                if isSelected {
                    Image(systemName: "checkmark")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(tokens.colors.accent)
                        .accessibilityHidden(true)
                }
            }
            .padding(.horizontal, tokens.spacing.md)
            .padding(.vertical, tokens.spacing.sm)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(theme.displayName). \(theme.subtitle).")
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .accessibilityIdentifier("settings.theme.\(theme.rawValue)")
    }
}

// MARK: - Layout option row

private struct LayoutOptionRow: View {

    @Environment(\.themeTokens) private var tokens

    let layout: ReadingLayout
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: tokens.spacing.md) {
                LayoutGlyph(layout: layout)
                    .frame(width: 40, height: 40)

                VStack(alignment: .leading, spacing: 2) {
                    Text(layout.displayName)
                        .font(tokens.type.serif.weight(.semibold))
                        .foregroundStyle(tokens.colors.ink)
                    Text(layout.subtitle)
                        .font(tokens.type.sansSmall)
                        .foregroundStyle(tokens.colors.ink3)
                }

                Spacer()

                if isSelected {
                    Image(systemName: "checkmark")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(tokens.colors.accent)
                        .accessibilityHidden(true)
                }
            }
            .padding(.horizontal, tokens.spacing.md)
            .padding(.vertical, tokens.spacing.sm)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(layout.displayName). \(layout.subtitle).")
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .accessibilityIdentifier("settings.layout.\(layout.rawValue)")
    }
}

// Tiny pictographic glyph for each reading layout — one big bar, three
// stacked bars (with middle accented), or a column of small bars.
private struct LayoutGlyph: View {

    @Environment(\.themeTokens) private var tokens

    let layout: ReadingLayout

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: tokens.radii.sm, style: .continuous)
                .fill(tokens.colors.bgSoft)

            switch layout {
            case .hero:
                Capsule()
                    .fill(tokens.colors.ink)
                    .frame(width: 22, height: 6)
            case .karaoke:
                VStack(spacing: 4) {
                    Capsule().fill(tokens.colors.ink3).frame(width: 18, height: 3)
                    Capsule().fill(tokens.colors.ink).frame(width: 22, height: 4)
                    Capsule().fill(tokens.colors.ink3).frame(width: 18, height: 3)
                }
            case .full:
                VStack(spacing: 3) {
                    ForEach(0 ..< 5, id: \.self) { idx in
                        Capsule()
                            .fill(idx == 2 ? tokens.colors.ink : tokens.colors.ink3.opacity(0.6))
                            .frame(width: idx == 2 ? 22 : 18, height: 2.5)
                    }
                }
            }
        }
        .overlay(
            RoundedRectangle(cornerRadius: tokens.radii.sm, style: .continuous)
                .stroke(tokens.colors.rule, lineWidth: 0.5)
        )
    }
}

// MARK: - Previews

#Preview("SettingsView · paper") {
    SettingsView()
        .environment(AppEnvironment.preview())
        .previewTheme(.paper)
}

#Preview("SettingsView · darbar") {
    SettingsView()
        .environment(AppEnvironment.preview(theme: .darbar))
        .previewTheme(.darbar)
}

#Preview("SettingsView · mool") {
    SettingsView()
        .environment(AppEnvironment.preview(theme: .mool))
        .previewTheme(.mool)
}
