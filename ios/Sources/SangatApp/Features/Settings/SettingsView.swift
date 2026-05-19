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

    @State private var showConfidence: Bool = false
    @State private var showHistory: Bool = false
    @State private var showCorrections: Bool = false

    /// M5.6: when the Sevadar long-presses a history row, we surface a
    /// follow-up sheet so they can supply the correct shabad. Nil →
    /// no sheet; non-nil → the row being retroactively flagged.
    @State private var pendingRetroactiveFlag: PendingRetroactiveFlag?

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

                    if env.mode == .sevadar {
                        sevadarToolsSection
                    }

                    improveDetectionSection
                }
                .padding(.horizontal, tokens.spacing.edge)
                .padding(.top, tokens.spacing.xxl)
                .padding(.bottom, tokens.spacing.xxl)
            }
        }
        .sheet(isPresented: $showConfidence) {
            ConfidenceView(
                onEndorseRunnerUp: { shabadId in
                    recordRunnerUpEndorsement(endorsedShabadId: shabadId)
                    env.haptics.play(.success)
                    env.captionModel.manuallyCommit(shabadId: shabadId)
                }
            )
            .environment(env)
            .environment(\.theme, env.theme)
            .environment(\.themeTokens, env.theme.tokens)
            .preferredColorScheme(env.theme.isDark ? .dark : .light)
        }
        .sheet(isPresented: $showHistory) {
            HistoryView(
                onFlagWrongShabad: { entry in
                    pendingRetroactiveFlag = PendingRetroactiveFlag(entry: entry)
                }
            )
            .environment(env)
            .environment(\.theme, env.theme)
            .environment(\.themeTokens, env.theme.tokens)
            .preferredColorScheme(env.theme.isDark ? .dark : .light)
        }
        .sheet(item: $pendingRetroactiveFlag) { pending in
            // Reusing the picker as the "supply the correct shabad"
            // surface. Same atom, framed for retroactive use.
            WrongShabadSheet(
                predictedShabadId: pending.entry.shabadId,
                onCorrect: { correctedShabadId in
                    recordRetroactiveFlag(
                        entry: pending.entry,
                        correctedShabadId: correctedShabadId
                    )
                    env.haptics.play(.success)
                    pendingRetroactiveFlag = nil
                }
            )
            .environment(env)
            .environment(\.theme, env.theme)
            .environment(\.themeTokens, env.theme.tokens)
            .preferredColorScheme(env.theme.isDark ? .dark : .light)
        }
        .sheet(isPresented: $showCorrections) {
            CorrectionsSettingsView()
                .environment(env)
                .environment(\.theme, env.theme)
                .environment(\.themeTokens, env.theme.tokens)
                .preferredColorScheme(env.theme.isDark ? .dark : .light)
        }
        .accessibilityIdentifier("settings.root")
    }

    // MARK: - Improve detection (M5.6)

    private var improveDetectionSection: some View {
        VStack(alignment: .leading, spacing: tokens.spacing.md) {
            SectionLabel("Improve detection")

            VStack(spacing: 0) {
                SevadarToolRow(
                    icon: "sparkles",
                    title: "Help improve detection",
                    subtitle: env.preferences.correctionsOptIn
                        ? "On · corrections saved on this device"
                        : "Off · corrections discarded"
                ) {
                    env.haptics.play(.selection)
                    showCorrections = true
                }
                .accessibilityIdentifier("settings.corrections")
            }
            .background(tokens.colors.surface)
            .clipShape(RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous)
                    .stroke(tokens.colors.rule, lineWidth: 0.5)
            )
        }
    }

    // MARK: - Correction emission helpers (M5.6)

    /// Same gate every emission site funnels through; mirrors the
    /// helper in `RootView` so the contract is uniform across files.
    private func recordIfOptedIn(_ build: () -> CorrectionEvent) {
        guard env.preferences.correctionsOptIn else {
            AppLogger.corrections.debug("Correction skipped — user has not opted in")
            return
        }
        env.correctionLog.record(build())
    }

    private func recordRunnerUpEndorsement(endorsedShabadId: Int) {
        let currentGuess = env.captionModel.currentGuess
        let predictedId = currentGuess?.shabadId ?? env.captionModel.committedShabadId
        guard let predictedId, predictedId != endorsedShabadId else { return }
        let runnerUps = Dictionary(
            uniqueKeysWithValues: env.captionModel.runnerUps.map { ($0.shabadId, $0.confidence) }
        )
        recordIfOptedIn {
            CorrectionEventBuilder.makeRunnerUpEndorsed(
                sessionId: env.sessionId,
                predictedShabadId: predictedId,
                predictedConfidence: currentGuess?.confidence,
                runnerUps: runnerUps,
                endorsedShabadId: endorsedShabadId,
                engineStateRaw: CorrectionEventBuilder.engineStateRaw(env.captionModel.state)
            )
        }
    }

    private func recordRetroactiveFlag(entry: SessionEntry, correctedShabadId: Int) {
        guard entry.shabadId != correctedShabadId else { return }
        recordIfOptedIn {
            CorrectionEventBuilder.makeRetroactive(
                sessionId: env.sessionId,
                flaggedShabadId: entry.shabadId,
                correctedShabadId: correctedShabadId,
                flaggedAt: entry.timestamp,
                engineStateRaw: CorrectionEventBuilder.engineStateRaw(env.captionModel.state)
            )
        }
    }

    private var sevadarToolsSection: some View {
        VStack(alignment: .leading, spacing: tokens.spacing.md) {
            SectionLabel("Sevadar tools")

            VStack(spacing: 0) {
                SevadarToolRow(
                    icon: "waveform",
                    title: "Engine confidence",
                    subtitle: "ASR state, candidates, recent chunks"
                ) {
                    env.haptics.play(.selection)
                    showConfidence = true
                }
                .accessibilityIdentifier("settings.sevadar.confidence")

                Divider().background(tokens.colors.ruleSoft)

                SevadarToolRow(
                    icon: "clock.arrow.circlepath",
                    title: "Session history",
                    subtitle: "Today's shabads, in order"
                ) {
                    env.haptics.play(.selection)
                    showHistory = true
                }
                .accessibilityIdentifier("settings.sevadar.history")
            }
            .background(tokens.colors.surface)
            .clipShape(RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous)
                    .stroke(tokens.colors.rule, lineWidth: 0.5)
            )
        }
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

            Text("Affects the phone only — the projector always shows a single hero line for readability at distance.")
                .font(tokens.type.sansSmall)
                .foregroundStyle(tokens.colors.ink3)
                .fixedSize(horizontal: false, vertical: true)
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

            Text("Applies to both the phone and the projector — toggle these to control what the sangat sees on screen.")
                .font(tokens.type.sansSmall)
                .foregroundStyle(tokens.colors.ink3)
                .fixedSize(horizontal: false, vertical: true)
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

// MARK: - Sevadar tool row

private struct SevadarToolRow: View {

    @Environment(\.themeTokens) private var tokens

    let icon: String
    let title: String
    let subtitle: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: tokens.spacing.md) {
                Image(systemName: icon)
                    .font(.system(size: 18, weight: .regular))
                    .foregroundStyle(tokens.colors.sevadar)
                    .frame(width: 32, height: 32)
                    .background(tokens.colors.bgSoft, in: RoundedRectangle(cornerRadius: tokens.radii.sm, style: .continuous))

                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(tokens.type.serif.weight(.semibold))
                        .foregroundStyle(tokens.colors.ink)
                    Text(subtitle)
                        .font(tokens.type.sansSmall)
                        .foregroundStyle(tokens.colors.ink3)
                }

                Spacer()

                Image(systemName: "chevron.right")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(tokens.colors.ink3)
            }
            .padding(.horizontal, tokens.spacing.md)
            .padding(.vertical, tokens.spacing.sm + 2)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Previews

#Preview("SettingsView · paper sangat") {
    SettingsView()
        .environment(AppEnvironment.preview())
        .previewTheme(.paper)
}

#Preview("SettingsView · darbar sevadar") {
    SettingsView()
        .environment(AppEnvironment.preview(theme: .darbar, mode: .sevadar))
        .previewTheme(.darbar)
}

#Preview("SettingsView · mool sevadar") {
    SettingsView()
        .environment(AppEnvironment.preview(theme: .mool, mode: .sevadar))
        .previewTheme(.mool)
}

// MARK: - Identifiable shim for retroactive-flag follow-up sheet

/// Wraps `SessionEntry` so SwiftUI's `.sheet(item:)` can drive the
/// "supply the corrected shabad" picker. Fresh UUID per construction
/// so the sheet re-presents if the user flags two entries back-to-back.
private struct PendingRetroactiveFlag: Identifiable {
    let id: UUID
    let entry: SessionEntry

    init(entry: SessionEntry) {
        self.id = UUID()
        self.entry = entry
    }
}
