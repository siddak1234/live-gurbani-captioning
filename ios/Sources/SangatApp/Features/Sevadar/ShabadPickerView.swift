//
//  ShabadPickerView.swift
//  GurbaniCaptioningApp · SangatApp · Features/Sevadar
//
//  Modal manual shabad picker — design canvas section 06, V1ShabadPicker
//  (assets/v1-paper.jsx lines 836-882). Presented as a sheet from
//  RootView when the Sevadar dock's "Pick shabad" button is tapped.
//
//  Layout (top to bottom):
//  - OnboardingMetaHeader (sevadar mode strip)
//  - Title row: "Pick shabad" + accent-colored Cancel
//  - Search field with mono "X results" counter
//  - Sections: Now playing → Recent today → All matches
//  - Each row uses the existing `PickRow` atom

import SwiftUI

public struct ShabadPickerView: View {

    @Environment(AppEnvironment.self) private var env
    @Environment(\.themeTokens) private var tokens
    @Environment(\.dismiss) private var dismiss

    @State private var viewModel: ShabadPickerViewModel

    public let onPick: (Int) -> Void

    /// Header text rendered above the search bar. Defaults to the
    /// Sevadar "Pick shabad" copy. M5.6 reuses the same picker as the
    /// Sangat "Wrong shabad" sheet with different framing strings;
    /// keeping the override here means there's still one picker, one
    /// search engine, and one row atom.
    public let headerTitle: String
    public let headerSubtitle: String?

    public init(
        viewModel: ShabadPickerViewModel,
        onPick: @escaping (Int) -> Void,
        headerTitle: String = "Pick shabad",
        headerSubtitle: String? = nil
    ) {
        _viewModel = State(initialValue: viewModel)
        self.onPick = onPick
        self.headerTitle = headerTitle
        self.headerSubtitle = headerSubtitle
    }

    public var body: some View {
        @Bindable var vm = viewModel

        ZStack(alignment: .top) {
            tokens.colors.bg.ignoresSafeArea()

            VStack(alignment: .leading, spacing: 0) {
                OnboardingMetaHeader(mode: .sevadar)

                titleRow

                searchBar(query: $vm.searchQuery, results: vm.totalResultsCount)

                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        if let nowPlaying = vm.nowPlayingEntry {
                            section(title: "Now playing", entries: [nowPlaying], activeId: nowPlaying.shabadId, showLive: true)
                        }

                        if !vm.recentEntries.isEmpty {
                            section(title: "Recent today", entries: vm.recentEntries, activeId: nil, showLive: false)
                        }

                        section(title: "All matches", entries: vm.allMatches, activeId: nil, showLive: false)

                        if vm.totalResultsCount == 0 {
                            emptyState
                        }
                    }
                    .padding(.bottom, tokens.spacing.xl)
                }
            }
        }
        .accessibilityIdentifier("shabadPicker.root")
    }

    private var titleRow: some View {
        VStack(alignment: .leading, spacing: tokens.spacing.xs) {
            HStack(alignment: .lastTextBaseline) {
                Text(headerTitle)
                    .font(tokens.type.serifTitle)
                    .foregroundStyle(tokens.colors.ink)
                    .accessibilityAddTraits(.isHeader)

                Spacer()

                Button {
                    env.haptics.play(.selection)
                    dismiss()
                } label: {
                    Text("Cancel")
                        .font(tokens.type.sans.weight(.semibold))
                        .foregroundStyle(tokens.colors.accent)
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("shabadPicker.cancel")
            }

            if let headerSubtitle, !headerSubtitle.isEmpty {
                Text(headerSubtitle)
                    .font(tokens.type.sansSmall)
                    .foregroundStyle(tokens.colors.ink3)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.horizontal, tokens.spacing.edge)
        .padding(.top, tokens.spacing.lg)
        .padding(.bottom, tokens.spacing.sm)
    }

    private func searchBar(query: Binding<String>, results: Int) -> some View {
        HStack(spacing: tokens.spacing.sm) {
            Circle()
                .fill(tokens.colors.ink3)
                .frame(width: 6, height: 6)

            TextField("", text: query, prompt: Text("Search Gurmukhi…")
                .foregroundStyle(tokens.colors.ink3))
                .font(tokens.type.gurmukhi)
                .foregroundStyle(tokens.colors.ink)
                .autocorrectionDisabled(true)
                .accessibilityIdentifier("shabadPicker.search")

            Spacer(minLength: tokens.spacing.sm)

            Text("\(results) results")
                .font(tokens.type.mono)
                .foregroundStyle(tokens.colors.ink3)
        }
        .padding(.horizontal, tokens.spacing.md)
        .padding(.vertical, tokens.spacing.sm + 2)
        .background(tokens.colors.surface, in: RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous)
                .stroke(tokens.colors.rule, lineWidth: 0.5)
        )
        .padding(.horizontal, tokens.spacing.edge)
        .padding(.bottom, tokens.spacing.md)
    }

    @ViewBuilder
    private func section(
        title: String,
        entries: [ShabadPickerEntry],
        activeId: Int?,
        showLive: Bool
    ) -> some View {
        if !entries.isEmpty {
            SectionLabel(title)
                .padding(.horizontal, tokens.spacing.edge)
                .padding(.top, tokens.spacing.md)
                .padding(.bottom, tokens.spacing.xs)

            ForEach(entries) { entry in
                PickRow(
                    titleGurmukhi: entry.firstLineGurmukhi,
                    meta: entry.meta,
                    isLive: showLive && entry.shabadId == activeId,
                    isActive: entry.shabadId == activeId,
                    action: {
                        env.haptics.play(.success)
                        onPick(entry.shabadId)
                        dismiss()
                    }
                )
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: tokens.spacing.sm) {
            Text("No matches")
                .font(tokens.type.serif.weight(.semibold))
                .foregroundStyle(tokens.colors.ink)
            Text("Try a different Gurmukhi prefix.")
                .font(tokens.type.sansSmall)
                .foregroundStyle(tokens.colors.ink3)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, tokens.spacing.xxl)
        .accessibilityElement(children: .combine)
    }
}

#Preview("ShabadPickerView · paper") {
    ShabadPickerView(
        viewModel: ShabadPickerViewModel(
            nowPlayingShabadId: 1789,
            recents: Array<ShabadPickerEntry>.demoRecents
        ),
        onPick: { _ in }
    )
    .environment(AppEnvironment.preview())
    .previewTheme(.paper)
}

#Preview("ShabadPickerView · darbar") {
    ShabadPickerView(
        viewModel: ShabadPickerViewModel(
            nowPlayingShabadId: 1789,
            recents: Array<ShabadPickerEntry>.demoRecents
        ),
        onPick: { _ in }
    )
    .environment(AppEnvironment.preview(theme: .darbar, mode: .sevadar))
    .previewTheme(.darbar)
}

#Preview("ShabadPickerView · mool, no now-playing") {
    ShabadPickerView(
        viewModel: ShabadPickerViewModel(recents: Array<ShabadPickerEntry>.demoRecents),
        onPick: { _ in }
    )
    .environment(AppEnvironment.preview(theme: .mool, mode: .sevadar))
    .previewTheme(.mool)
}
