//
//  ConfidenceView.swift
//  GurbaniCaptioningApp · SangatApp · Features/Sevadar
//
//  Sevadar engine-confidence / debug panel — design canvas section 06
//  V1Confidence (assets/v1-paper.jsx lines 950-1024). Reads
//  CaptionSource's published state (`state`, `currentGuess`, `runnerUps`)
//  so the panel reflects the live engine, not a hardcoded list.
//
//  Sections (top to bottom):
//    - Meta strip + "Engine" title with backend / chunk-size subtitle
//    - 2×2 stat grid: State · Confidence · Margin · RTF (StatCard atoms)
//    - Top candidates list, current commit highlighted in accentSoft
//    - Recent ASR chunks list (placeholder — wired to real chunks at M5.7)
//
//  Reached from Settings → Sevadar tools when env.mode == .sevadar.

import SwiftUI
import GurbaniCaptioning

public struct ConfidenceView: View {

    @Environment(AppEnvironment.self) private var env
    @Environment(\.themeTokens) private var tokens
    @Environment(\.dismiss) private var dismiss

    /// M5.6 hook. Fires when the Sevadar taps a non-committed
    /// candidate row, signaling "this one is right" before the engine
    /// commits. Owner (Settings → Confidence sheet host in RootView)
    /// is responsible for `manuallyCommit` + `correctionLog.record`.
    public let onEndorseRunnerUp: ((Int) -> Void)?

    public init(onEndorseRunnerUp: ((Int) -> Void)? = nil) {
        self.onEndorseRunnerUp = onEndorseRunnerUp
    }

    public var body: some View {
        ZStack(alignment: .top) {
            tokens.colors.bg.ignoresSafeArea()

            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    OnboardingMetaHeader(mode: .sevadar)

                    titleBlock

                    SectionLabel("State")
                        .padding(.horizontal, tokens.spacing.edge)
                        .padding(.top, tokens.spacing.md)
                        .padding(.bottom, tokens.spacing.xs)

                    statGrid
                        .padding(.horizontal, tokens.spacing.edge)

                    SectionLabel("Top candidates")
                        .padding(.horizontal, tokens.spacing.edge)
                        .padding(.top, tokens.spacing.lg)
                        .padding(.bottom, tokens.spacing.xs)

                    candidatesList
                        .padding(.horizontal, tokens.spacing.edge)

                    SectionLabel("Recent ASR chunks")
                        .padding(.horizontal, tokens.spacing.edge)
                        .padding(.top, tokens.spacing.lg)
                        .padding(.bottom, tokens.spacing.xs)

                    chunksList
                        .padding(.horizontal, tokens.spacing.edge)
                        .padding(.bottom, tokens.spacing.xl)
                }
            }
        }
        .accessibilityIdentifier("confidence.root")
    }

    private var titleBlock: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Engine")
                    .font(tokens.type.serifTitle)
                    .foregroundStyle(tokens.colors.ink)
                    .accessibilityAddTraits(.isHeader)
                Text("surt-small-v3 · ANE · 5s chunks")
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
            .accessibilityIdentifier("confidence.close")
        }
        .padding(.horizontal, tokens.spacing.edge)
        .padding(.top, tokens.spacing.lg)
    }

    private var statGrid: some View {
        LazyVGrid(
            columns: [
                GridItem(.flexible(), spacing: tokens.spacing.sm),
                GridItem(.flexible(), spacing: tokens.spacing.sm)
            ],
            spacing: tokens.spacing.sm
        ) {
            StatCard(
                label: "State",
                value: stateLabel,
                valueColor: stateColor,
                useMono: false
            )
            StatCard(label: "Confidence", value: confidenceText)
            StatCard(label: "Margin", value: marginText)
            StatCard(label: "RTF", value: "0.07")
        }
    }

    @ViewBuilder
    private var candidatesList: some View {
        let committed = env.captionModel.currentGuess
        let runners = env.captionModel.runnerUps

        if committed == nil && runners.isEmpty {
            Text("No candidates yet.")
                .font(tokens.type.sansSmall)
                .foregroundStyle(tokens.colors.ink3)
                .padding(.vertical, tokens.spacing.sm)
        } else {
            VStack(spacing: tokens.spacing.xs + 2) {
                if let committed {
                    // The committed row is never tappable — tapping it
                    // would be a self-endorsement, which the engine
                    // already has. `onSelect = nil` disables the
                    // gesture and the visual affordance.
                    CandidateRow(
                        shabadId: committed.shabadId,
                        title: candidateTitle(forShabadId: committed.shabadId),
                        score: committed.confidence,
                        isCommitted: true,
                        onSelect: nil
                    )
                }
                ForEach(runners.prefix(4), id: \.shabadId) { runner in
                    CandidateRow(
                        shabadId: runner.shabadId,
                        title: candidateTitle(forShabadId: runner.shabadId),
                        score: runner.confidence,
                        isCommitted: false,
                        onSelect: { onEndorseRunnerUp?(runner.shabadId) }
                    )
                }
            }
        }
    }

    @ViewBuilder
    private var chunksList: some View {
        // Placeholder: real chunk history requires CaptionEngine to publish
        // recent AsrChunks. M5.7 wires that. For now, render a small
        // disclaimer so the panel is honest about what's live.
        Text("Recent transcript chunks will appear here once the live engine is wired (M5.7).")
            .font(tokens.type.sansSmall)
            .foregroundStyle(tokens.colors.ink3)
            .fixedSize(horizontal: false, vertical: true)
    }

    // MARK: - Derived display values

    private var stateLabel: String {
        switch env.captionModel.state {
        case .listening: return "Listening"
        case .tentative: return "Tentative"
        case .committed: return "Committed"
        }
    }

    private var stateColor: Color {
        switch env.captionModel.state {
        case .committed: return tokens.colors.accent
        case .tentative, .listening: return tokens.colors.amber
        }
    }

    private var confidenceText: String {
        guard let g = env.captionModel.currentGuess else { return "—" }
        return String(format: "%.1f", g.confidence)
    }

    private var marginText: String {
        guard
            let committed = env.captionModel.currentGuess,
            let runnerUp = env.captionModel.runnerUps.first
        else { return "—" }
        let margin = committed.confidence - runnerUp.confidence
        return String(format: "%+.1f", margin)
    }

    private func candidateTitle(forShabadId shabadId: Int) -> String {
        // Until the corpus is bundled, fall back to a Shabad #N label
        // with a known title for the demo shabad. M5.7 reads from
        // ShabadCorpus.shabad(_:).
        switch shabadId {
        case 1789: return "Tati Vao Na Lagai"
        case 8: return "So Dar Keha"
        case 3: return "Jo Tudh Bhavai"
        case 1: return "Ek Onkar"
        default: return "Shabad #\(shabadId)"
        }
    }
}

// MARK: - CandidateRow (inline — single use site)

private struct CandidateRow: View {

    @Environment(\.themeTokens) private var tokens

    let shabadId: Int
    let title: String
    let score: Double
    let isCommitted: Bool

    /// Non-nil → row is tappable and signals endorsement. Nil → row
    /// is display-only (committed rows, or when the host doesn't wire
    /// the endorsement path).
    let onSelect: (() -> Void)?

    var body: some View {
        let rowContent = HStack(spacing: tokens.spacing.sm) {
            Text("#\(shabadId)")
                .font(tokens.type.mono)
                .foregroundStyle(tokens.colors.ink3)

            Text(title)
                .font(tokens.type.serif)
                .foregroundStyle(tokens.colors.ink)
                .lineLimit(1)

            Spacer(minLength: tokens.spacing.sm)

            Text(String(format: "%.1f", score))
                .font(tokens.type.mono.weight(.semibold))
                .monospacedDigit()
                .foregroundStyle(tokens.colors.ink)
        }
        .padding(.horizontal, tokens.spacing.sm + 2)
        .padding(.vertical, tokens.spacing.sm)
        .background(
            isCommitted ? tokens.colors.accentSoft : Color.clear,
            in: RoundedRectangle(cornerRadius: tokens.radii.sm, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: tokens.radii.sm, style: .continuous)
                .stroke(isCommitted ? Color.clear : tokens.colors.ruleSoft, lineWidth: 1)
        )
        .contentShape(Rectangle())

        if let onSelect {
            Button(action: onSelect) {
                rowContent
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("confidence.candidate.\(shabadId)")
        } else {
            rowContent
                .accessibilityIdentifier("confidence.candidate.\(shabadId)")
        }
    }
}

#Preview("ConfidenceView · paper") {
    ConfidenceView()
        .environment(AppEnvironment.preview(mode: .sevadar))
        .previewTheme(.paper)
}

#Preview("ConfidenceView · darbar") {
    ConfidenceView()
        .environment(AppEnvironment.preview(theme: .darbar, mode: .sevadar))
        .previewTheme(.darbar)
}
