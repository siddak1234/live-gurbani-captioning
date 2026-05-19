//
//  HistoryView.swift
//  GurbaniCaptioningApp · SangatApp · Features/Sevadar
//
//  Today's session history — design canvas section 06, V1History
//  (assets/v1-paper.jsx lines 911-948). Reads from
//  `env.sessionHistory.recent(limit:)` so it stays decoupled from any
//  specific store implementation. Until the live engine starts
//  appending entries (M5.7), we seed a small demo list in
//  `seedDemoHistoryIfEmpty()` so the screen has data to show.
//
//  Each row: time (mono, right-aligned), Gurmukhi first line,
//  "Ang X · MM:SS" meta, accent dot if isLive.

import SwiftUI
import GurbaniCaptioning

public struct HistoryView: View {

    @Environment(AppEnvironment.self) private var env
    @Environment(\.themeTokens) private var tokens
    @Environment(\.dismiss) private var dismiss

    @State private var entries: [SessionEntry] = []

    /// M5.6 hook. Fires when the Sevadar long-presses a past history
    /// row, signaling "this entry was the wrong shabad". Host
    /// (Settings → History sheet in RootView) presents a follow-up
    /// picker so the user can supply the corrected shabad id.
    public let onFlagWrongShabad: ((SessionEntry) -> Void)?

    public init(onFlagWrongShabad: ((SessionEntry) -> Void)? = nil) {
        self.onFlagWrongShabad = onFlagWrongShabad
    }

    public var body: some View {
        ZStack(alignment: .top) {
            tokens.colors.bg.ignoresSafeArea()

            VStack(alignment: .leading, spacing: 0) {
                OnboardingMetaHeader(mode: .sevadar)

                titleBlock

                if entries.isEmpty {
                    emptyState
                        .padding(.top, tokens.spacing.xxl)
                } else {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 0) {
                            ForEach(entries) { entry in
                                HistoryRow(
                                    entry: entry,
                                    activeShabadId: env.captionModel.committedShabadId,
                                    onFlag: onFlagWrongShabad.map { handler in
                                        {
                                            env.haptics.play(.warning)
                                            handler(entry)
                                        }
                                    }
                                )
                                Divider().background(tokens.colors.ruleSoft)
                            }
                        }
                        .padding(.bottom, tokens.spacing.xl)
                    }
                }
            }
        }
        .onAppear {
            seedDemoIfEmpty()
            entries = env.sessionHistory.recent(limit: 50)
        }
        .accessibilityIdentifier("history.root")
    }

    private var titleBlock: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Today")
                    .font(tokens.type.serifTitle)
                    .foregroundStyle(tokens.colors.ink)
                    .accessibilityAddTraits(.isHeader)
                Text(summaryLine)
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
            .accessibilityIdentifier("history.close")
        }
        .padding(.horizontal, tokens.spacing.edge)
        .padding(.top, tokens.spacing.lg)
        .padding(.bottom, tokens.spacing.md)
    }

    private var emptyState: some View {
        VStack(spacing: tokens.spacing.sm) {
            Text("No shabads yet today")
                .font(tokens.type.serif.weight(.semibold))
                .foregroundStyle(tokens.colors.ink)
            Text("Once a shabad commits, it'll appear here.")
                .font(tokens.type.sansSmall)
                .foregroundStyle(tokens.colors.ink3)
        }
        .frame(maxWidth: .infinity)
    }

    private var summaryLine: String {
        if entries.isEmpty {
            return "No sessions yet"
        }
        let totalSeconds = Int(entries.reduce(0) { $0 + $1.durationSeconds })
        let mins = totalSeconds / 60
        let secs = totalSeconds % 60
        let durText = String(format: "%d:%02d", mins, secs)
        return "\(entries.count) shabads · \(durText) of kirtan"
    }

    /// Drop a small demo list into the session history the first time
    /// HistoryView appears with an empty store. Keeps the screen
    /// honest-looking until M5.7's live engine appends real entries.
    private func seedDemoIfEmpty() {
        guard env.sessionHistory.recent(limit: 1).isEmpty else { return }

        let now = Date()
        let demoSeeds: [(offset: TimeInterval, shabadId: Int, firstLine: String, ang: Int, dur: TimeInterval, ended: SessionEntry.EndedReason)] = [
            (-60 * 18, 1789, "ਤਾਤੀ ਵਾਉ ਨ ਲਗਈ", 819, 254, .engineCommittedNew),
            (-60 * 12, 8, "ਸੋ ਦਰੁ ਕੇਹਾ ਸੋ ਘਰੁ ਕੇਹਾ", 8, 311, .userManualPicked),
            (-60 * 6, 3, "ਜੋ ਤੁਧੁ ਭਾਵੈ ਸਾਈ ਭਲੀ ਕਾਰ", 3, 198, .userReset),
            (-60 * 2, 1789, "ਤਾਤੀ ਵਾਉ ਨ ਲਗਈ", 819, 230, .engineCommittedNew)
        ]
        for seed in demoSeeds {
            env.sessionHistory.append(SessionEntry(
                timestamp: now.addingTimeInterval(seed.offset),
                shabadId: seed.shabadId,
                firstLineGurmukhi: seed.firstLine,
                ang: seed.ang,
                durationSeconds: seed.dur,
                endedReason: seed.ended
            ))
        }
    }
}

// MARK: - HistoryRow (inline — single use site)

private struct HistoryRow: View {

    @Environment(\.themeTokens) private var tokens

    let entry: SessionEntry
    let activeShabadId: Int?

    /// Non-nil → row exposes a "Flag as wrong" context-menu item.
    /// Nil → no menu, row is read-only (the natural state outside
    /// the Sevadar correction flow).
    let onFlag: (() -> Void)?

    var body: some View {
        HStack(alignment: .center, spacing: tokens.spacing.md) {
            Text(timeText)
                .font(tokens.type.mono)
                .foregroundStyle(tokens.colors.ink3)
                .frame(width: 44, alignment: .trailing)

            VStack(alignment: .leading, spacing: 2) {
                Text(entry.firstLineGurmukhi)
                    .font(tokens.type.gurmukhi)
                    .foregroundStyle(tokens.colors.ink)
                    .lineLimit(1)

                Text("Ang \(entry.ang) · \(durationText)")
                    .font(tokens.type.sansSmall)
                    .foregroundStyle(tokens.colors.ink3)
            }

            Spacer(minLength: tokens.spacing.sm)

            if entry.shabadId == activeShabadId {
                Circle()
                    .fill(tokens.colors.accent)
                    .frame(width: 8, height: 8)
                    .accessibilityHidden(true)
            }
        }
        .padding(.horizontal, tokens.spacing.edge)
        .padding(.vertical, tokens.spacing.sm + 2)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(entry.firstLineGurmukhi). Ang \(entry.ang). \(durationText).")
        .contentShape(Rectangle())
        .contextMenu {
            if let onFlag {
                Button(role: .destructive) {
                    onFlag()
                } label: {
                    Label("Flag as wrong shabad", systemImage: "flag.fill")
                }
                .accessibilityIdentifier("history.row.flag")
            }
        }
    }

    private var timeText: String {
        let f = DateFormatter()
        f.dateFormat = "HH:mm"
        return f.string(from: entry.timestamp)
    }

    private var durationText: String {
        let total = Int(entry.durationSeconds)
        let mins = total / 60
        let secs = total % 60
        return String(format: "%d:%02d", mins, secs)
    }
}

#Preview("HistoryView · paper") {
    HistoryView()
        .environment(AppEnvironment.preview(mode: .sevadar))
        .previewTheme(.paper)
}

#Preview("HistoryView · darbar") {
    HistoryView()
        .environment(AppEnvironment.preview(theme: .darbar, mode: .sevadar))
        .previewTheme(.darbar)
}
