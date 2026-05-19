//
//  WrongShabadSheet.swift
//  GurbaniCaptioningApp · SangatApp · Features/Corrections
//
//  Created for the Sangat iOS app, M5.6 (Correction loop surfaces).
//
//  The Sangat-side mistake-recovery surface. Reached by long-pressing
//  the active reading view (Hero / Karaoke / Full) when the engine has
//  committed to a shabad the user believes is wrong. Re-uses
//  `ShabadPickerView` with reframed header strings so there is exactly
//  one search/picker engine across the app.
//
//  Action on pick:
//  1. Commit the corrected shabad via `captionModel.manuallyCommit(...)`
//     so the on-screen content updates immediately.
//  2. Record a `hardNegPos` `CorrectionEvent` (highest-value training
//     signal) — gated on `preferences.correctionsOptIn`.
//
//  Step 2 lives in `RootView` so this view stays presentation-only;
//  RootView owns the env-wide effects.

import SwiftUI

public struct WrongShabadSheet: View {

    @Environment(AppEnvironment.self) private var env

    /// The shabad currently committed by the engine — used to seed the
    /// picker's "Now playing" section so the user sees what they're
    /// correcting against.
    public let predictedShabadId: Int

    /// Called when the user picks a corrected shabad. Receives the
    /// chosen shabad id; the caller (RootView) is responsible for the
    /// commit + event-record side effects.
    public let onCorrect: (Int) -> Void

    public init(predictedShabadId: Int, onCorrect: @escaping (Int) -> Void) {
        self.predictedShabadId = predictedShabadId
        self.onCorrect = onCorrect
    }

    public var body: some View {
        ShabadPickerView(
            viewModel: ShabadPickerViewModel(
                nowPlayingShabadId: predictedShabadId,
                recents: []
            ),
            onPick: onCorrect,
            headerTitle: "Wrong shabad?",
            headerSubtitle: "Pick the correct one. The engine learns from this if you've opted in."
        )
        .accessibilityIdentifier("wrongShabad.root")
    }
}

#Preview("WrongShabadSheet · paper") {
    WrongShabadSheet(predictedShabadId: 1789, onCorrect: { _ in })
        .environment(AppEnvironment.preview())
        .previewTheme(.paper)
}

#Preview("WrongShabadSheet · darbar") {
    WrongShabadSheet(predictedShabadId: 1789, onCorrect: { _ in })
        .environment(AppEnvironment.preview(theme: .darbar))
        .previewTheme(.darbar)
}
