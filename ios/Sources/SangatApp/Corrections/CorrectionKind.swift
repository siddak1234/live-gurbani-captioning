//
//  CorrectionKind.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Enumerates the signals the correction loop can capture. Each kind has
//  different training value (see `docs/ios_app_architecture.md` §
//  "Correction loop signal taxonomy"); we surface the kind on every event
//  so the eventual fine-tune pipeline can weight or filter them.
//
//  Cases mirror the plan from the design phase 1:1.

import Foundation

/// Type of user correction recorded.
public enum CorrectionKind: String, Codable, Sendable, CaseIterable {

    /// Engine committed shabad A, user explicitly chose shabad B.
    /// Highest-value signal: yields a labeled hard-negative + hard-positive
    /// pair pointing at the same audio chunk.
    case hardNegPos

    /// Engine committed shabad A, user read for ≥60s without correcting.
    /// Soft positive — useful confidence weighting, low certainty.
    case softPos

    /// User picked from the displayed runner-up candidates *before* the
    /// engine committed. Pre-emptive correction.
    case runnerUpEndorsed

    /// Within a committed shabad, user nudged the displayed line forward
    /// or backward by one. Feeds the smoother / loop-aligner.
    case lineNudge

    /// User flagged a past session entry as wrong via the History view.
    /// Lowest-confidence — they didn't catch it live.
    case retroactive
}
