//
//  OnboardingViewModel.swift
//  GurbaniCaptioningApp · SangatApp · Features/Onboarding
//
//  State machine for the four-card onboarding flow defined in
//  `assets/v1-paper.jsx` (V1Onb01_Welcome → V1Onb04_RolePick).
//  Held by `OnboardingFlow` and shared with each card view; mic
//  permission is the only step that performs real work.

import Foundation
import Observation

/// The four cards in the onboarding sequence, in display order.
public enum OnboardingStep: String, CaseIterable, Identifiable, Sendable {
    case welcome
    case howItWorks
    case mic
    case role

    public var id: String { rawValue }

    public var index: Int {
        // CaseIterable order is the source of truth — keep it equal to
        // the visual order so navigation maps cleanly.
        Self.allCases.firstIndex(of: self) ?? 0
    }
}

/// Pure state machine — does not touch `AppEnvironment`. The parent
/// `OnboardingFlow` reads the final selection (`selectedRole`,
/// `micPermissionStatus`) and commits to the env when the flow completes.
@MainActor
@Observable
public final class OnboardingViewModel {

    // MARK: - State

    public private(set) var step: OnboardingStep = .welcome
    public var selectedRole: AppMode = .sangat
    public private(set) var micPermissionStatus: MicPermissionStatus = .notDetermined
    public private(set) var micRequestInFlight: Bool = false

    public init(initial: OnboardingStep = .welcome) {
        self.step = initial
    }

    // MARK: - Navigation

    /// Move forward one step. No-op at `.role`; completion is handled by
    /// `OnboardingFlow` which observes the step + writes to env.
    public func advance() {
        guard let next = nextStep(after: step) else { return }
        step = next
    }

    /// Move back one step. No-op at `.welcome` — the Welcome card has
    /// no Back button per the design.
    public func back() {
        guard let prev = previousStep(before: step) else { return }
        step = prev
    }

    // MARK: - Mic permission

    /// Prompt the user for microphone access via `AudioPermissions`.
    /// After the user answers the system dialog this returns and we
    /// advance to the role-pick step regardless of the answer
    /// (denial is acceptable; the user just won't get live captions).
    @discardableResult
    public func requestMicPermission() async -> MicPermissionStatus {
        guard step == .mic else { return micPermissionStatus }

        micRequestInFlight = true
        defer { micRequestInFlight = false }

        let status = await AudioPermissions.request()
        micPermissionStatus = status
        advance()
        return status
    }

    /// "Not now" path — user declines to give a definitive answer.
    /// Status stays `.notDetermined` so `OnboardingFlow` records
    /// `micPermissionAcknowledged = false`, and IdleView can surface an
    /// "Enable microphone" affordance later (deferred to M5.4.1 / M5.3).
    public func skipMic() {
        guard step == .mic else { return }
        micPermissionStatus = .notDetermined
        advance()
    }

    /// True once the user reached the role-pick step. Drives the
    /// "Continue as Sangat" / "Continue as Sevadar" CTA visibility in
    /// `OnboardingFlow`.
    public var isAtFinalStep: Bool {
        step == .role
    }

    /// True once the user explicitly answered the system mic dialog
    /// (granted OR denied via the system alert). `Not now` keeps this
    /// at false so `micPermissionAcknowledged` records the truth.
    public var micPermissionAcknowledged: Bool {
        switch micPermissionStatus {
        case .granted, .denied, .restricted: return true
        case .notDetermined, .unavailable:   return false
        }
    }

    // MARK: - Step neighbors (private)

    private func nextStep(after current: OnboardingStep) -> OnboardingStep? {
        let cases = OnboardingStep.allCases
        guard
            let idx = cases.firstIndex(of: current),
            idx + 1 < cases.count
        else { return nil }
        return cases[idx + 1]
    }

    private func previousStep(before current: OnboardingStep) -> OnboardingStep? {
        let cases = OnboardingStep.allCases
        guard
            let idx = cases.firstIndex(of: current),
            idx > 0
        else { return nil }
        return cases[idx - 1]
    }
}
