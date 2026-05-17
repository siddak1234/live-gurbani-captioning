//
//  HapticsService.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Thin protocol over `UIImpactFeedbackGenerator` so views can fire haptic
//  events through dependency injection. Lets unit tests swap in a no-op
//  service and avoid the UIKit dependency.

import Foundation

#if canImport(UIKit)
import UIKit
#endif

// MARK: - Protocol

/// Hooks for in-app haptic feedback. Keep cases small and semantic — UI
/// authors should pick *what they're confirming*, not *what kind of buzz*.
public protocol HapticsService: AnyObject, Sendable {
    /// Fire haptic appropriate to the given event.
    func play(_ event: HapticEvent)
}

/// What semantic event the haptic accompanies.
public enum HapticEvent: Sendable {
    /// Light tap — buttons, toggles.
    case selection
    /// Confirmation — committed shabad, action completed.
    case success
    /// Soft warning — invalid input, edge case.
    case warning
    /// Hard error — operation failed.
    case error
    /// Heavy thunk — major mode change (entering Sevadar).
    case impactHeavy
    /// Medium thunk — secondary mode change.
    case impactMedium
    /// Light tick — state transition.
    case impactLight
}

// MARK: - UIKit-backed implementation

#if canImport(UIKit)

@MainActor
public final class UIKitHapticsService: HapticsService {

    private let selectionGen = UISelectionFeedbackGenerator()
    private let notificationGen = UINotificationFeedbackGenerator()
    private let impactLight = UIImpactFeedbackGenerator(style: .light)
    private let impactMedium = UIImpactFeedbackGenerator(style: .medium)
    private let impactHeavy = UIImpactFeedbackGenerator(style: .heavy)

    public init() {}

    public nonisolated func play(_ event: HapticEvent) {
        // Trampoline back to main actor — `UIImpactFeedbackGenerator` is
        // main-actor-bound. Doing this internally keeps callers from having
        // to remember.
        Task { @MainActor in
            switch event {
            case .selection:
                self.selectionGen.selectionChanged()
            case .success:
                self.notificationGen.notificationOccurred(.success)
            case .warning:
                self.notificationGen.notificationOccurred(.warning)
            case .error:
                self.notificationGen.notificationOccurred(.error)
            case .impactLight:
                self.impactLight.impactOccurred()
            case .impactMedium:
                self.impactMedium.impactOccurred()
            case .impactHeavy:
                self.impactHeavy.impactOccurred()
            }
        }
    }
}

#endif

// MARK: - No-op implementation (tests, previews)

public final class NoopHapticsService: HapticsService {
    public init() {}
    public func play(_ event: HapticEvent) {}
}
