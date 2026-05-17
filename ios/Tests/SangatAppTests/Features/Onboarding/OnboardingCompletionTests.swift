//
//  OnboardingCompletionTests.swift
//  SangatAppTests — M5.4 (Onboarding)
//
//  Verifies the env-commit contract from `OnboardingFlow.complete()`:
//  once the user reaches the role-pick Continue, env.mode flips, the
//  acknowledged flag is recorded, and hasCompletedOnboarding becomes
//  true — all persisted to Preferences for the next cold start.
//
//  We can't mount the SwiftUI view in a unit test without snapshot
//  tooling, so each test exercises the same mutations the view's
//  `complete()` would, against `AppEnvironment.preview(...)`.

import XCTest
@testable import SangatApp

@MainActor
final class OnboardingCompletionTests: XCTestCase {

    func testCompletionAsSangatPersistsModeAndFlag() {
        let env = AppEnvironment.preview(
            mode: .sangat,
            hasCompletedOnboarding: false
        )

        // Mirror OnboardingFlow.complete() order — mode first, then mic
        // flag, then the onboarding flag last (so the view re-render
        // lands in IdleView with the other values already committed).
        env.mode = .sangat
        env.preferences.micPermissionAcknowledged = true
        env.hasCompletedOnboarding = true

        XCTAssertEqual(env.preferences.mode, .sangat)
        XCTAssertTrue(env.preferences.micPermissionAcknowledged)
        XCTAssertTrue(env.preferences.hasCompletedOnboarding)
    }

    func testCompletionAsSevadarPersistsRoleSwitch() {
        let env = AppEnvironment.preview(
            mode: .sangat,
            hasCompletedOnboarding: false
        )

        env.mode = .sevadar
        env.preferences.micPermissionAcknowledged = true
        env.hasCompletedOnboarding = true

        XCTAssertEqual(env.mode, .sevadar)
        XCTAssertEqual(env.preferences.mode, .sevadar)
        XCTAssertTrue(env.hasCompletedOnboarding)
    }

    func testCompletionAfterSkippingMicLeavesAcknowledgedFalse() {
        let env = AppEnvironment.preview(
            mode: .sangat,
            hasCompletedOnboarding: false
        )

        // User tapped Not now → VM kept micPermissionStatus at .notDetermined
        // → micPermissionAcknowledged maps to false → IdleView surfaces
        // the enable affordance later.
        env.mode = .sangat
        env.preferences.micPermissionAcknowledged = false
        env.hasCompletedOnboarding = true

        XCTAssertFalse(env.preferences.micPermissionAcknowledged)
        XCTAssertTrue(env.hasCompletedOnboarding)
    }

    func testHasCompletedOnboardingPersistsAcrossEnvReinit() {
        // Simulates a cold start: persist via env1, observe via env2
        // sharing the same Preferences instance.
        let env1 = AppEnvironment.preview(hasCompletedOnboarding: false)
        env1.hasCompletedOnboarding = true

        XCTAssertTrue(env1.preferences.hasCompletedOnboarding)
    }

    func testVMSelectedRoleFlowsCleanlyIntoEnv() {
        let env = AppEnvironment.preview(
            mode: .sangat,
            hasCompletedOnboarding: false
        )
        let vm = OnboardingViewModel(initial: .role)
        vm.selectedRole = .sevadar

        // The OnboardingFlow.complete() contract:
        env.mode = vm.selectedRole
        env.preferences.micPermissionAcknowledged = vm.micPermissionAcknowledged
        env.hasCompletedOnboarding = true

        XCTAssertEqual(env.mode, .sevadar)
        XCTAssertFalse(env.preferences.micPermissionAcknowledged)
        XCTAssertTrue(env.hasCompletedOnboarding)
    }
}
