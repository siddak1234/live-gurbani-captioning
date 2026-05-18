//
//  CastSceneCoordinatorTests.swift
//  SangatAppTests — M5.5 (Cast)
//
//  External-screen attach/detach can't be unit-tested without
//  simulator hardware — those gates are in the M5.5 simulator
//  walkthrough (5.1-5.6). These tests cover the parts that work
//  on the macOS test host:
//    - Init doesn't crash when no external screen present
//    - Observable state starts in the right shape
//    - Test reset path works for tearing down inside tests

import XCTest
@testable @_spi(Testing) import SangatApp

@MainActor
final class CastSceneCoordinatorTests: XCTestCase {

    func testInitDoesNotCrashWithoutExternalScreen() {
        let env = AppEnvironment.preview()
        let coordinator = CastSceneCoordinator(env: env)
        // If init crashed, we wouldn't reach this line.
        XCTAssertNotNil(coordinator)
    }

    func testInitialStateIsNotConnected() {
        let env = AppEnvironment.preview()
        let coordinator = CastSceneCoordinator(env: env)
        XCTAssertFalse(coordinator.isExternalScreenConnected)
        XCTAssertNil(coordinator.externalScreenName)
    }

    func testTestResetClearsState() {
        let env = AppEnvironment.preview()
        let coordinator = CastSceneCoordinator(env: env)
        coordinator._testReset()
        XCTAssertFalse(coordinator.isExternalScreenConnected)
        XCTAssertNil(coordinator.externalScreenName)
    }

    func testCoordinatorHoldsWeakEnvReference() {
        // The coordinator must NOT retain AppEnvironment — RootView
        // owns it via @State, and a retain cycle here would leak the
        // entire env (which retains the caption source + history
        // store) until the app process ends.
        weak var weakEnv: AppEnvironment?
        var coordinator: CastSceneCoordinator?
        autoreleasepool {
            let env = AppEnvironment.preview()
            weakEnv = env
            coordinator = CastSceneCoordinator(env: env)
        }
        // After the autoreleasepool drains, the strong `env` is gone.
        // If the coordinator held a strong reference, weakEnv would
        // still be non-nil.
        _ = coordinator  // keep the coordinator alive so this is a real test
        XCTAssertNil(weakEnv, "CastSceneCoordinator must not retain AppEnvironment")
    }

    // MARK: - Role-policy (carried-forward M5.5 finding, fixed in M5.4.1)

    func testCastSurfaceIsAllowedForSevadar() {
        // The projector view exists to support a Sevadar operating
        // for a sangat audience. The role-policy helper is the gate
        // both `attach(to:)` and `modeDidChange(to:)` consult.
        XCTAssertTrue(CastSceneCoordinator.shouldHostCastSurface(for: .sevadar))
    }

    func testCastSurfaceIsBlockedForSangat() {
        // A Sangat-mode user connecting an external display must NOT
        // auto-mount the projector window — confirmed by the design
        // canvas (RolePickCard/RolePickerSheet describe casting as a
        // Sevadar capability) and by the fact that SevadarDock is the
        // only in-app surface with an `onCast` affordance.
        XCTAssertFalse(CastSceneCoordinator.shouldHostCastSurface(for: .sangat))
    }

    func testEveryAppModeHasAnExplicitCastPolicy() {
        // Exhaustive sweep so future cases on AppMode can't silently
        // default to "allowed" or "blocked" without a deliberate call.
        for mode in AppMode.allCases {
            // Compiles iff the helper covers `mode`; the assertion
            // body documents the contract.
            let allowed = CastSceneCoordinator.shouldHostCastSurface(for: mode)
            switch mode {
            case .sevadar: XCTAssertTrue(allowed)
            case .sangat:  XCTAssertFalse(allowed)
            }
        }
    }

    func testModeDidChangeFromSangatToSevadarIsSafeWithoutScreen() {
        // No external screen + flip to Sevadar should not crash and
        // should not falsely mark the coordinator as connected. The
        // implementation calls `attach(to:)` only when `UIScreen` has
        // a non-main entry; macOS test host has none, so this is the
        // safe-no-op branch.
        let env = AppEnvironment.preview()
        env.mode = .sangat
        let coord = CastSceneCoordinator(env: env)
        XCTAssertFalse(coord.isExternalScreenConnected)
        env.mode = .sevadar
        coord.modeDidChange(to: .sevadar)
        XCTAssertFalse(coord.isExternalScreenConnected,
                       "No external screen present → mode flip must not fabricate a connection.")
    }

    func testModeDidChangeFromSevadarToSangatDetachesIdempotently() {
        // Starting state: not attached. Flip to Sangat — already in
        // the right state, must be a no-op (no crash, no flag flip).
        let env = AppEnvironment.preview()
        env.mode = .sevadar
        let coord = CastSceneCoordinator(env: env)
        XCTAssertFalse(coord.isExternalScreenConnected)
        coord.modeDidChange(to: .sangat)
        XCTAssertFalse(coord.isExternalScreenConnected)
    }
}
