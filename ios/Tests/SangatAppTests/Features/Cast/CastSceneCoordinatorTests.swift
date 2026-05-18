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
}
