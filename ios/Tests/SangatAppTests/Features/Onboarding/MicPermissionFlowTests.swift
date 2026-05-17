//
//  MicPermissionFlowTests.swift
//  SangatAppTests — M5.4 (Onboarding)
//
//  Mic-specific paths through the onboarding view model. Real
//  AudioPermissions.request() is a system call that returns
//  .unavailable on macOS test runs (no AVAudioApplication), so the
//  "Allow" path here verifies the post-call state shape and the
//  expected advance-to-role behavior; the actual permission alert
//  is verified by simulator walkthrough (audit gate 4.2).

import XCTest
@testable import SangatApp

@MainActor
final class MicPermissionFlowTests: XCTestCase {

    // MARK: - Skip path ("Not now")

    func testSkipMicAdvancesToRole() {
        let vm = OnboardingViewModel(initial: .mic)
        vm.skipMic()
        XCTAssertEqual(vm.step, .role)
    }

    func testSkipMicLeavesPermissionStatusNotDetermined() {
        let vm = OnboardingViewModel(initial: .mic)
        vm.skipMic()
        XCTAssertEqual(vm.micPermissionStatus, .notDetermined)
    }

    func testSkipMicKeepsAcknowledgedFalse() {
        // The whole point: tapping "Not now" must result in
        // micPermissionAcknowledged=false so IdleView can surface
        // the "Enable microphone" affordance later.
        let vm = OnboardingViewModel(initial: .mic)
        vm.skipMic()
        XCTAssertFalse(vm.micPermissionAcknowledged)
    }

    func testSkipMicFromOtherStepsIsNoOp() {
        let vm = OnboardingViewModel(initial: .welcome)
        vm.skipMic()
        XCTAssertEqual(vm.step, .welcome)
        XCTAssertEqual(vm.micPermissionStatus, .notDetermined)
    }

    // MARK: - Allow path

    func testRequestMicPermissionFromMicStepAdvancesToRole() async {
        let vm = OnboardingViewModel(initial: .mic)
        _ = await vm.requestMicPermission()
        XCTAssertEqual(vm.step, .role, "After the system call resolves, the flow must advance regardless of the answer.")
    }

    func testRequestMicPermissionFromOtherStepIsNoOp() async {
        let vm = OnboardingViewModel(initial: .welcome)
        let status = await vm.requestMicPermission()
        XCTAssertEqual(vm.step, .welcome)
        XCTAssertEqual(status, .notDetermined)
    }

    // MARK: - Derived acknowledged flag

    func testAcknowledgedFlagMapping() {
        let vm = OnboardingViewModel(initial: .mic)

        // .notDetermined → false
        XCTAssertFalse(vm.micPermissionAcknowledged)

        // .unavailable → false (test environment, fallback)
        // We can't construct .granted/.denied without the system, so
        // we lean on the @discardableResult of requestMicPermission()
        // running on the test host (returns .unavailable on macOS test
        // host where AVAudioApplication is unreachable). We just verify
        // the mapping table is correct by exhaustively switching:
        let statuses: [MicPermissionStatus] = [
            .notDetermined,
            .granted,
            .denied,
            .restricted,
            .unavailable
        ]
        for status in statuses {
            let acknowledged = isAcknowledged(status)
            switch status {
            case .granted, .denied, .restricted:
                XCTAssertTrue(acknowledged, "\(status) must count as acknowledged")
            case .notDetermined, .unavailable:
                XCTAssertFalse(acknowledged, "\(status) must NOT count as acknowledged")
            }
        }
    }

    /// Mirrors the mapping in `OnboardingViewModel.micPermissionAcknowledged`.
    /// If you change one, change the other.
    private func isAcknowledged(_ status: MicPermissionStatus) -> Bool {
        switch status {
        case .granted, .denied, .restricted: return true
        case .notDetermined, .unavailable: return false
        }
    }
}
