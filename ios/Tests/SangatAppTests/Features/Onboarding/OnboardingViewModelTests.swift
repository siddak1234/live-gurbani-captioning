//
//  OnboardingViewModelTests.swift
//  SangatAppTests — M5.4 (Onboarding)
//
//  Pure state-machine tests for the onboarding view model. Excludes
//  AVAudio-bound behavior (those live in MicPermissionFlowTests) and
//  env-commit behavior (those live in OnboardingCompletionTests).

import XCTest
@testable import SangatApp

@MainActor
final class OnboardingViewModelTests: XCTestCase {

    func testStartsAtWelcome() {
        let vm = OnboardingViewModel()
        XCTAssertEqual(vm.step, .welcome)
    }

    func testAdvanceCyclesThroughAllStepsAndStopsAtRole() {
        let vm = OnboardingViewModel()

        vm.advance()
        XCTAssertEqual(vm.step, .howItWorks)

        vm.advance()
        XCTAssertEqual(vm.step, .mic)

        vm.advance()
        XCTAssertEqual(vm.step, .role)

        // Advancing past role is a no-op — completion is owned by
        // OnboardingFlow, not the view model.
        vm.advance()
        XCTAssertEqual(vm.step, .role)
        XCTAssertTrue(vm.isAtFinalStep)
    }

    func testBackFromWelcomeIsNoOp() {
        let vm = OnboardingViewModel()
        vm.back()
        XCTAssertEqual(vm.step, .welcome)
    }

    func testBackFromHowItWorksReturnsToWelcome() {
        let vm = OnboardingViewModel(initial: .howItWorks)
        vm.back()
        XCTAssertEqual(vm.step, .welcome)
    }

    func testBackFromMicReturnsToHowItWorks() {
        let vm = OnboardingViewModel(initial: .mic)
        vm.back()
        XCTAssertEqual(vm.step, .howItWorks)
    }

    func testBackFromRoleReturnsToMic() {
        let vm = OnboardingViewModel(initial: .role)
        vm.back()
        XCTAssertEqual(vm.step, .mic)
    }

    func testDefaultRoleIsSangat() {
        let vm = OnboardingViewModel()
        XCTAssertEqual(vm.selectedRole, .sangat)
    }

    func testSelectingSevadarFlipsRole() {
        let vm = OnboardingViewModel(initial: .role)
        vm.selectedRole = .sevadar
        XCTAssertEqual(vm.selectedRole, .sevadar)
    }

    func testInitialStepCanBeOverridden() {
        let vm = OnboardingViewModel(initial: .mic)
        XCTAssertEqual(vm.step, .mic)
    }

    func testOnboardingStepIndexMatchesCaseIterableOrder() {
        XCTAssertEqual(OnboardingStep.welcome.index, 0)
        XCTAssertEqual(OnboardingStep.howItWorks.index, 1)
        XCTAssertEqual(OnboardingStep.mic.index, 2)
        XCTAssertEqual(OnboardingStep.role.index, 3)
    }
}
