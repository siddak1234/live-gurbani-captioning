//
//  AppEnvironmentTests.swift
//  SangatAppTests — M5.1 (Foundations)
//
//  Tests for the composition root. Verifies that env construction is
//  deterministic, persistence writes through `Preferences`, and the
//  observable state changes flip the right preferences keys.

import XCTest
@testable import SangatApp

@MainActor
final class AppEnvironmentTests: XCTestCase {

    func testPreviewEnvIsConstructibleWithDefaults() {
        let env = AppEnvironment.preview()
        XCTAssertEqual(env.theme, Theme.default)
        XCTAssertEqual(env.mode, AppMode.default)
        XCTAssertTrue(env.hasCompletedOnboarding)
        XCTAssertEqual(env.readingLayout, ReadingLayout.default)
        XCTAssertTrue(env.translitEnabled)
        XCTAssertTrue(env.meaningEnabled)
    }

    func testPreviewEnvUsesGivenInitialTheme() {
        let env = AppEnvironment.preview(theme: .darbar)
        XCTAssertEqual(env.theme, .darbar)
        XCTAssertEqual(env.preferences.theme, .darbar)
    }

    func testThemeChangePersistsThroughPreferences() {
        let env = AppEnvironment.preview(theme: .paper)
        env.theme = .mool
        XCTAssertEqual(env.preferences.theme, .mool)
    }

    func testModeChangePersistsThroughPreferences() {
        let env = AppEnvironment.preview(mode: .sangat)
        env.mode = .sevadar
        XCTAssertEqual(env.preferences.mode, .sevadar)
    }

    func testOnboardingFlagPersists() {
        // M5.4 re-introduces persistence: completing onboarding (which now
        // asks for mic permission and picks a role) must survive cold
        // starts. The brief session-only window from `b63bd38` was a
        // stopgap for the M5.1 placeholder.
        let env = AppEnvironment.preview(hasCompletedOnboarding: false)
        XCTAssertFalse(env.hasCompletedOnboarding)
        env.hasCompletedOnboarding = true
        XCTAssertTrue(env.preferences.hasCompletedOnboarding)
    }

    func testReadingLayoutPersists() {
        let env = AppEnvironment.preview()
        env.readingLayout = .karaoke
        XCTAssertEqual(env.preferences.readingLayout, .karaoke)
    }

    func testServicesAreInjected() {
        let env = AppEnvironment.preview()
        XCTAssertNotNil(env.captionSource)
        XCTAssertNotNil(env.captionModel)
        XCTAssertNotNil(env.correctionLog)
        XCTAssertNotNil(env.preferences)
        XCTAssertNotNil(env.haptics)
        XCTAssertNotNil(env.sessionHistory)
    }
}
