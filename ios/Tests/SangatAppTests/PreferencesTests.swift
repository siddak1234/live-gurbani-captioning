//
//  PreferencesTests.swift
//  SangatAppTests — M5.1 (Foundations)
//
//  Verifies the in-memory `Preferences` facade. We do not test the
//  `.standard` `UserDefaults` path because tests must not leak global state.

import XCTest
@testable import SangatApp

@MainActor
final class PreferencesTests: XCTestCase {

    func testThemeRoundtrip() {
        let prefs = Preferences.inMemory()
        XCTAssertNil(prefs.theme)
        prefs.theme = .darbar
        XCTAssertEqual(prefs.theme, .darbar)
    }

    func testModeRoundtrip() {
        let prefs = Preferences.inMemory()
        XCTAssertNil(prefs.mode)
        prefs.mode = .sevadar
        XCTAssertEqual(prefs.mode, .sevadar)
    }

    func testOnboardingDefaultsToFalse() {
        let prefs = Preferences.inMemory()
        XCTAssertFalse(prefs.hasCompletedOnboarding)
        prefs.hasCompletedOnboarding = true
        XCTAssertTrue(prefs.hasCompletedOnboarding)
    }

    func testReadingLayoutRoundtrip() {
        let prefs = Preferences.inMemory()
        XCTAssertNil(prefs.readingLayout)
        prefs.readingLayout = .karaoke
        XCTAssertEqual(prefs.readingLayout, .karaoke)
    }

    func testTranslitDefaultsToTrue() {
        let prefs = Preferences.inMemory()
        XCTAssertTrue(prefs.translitEnabled)
        prefs.translitEnabled = false
        XCTAssertFalse(prefs.translitEnabled)
    }

    func testMeaningDefaultsToTrue() {
        let prefs = Preferences.inMemory()
        XCTAssertTrue(prefs.meaningEnabled)
        prefs.meaningEnabled = false
        XCTAssertFalse(prefs.meaningEnabled)
    }

    func testCorrectionsOptInDefaultsToFalse() {
        let prefs = Preferences.inMemory()
        XCTAssertFalse(prefs.correctionsOptIn)
    }

    func testEachInMemoryInstanceIsIsolated() {
        let p1 = Preferences.inMemory()
        let p2 = Preferences.inMemory()
        p1.theme = .darbar
        XCTAssertNil(p2.theme, "In-memory Preferences should be isolated per instance")
    }
}
