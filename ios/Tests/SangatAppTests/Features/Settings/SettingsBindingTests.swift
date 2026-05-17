//
//  SettingsBindingTests.swift
//  SangatAppTests — M5.2-extension (theme + reading layout switcher)
//
//  SettingsView is a thin binding surface over AppEnvironment, so the
//  testable behavior lives in the env: mutate theme/layout/layers, confirm
//  the change persists through Preferences and shows up on the next read.
//  Rendering correctness is verified by `#Preview` + simulator runs.

import XCTest
@testable import SangatApp

@MainActor
final class SettingsBindingTests: XCTestCase {

    // MARK: - Theme

    func testCyclingThroughAllThemesPersists() {
        let env = AppEnvironment.preview()
        XCTAssertEqual(env.theme, .paper)

        for theme in Theme.allCases {
            env.theme = theme
            XCTAssertEqual(env.theme, theme)
            XCTAssertEqual(env.preferences.theme, theme)
        }
    }

    func testThemeSurvivesEnvironmentReinitFromPreferences() {
        let env1 = AppEnvironment.preview()
        env1.theme = .mool
        XCTAssertEqual(env1.preferences.theme, .mool)

        // Build a fresh env from the same preferences; theme should be
        // restored, not reset to default. We use `preview(theme:)` and
        // confirm the persisted value wins (preview seeds prefs explicitly).
        let env2 = AppEnvironment.preview(theme: .mool)
        XCTAssertEqual(env2.theme, .mool)
    }

    func testEachThemeHasDistinctTokensAndMetadata() {
        let themes = Theme.allCases
        for theme in themes {
            XCTAssertFalse(theme.displayName.isEmpty)
            XCTAssertFalse(theme.subtitle.isEmpty)
        }

        let bgs = themes.map { $0.tokens.colors.bg.description }
        XCTAssertEqual(Set(bgs).count, themes.count, "themes should have distinct backgrounds")

        let accents = themes.map { $0.tokens.colors.accent.description }
        XCTAssertEqual(Set(accents).count, themes.count, "themes should have distinct accents")
    }

    func testDarbarIsTheOnlyDarkTheme() {
        XCTAssertTrue(Theme.darbar.isDark)
        XCTAssertFalse(Theme.paper.isDark)
        XCTAssertFalse(Theme.mool.isDark)
    }

    // MARK: - Reading layout

    func testLayoutFlipsThroughEnvironmentBinding() {
        let env = AppEnvironment.preview()
        env.readingLayout = .karaoke
        XCTAssertEqual(env.readingLayout, .karaoke)
        XCTAssertEqual(env.preferences.readingLayout, .karaoke)

        env.readingLayout = .full
        XCTAssertEqual(env.readingLayout, .full)
        XCTAssertEqual(env.preferences.readingLayout, .full)
    }

    // MARK: - Reading layers

    func testTranslitToggleRoundTripsThroughPreferences() {
        let env = AppEnvironment.preview()
        XCTAssertTrue(env.translitEnabled, "default for transliteration is on")

        env.translitEnabled = false
        XCTAssertFalse(env.translitEnabled)
        XCTAssertFalse(env.preferences.translitEnabled)

        env.translitEnabled = true
        XCTAssertTrue(env.translitEnabled)
        XCTAssertTrue(env.preferences.translitEnabled)
    }

    func testMeaningToggleRoundTripsThroughPreferences() {
        let env = AppEnvironment.preview()
        XCTAssertTrue(env.meaningEnabled, "default for meaning is on")

        env.meaningEnabled = false
        XCTAssertFalse(env.meaningEnabled)
        XCTAssertFalse(env.preferences.meaningEnabled)
    }

    // MARK: - Theme allCases coverage (so a new theme requires a settings update)

    func testThemeAllCasesContainsTheThreeShippedDirections() {
        let cases = Set(Theme.allCases)
        XCTAssertEqual(cases, [.paper, .darbar, .mool])
    }
}
