//
//  CastReadingViewSmokeTests.swift
//  SangatAppTests — M5.5 (Cast)
//
//  Pixel-level rendering and external-display attachment are sim
//  gates (5.1-5.6). These tests cover the parts that can run on
//  the macOS test host: presence of the new tokens in all three
//  presets, and init of the view value itself.

import XCTest
import SwiftUI
@testable import SangatApp

@MainActor
final class CastReadingViewSmokeTests: XCTestCase {

    /// Catches the most likely M5.5 regression: forgetting to add a
    /// new typography token to one of the two presets. Without this
    /// guard a wrong-theme cast view would crash at render time on a
    /// connected projector — late, embarrassing, and only visible to
    /// the sangat.
    func testCastTokensAreDefinedAcrossPresets() {
        for theme in Theme.allCases {
            XCTAssertNotNil(theme.tokens.type.gurmukhiCast,
                            "gurmukhiCast must be defined in \(theme.rawValue) preset")
            XCTAssertNotNil(theme.tokens.type.serifCastBody,
                            "serifCastBody must be defined in \(theme.rawValue) preset")
        }
    }

    /// Verify the cast view value can be constructed without trapping
    /// in init for any theme. SwiftUI body invocation is deferred to
    /// real rendering, which only happens in the sim — but init-time
    /// errors (like force-unwrap or missing required env) would
    /// surface here.
    func testInitDoesNotTrapAcrossThemes() {
        _ = CastReadingView()  // bare init must work
        for theme in Theme.allCases {
            _ = CastReadingView()
                .environment(AppEnvironment.preview(theme: theme))
        }
    }
}
