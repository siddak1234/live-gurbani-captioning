//
//  ReadingLayoutTests.swift
//  SangatAppTests — M5.2 (Sangat reading)
//
//  Verifies reading-layout switching round-trips through Preferences and
//  the env exposes the right value to consumers (ReadingHost).

import XCTest
@testable import SangatApp

@MainActor
final class ReadingLayoutTests: XCTestCase {

    func testDefaultReadingLayoutIsHero() {
        let env = AppEnvironment.preview()
        XCTAssertEqual(env.readingLayout, .hero)
    }

    func testCycleThroughAllLayouts() {
        let env = AppEnvironment.preview()
        for layout in ReadingLayout.allCases {
            env.readingLayout = layout
            XCTAssertEqual(env.readingLayout, layout)
            XCTAssertEqual(env.preferences.readingLayout, layout)
        }
    }

    func testEachLayoutHasDisplayMetadata() {
        for layout in ReadingLayout.allCases {
            XCTAssertFalse(layout.displayName.isEmpty)
            XCTAssertFalse(layout.subtitle.isEmpty)
        }
    }

    func testLayoutRawValueRoundtripsThroughCodable() throws {
        let original: [ReadingLayout] = ReadingLayout.allCases
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode([ReadingLayout].self, from: data)
        XCTAssertEqual(original, decoded)
    }
}
