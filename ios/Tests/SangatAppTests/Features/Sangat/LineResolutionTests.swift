//
//  LineResolutionTests.swift
//  SangatAppTests — M5.2 (Sangat reading)
//
//  Pins the AppEnvironment line-resolution helpers used by every Sangat
//  reading view. If a future change breaks the demo-fallback path,
//  reading views would silently fail to render content — these tests
//  catch that.

import XCTest
@testable import SangatApp

@MainActor
final class LineResolutionTests: XCTestCase {

    func testResolveLineReturnsKnownDemoLine() {
        let env = AppEnvironment.preview()
        let resolved = env.resolveLine(shabadId: 1789, lineIdx: 0)
        XCTAssertNotNil(resolved)
        XCTAssertTrue(resolved?.gurmukhi.contains("ਤਾਤੀ") ?? false)
        XCTAssertNotNil(resolved?.transliteration)
        XCTAssertNotNil(resolved?.english)
        XCTAssertNotNil(resolved?.verseId)
    }

    func testResolveLineForEachDemoLineIdx() {
        let env = AppEnvironment.preview()
        for idx in 0..<6 {
            let resolved = env.resolveLine(shabadId: 1789, lineIdx: idx)
            XCTAssertNotNil(resolved, "Line \(idx) should resolve in demo data")
            XCTAssertFalse(resolved?.gurmukhi.isEmpty ?? true, "Gurmukhi at \(idx) is empty")
        }
    }

    func testResolveLineOutOfBoundsReturnsNil() {
        let env = AppEnvironment.preview()
        XCTAssertNil(env.resolveLine(shabadId: 1789, lineIdx: -1))
        XCTAssertNil(env.resolveLine(shabadId: 1789, lineIdx: 999))
    }

    func testTotalLinesMatchesDemoLength() {
        let env = AppEnvironment.preview()
        XCTAssertEqual(env.totalLines(forShabadId: 1789), 6)
    }

    func testShabadMetaReturnsBilaaval() {
        let env = AppEnvironment.preview()
        let meta = env.shabadMeta(forShabadId: 1789)
        XCTAssertEqual(meta.raag, "Bilaaval")
        XCTAssertEqual(meta.ang, 819)
        XCTAssertNotNil(meta.author)
        XCTAssertNotNil(meta.authorGurmukhi)
    }

    func testResolvedLineIsEquatable() {
        let a = ResolvedLine(gurmukhi: "ਤੇਸਟ", transliteration: "test")
        let b = ResolvedLine(gurmukhi: "ਤੇਸਟ", transliteration: "test")
        let c = ResolvedLine(gurmukhi: "ਤੇਸਟ2", transliteration: "test")
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }

    func testShabadMetaIsEquatable() {
        let a = ShabadMeta(raag: "X", ang: 1, author: "A", authorGurmukhi: "ਾ")
        let b = ShabadMeta(raag: "X", ang: 1, author: "A", authorGurmukhi: "ਾ")
        let c = ShabadMeta(raag: "Y", ang: 1)
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }
}
