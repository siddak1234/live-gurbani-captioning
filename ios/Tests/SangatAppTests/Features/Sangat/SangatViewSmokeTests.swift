//
//  SangatViewSmokeTests.swift
//  SangatAppTests — M5.2 (Sangat reading)
//
//  Smoke tests that every Sangat reading view can be instantiated with a
//  preview environment without crashing. SwiftUI views are values, so
//  these are constructor + initial-body-evaluation checks.
//
//  Real interaction tests (state transitions on tap, scroll behavior in
//  FullShabadView) require XCUITest infrastructure and land later.

import XCTest
import SwiftUI
import GurbaniCaptioning
@testable import SangatApp

#if canImport(UIKit)
import UIKit
#endif

@MainActor
final class SangatViewSmokeTests: XCTestCase {

    func testIdleViewConstructs() {
        let view = IdleView()
            .environment(AppEnvironment.preview())
        _ = ViewProbe(content: view)
    }

    func testListeningViewConstructs() {
        let view = ListeningView()
            .environment(AppEnvironment.preview())
        _ = ViewProbe(content: view)
    }

    func testTentativeViewConstructs() {
        let view = TentativeView(shabadId: 1789)
            .environment(AppEnvironment.preview())
        _ = ViewProbe(content: view)
    }

    func testHeroLineViewConstructsForEveryDemoLine() {
        for idx in 0..<6 {
            let view = HeroLineView(guess: sampleGuess(lineIdx: idx))
                .environment(AppEnvironment.preview())
            _ = ViewProbe(content: view)
        }
    }

    func testKaraokeViewConstructsForBoundaryLines() {
        for idx in [0, 1, 5] {
            let view = KaraokeView(guess: sampleGuess(lineIdx: idx))
                .environment(AppEnvironment.preview())
            _ = ViewProbe(content: view)
        }
    }

    func testFullShabadViewConstructs() {
        let view = FullShabadView(guess: sampleGuess(lineIdx: 2))
            .environment(AppEnvironment.preview())
        _ = ViewProbe(content: view)
    }

    func testReadingHostConstructsForEveryLayout() {
        for layout in ReadingLayout.allCases {
            let env = AppEnvironment.preview()
            env.readingLayout = layout
            let view = ReadingHost()
                .environment(env)
            _ = ViewProbe(content: view)
        }
    }

    // MARK: - Fixtures

    private func sampleGuess(lineIdx: Int) -> LineGuess {
        LineGuess(
            chunk: AsrChunk(start: 0, end: 5, text: ""),
            shabadId: 1789,
            lineIdx: lineIdx,
            confidence: 88.4,
            isCommitted: true
        )
    }
}

/// Forces a view's body to evaluate for instantiation checks. We don't
/// render to pixels — just construct the wrapper so the body computation
/// runs at least once.
private struct ViewProbe<Content: View> {
    init(content: Content) {
        #if canImport(UIKit)
        _ = UIHostingController(rootView: content)
        #else
        _ = NSHostingController(rootView: content)
        #endif
    }
}

#if canImport(AppKit) && !canImport(UIKit)
import AppKit
#endif
