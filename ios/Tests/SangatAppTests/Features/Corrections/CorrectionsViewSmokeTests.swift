//
//  CorrectionsViewSmokeTests.swift
//  SangatAppTests — M5.6 (Correction loop surfaces)
//
//  Body-evaluation smokes for the two M5.6 surfaces. We can't drive
//  taps or sheet presentation from the macOS test host, so the gates
//  here are "constructs cleanly under a preview env" + "doesn't crash
//  when forced through `UIHostingController`'s rootView assignment".
//
//  Interaction coverage of the wired emissions lives in the
//  simulator walkthrough (audit § 5).

import XCTest
import SwiftUI
@testable import SangatApp

#if canImport(UIKit)
import UIKit
#endif

@MainActor
final class CorrectionsViewSmokeTests: XCTestCase {

    func testWrongShabadSheetConstructs() {
        let view = WrongShabadSheet(predictedShabadId: 1789, onCorrect: { _ in })
            .environment(AppEnvironment.preview())
        _ = ViewProbe(content: view)
    }

    func testCorrectionsSettingsViewConstructs() {
        let view = CorrectionsSettingsView()
            .environment(AppEnvironment.preview())
        _ = ViewProbe(content: view)
    }

    func testCorrectionsSettingsViewConstructsWhenOptedIn() {
        let env = AppEnvironment.preview()
        env.preferences.correctionsOptIn = true
        let view = CorrectionsSettingsView()
            .environment(env)
        _ = ViewProbe(content: view)
    }
}

/// Mirrors the existing `ViewProbe` pattern used elsewhere in the
/// test target so test files stay self-contained.
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
