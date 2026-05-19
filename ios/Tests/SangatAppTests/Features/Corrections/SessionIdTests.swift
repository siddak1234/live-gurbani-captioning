//
//  SessionIdTests.swift
//  SangatAppTests — M5.6 (Correction loop surfaces)
//
//  Behavior contract for `AppEnvironment.sessionId` + `startNewSession()`.
//  These are tiny but load-bearing: every `CorrectionEvent` stamps
//  `sessionId`, so any regression here corrupts the off-device fine-
//  tune pipeline's ability to group corrections by sitting.

import XCTest
@testable import SangatApp

@MainActor
final class SessionIdTests: XCTestCase {

    func testEnvSeedsASessionIdAtInit() {
        let env = AppEnvironment.preview()
        // No assertion on the specific value — it's a UUID. The
        // important thing is that the field is present (the type
        // system already guarantees non-nil); read it once to
        // confirm no crash and a stable value across reads.
        let first = env.sessionId
        XCTAssertEqual(first, env.sessionId, "sessionId must be stable until startNewSession()")
    }

    func testStartNewSessionRollsTheId() {
        let env = AppEnvironment.preview()
        let original = env.sessionId
        env.startNewSession()
        XCTAssertNotEqual(env.sessionId, original,
                          "startNewSession() must mint a fresh UUID")
    }

    func testStartNewSessionIsIdempotentInTheSenseThatItKeepsRolling() {
        // Two back-to-back calls must produce two different ids — the
        // "two Let's Begin taps without a relaunch" case.
        let env = AppEnvironment.preview()
        env.startNewSession()
        let first = env.sessionId
        env.startNewSession()
        XCTAssertNotEqual(env.sessionId, first)
    }
}
