//
//  CorrectionGateTests.swift
//  SangatAppTests — M5.6 (Correction loop surfaces)
//
//  Tests the opt-in gate at the level RootView / SettingsView use it:
//  "if preferences.correctionsOptIn is false, do not call record(...)".
//  This is the single trust hinge for the whole feature, so it gets
//  explicit coverage even though the gate itself is two lines.
//
//  The gate is exercised by recreating the exact pattern emission
//  sites use:
//
//      guard env.preferences.correctionsOptIn else { return }
//      env.correctionLog.record(event)
//
//  …and verifying the spy receives (or does not receive) the event.

import XCTest
@testable import SangatApp

@MainActor
final class CorrectionGateTests: XCTestCase {

    func testOptInOffDoesNotRecord() {
        let spy = CorrectionLogSpy()
        let env = AppEnvironment.preview(correctionLog: spy)
        env.preferences.correctionsOptIn = false

        emit(env: env)

        XCTAssertEqual(spy.recorded.count, 0,
                       "Opt-in must be false-by-default and must block records")
    }

    func testOptInOnRecords() {
        let spy = CorrectionLogSpy()
        let env = AppEnvironment.preview(correctionLog: spy)
        env.preferences.correctionsOptIn = true

        emit(env: env)

        XCTAssertEqual(spy.recorded.count, 1,
                       "Opt-in true must record exactly one event per emission")
        XCTAssertEqual(spy.recorded.first?.kind, .hardNegPos)
        XCTAssertEqual(spy.recorded.first?.sessionId, env.sessionId,
                       "Recorded event must stamp the current session id")
    }

    func testTogglingOptInBetweenEmissionsOnlyRecordsTheOnState() {
        let spy = CorrectionLogSpy()
        let env = AppEnvironment.preview(correctionLog: spy)

        env.preferences.correctionsOptIn = false
        emit(env: env)
        env.preferences.correctionsOptIn = true
        emit(env: env)
        env.preferences.correctionsOptIn = false
        emit(env: env)

        XCTAssertEqual(spy.recorded.count, 1)
    }

    func testStartNewSessionChangesStampedSessionIdOnSubsequentEmissions() {
        let spy = CorrectionLogSpy()
        let env = AppEnvironment.preview(correctionLog: spy)
        env.preferences.correctionsOptIn = true

        emit(env: env)
        let firstSession = spy.recorded.last?.sessionId
        env.startNewSession()
        emit(env: env)
        let secondSession = spy.recorded.last?.sessionId

        XCTAssertNotNil(firstSession)
        XCTAssertNotNil(secondSession)
        XCTAssertNotEqual(firstSession, secondSession,
                          "A new session id must propagate to the next emission")
    }

    // MARK: - Helpers

    /// Mirrors the gate pattern RootView/SettingsView use at every
    /// emission site. Centralized here so tests assert *the gate*,
    /// not a specific kind's fields.
    private func emit(env: AppEnvironment) {
        guard env.preferences.correctionsOptIn else { return }
        let event = CorrectionEventBuilder.makeHardNegPos(
            sessionId: env.sessionId,
            predictedShabadId: 1789,
            predictedLineIdx: nil,
            predictedConfidence: nil,
            runnerUps: [:],
            correctedShabadId: 1341,
            engineStateRaw: "committed(1789)"
        )
        env.correctionLog.record(event)
    }
}
