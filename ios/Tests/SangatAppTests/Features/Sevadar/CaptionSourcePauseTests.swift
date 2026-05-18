//
//  CaptionSourcePauseTests.swift
//  SangatAppTests — M5.3 (Sevadar surfaces)
//
//  Verifies `CaptionSource.pause()` / `resume()` semantics on
//  `DemoCaptionSource`. Pause is distinct from stop: state and
//  currentGuess survive a pause. The dock's "Pause auto" button
//  depends on this contract so manual nudges stick while paused.

import XCTest
@testable @_spi(Testing) import SangatApp
import GurbaniCaptioning

@MainActor
final class CaptionSourcePauseTests: XCTestCase {

    func testInitialPauseFlagIsFalse() {
        let source = DemoCaptionSource()
        XCTAssertFalse(source.isPaused)
    }

    func testPauseSetsIsPausedTrue() {
        let source = DemoCaptionSource()
        source.pause()
        XCTAssertTrue(source.isPaused)
    }

    func testResumeClearsIsPaused() {
        let source = DemoCaptionSource()
        source.pause()
        source.resume()
        XCTAssertFalse(source.isPaused)
    }

    func testPauseIsIdempotent() {
        let source = DemoCaptionSource()
        source.pause()
        source.pause()
        XCTAssertTrue(source.isPaused)
    }

    func testResumeWhenNotPausedIsNoOp() {
        let source = DemoCaptionSource()
        source.resume()
        XCTAssertFalse(source.isPaused)
    }

    func testPausePreservesCommittedStateAndGuess() {
        let source = DemoCaptionSource()
        source.manuallyCommit(shabadId: 1789)

        let chunk = AsrChunk(start: 0, end: 5, text: "ਤਾਤੀ")
        let guess = LineGuess(
            chunk: chunk, shabadId: 1789, lineIdx: 2,
            confidence: 88.0, isCommitted: true
        )
        source._testInject(guess: guess)

        source.pause()

        // pause must NOT clear state or currentGuess (that's stop()'s job)
        XCTAssertEqual(source.committedShabadId, 1789)
        XCTAssertEqual(source.currentGuess?.lineIdx, 2)
    }

    func testStartAfterStopResetsToListening() async throws {
        // Regression: tapping the back chevron calls stop(), then
        // tapping LISTEN again calls start(). If start() doesn't
        // reset state, the second session inherits the previous
        // session's `state = .committed` and the user is taken
        // straight to a stale reading view, skipping the
        // listening → tentative → committed transitions entirely.
        let source = DemoCaptionSource()

        // First session: drive into a committed-with-Hum-Aadmi state
        // (matches what a user would see after picking a shabad).
        source.manuallyCommit(shabadId: 660)
        XCTAssertEqual(source.committedShabadId, 660)
        XCTAssertTrue(source.isPaused, "manuallyCommit auto-pauses")

        // User taps back chevron.
        source.stop()
        XCTAssertFalse(source.isRunning)
        // stop() intentionally preserves state for reference.
        XCTAssertEqual(source.committedShabadId, 660)

        // User taps LISTEN again → expects a fresh session.
        try await source.start()

        XCTAssertTrue(source.isRunning, "start must flip isRunning back on")
        XCTAssertEqual(source.state, .listening,
                       "start after stop must reset state to .listening")
        XCTAssertNil(source.currentGuess,
                     "start after stop must clear stale currentGuess")
        XCTAssertFalse(source.isPaused,
                       "start after stop must clear any prior auto-pause")
    }

    func testManuallyCommitAutoPausesTheEngine() {
        // Regression: the demo playback task continues to emit
        // scripted events for Tati Vao after the user picks a
        // different shabad via the picker. Without auto-pause, the
        // next scripted `guessUpdated` overwrites the manual
        // selection a few seconds later — the screen flips back to
        // Tati Vao without any user action. manuallyCommit must
        // pause the engine so the manual pick sticks.
        let source = DemoCaptionSource()
        XCTAssertFalse(source.isPaused, "starts unpaused")

        source.manuallyCommit(shabadId: 660)

        XCTAssertTrue(source.isPaused, "manuallyCommit must auto-pause the engine")
        XCTAssertEqual(source.currentGuess?.shabadId, 660, "currentGuess follows the picked shabad")
    }

    func testModelManuallyCommitMirrorsPausedFlag() {
        // The dock's Pause/Resume label binds to model.isPaused.
        // After manuallyCommit auto-pauses the source, the model's
        // stored mirror must flip too — otherwise the dock keeps
        // showing "Pause auto" even though the engine is paused.
        let env = AppEnvironment.preview()
        XCTAssertFalse(env.captionModel.isPaused)

        env.captionModel.manuallyCommit(shabadId: 660)

        XCTAssertTrue(env.captionModel.isPaused, "model.isPaused must mirror after manuallyCommit auto-pause")
    }

    func testModelIsPausedMirrorsSourceAfterModelMethod() {
        // The original implementation made model.isPaused a computed
        // pass-through to source.isPaused, which didn't trigger
        // Observation — the dock's Pause/Resume label never updated
        // when the user tapped the button. The fix makes isPaused a
        // stored property on the model, mirrored after each call.
        let env = AppEnvironment.preview()
        XCTAssertFalse(env.captionModel.isPaused)

        env.captionModel.pause()
        XCTAssertTrue(env.captionModel.isPaused,
                      "model.isPaused must flip after model.pause()")

        env.captionModel.resume()
        XCTAssertFalse(env.captionModel.isPaused,
                       "model.isPaused must flip back after model.resume()")
    }

    func testNudgeStillWorksWhilePaused() {
        // The whole point of pause: stop engine emissions, but let
        // the Sevadar continue to nudge manually.
        let source = DemoCaptionSource()
        source.manuallyCommit(shabadId: 1789)

        let chunk = AsrChunk(start: 0, end: 5, text: "ਤਾਤੀ")
        let guess = LineGuess(
            chunk: chunk, shabadId: 1789, lineIdx: 1,
            confidence: 88.0, isCommitted: true
        )
        source._testInject(guess: guess)

        source.pause()
        source.nudge(by: 2)

        XCTAssertTrue(source.isPaused)
        XCTAssertEqual(source.currentGuess?.lineIdx, 3, "nudge must still work while paused")
    }
}
