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
