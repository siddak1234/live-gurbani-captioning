//
//  CaptionSourceNudgeTests.swift
//  SangatAppTests — M5.3 (Sevadar surfaces)
//
//  Verifies `CaptionSource.nudge(by:)` semantics on `DemoCaptionSource`:
//  the dock's Line ±1 buttons depend on this contract. Upper-bound
//  clamping (totalLines) is the dock's responsibility; the source
//  only clamps at zero.

import XCTest
@testable @_spi(Testing) import SangatApp
import GurbaniCaptioning

@MainActor
final class CaptionSourceNudgeTests: XCTestCase {

    private func makeSource(initialLineIdx: Int) -> DemoCaptionSource {
        let source = DemoCaptionSource()
        // Drive the source into a .committed state with a guess so
        // nudge has something to mutate. We do this via manuallyCommit
        // + a synthetic guess on the public surface.
        source.manuallyCommit(shabadId: 1789)
        let chunk = AsrChunk(start: 0, end: 5, text: "ਤਾਤੀ")
        let guess = LineGuess(
            chunk: chunk,
            shabadId: 1789,
            lineIdx: initialLineIdx,
            confidence: 88.0,
            isCommitted: true
        )
        // We can't directly set currentGuess from outside, but the
        // source's apply(step:) path runs the same logic — we use the
        // DemoStep API to inject a guess.
        // For these tests we replicate that injection via a quick
        // workaround: post a step to the playback loop is overkill;
        // instead we rely on the public DemoStep path being used by
        // the script, and use the nudge() method which only requires
        // currentGuess + .committed. Since we can't set it directly,
        // we use a known-good script entry by running the script then
        // calling nudge. But simpler: use the package's `manuallyCommit
        // + nudge` sequence — nudge is a no-op if currentGuess is nil.
        //
        // Workaround: bypass setup constraints by exposing a test
        // shim on DemoCaptionSource that lets us seed currentGuess.
        source._testInject(guess: guess)
        return source
    }

    func testNudgeForwardIncrementsLineIdx() {
        let source = makeSource(initialLineIdx: 2)
        source.nudge(by: 1)
        XCTAssertEqual(source.currentGuess?.lineIdx, 3)
    }

    func testNudgeBackwardDecrementsLineIdx() {
        let source = makeSource(initialLineIdx: 4)
        source.nudge(by: -1)
        XCTAssertEqual(source.currentGuess?.lineIdx, 3)
    }

    func testNudgeClampsAtZero() {
        let source = makeSource(initialLineIdx: 0)
        source.nudge(by: -1)
        XCTAssertEqual(source.currentGuess?.lineIdx, 0, "lineIdx must not go negative")
    }

    func testNudgeWithoutCommittedStateIsNoOp() {
        let source = DemoCaptionSource()
        // Not committed, no currentGuess — nudge must do nothing.
        source.nudge(by: 5)
        XCTAssertNil(source.currentGuess)
    }

    func testZeroNudgeIsNoOp() {
        let source = makeSource(initialLineIdx: 3)
        let before = source.currentGuess
        source.nudge(by: 0)
        XCTAssertEqual(source.currentGuess?.lineIdx, before?.lineIdx)
    }

    func testNudgePreservesShabadIdAndChunk() {
        let source = makeSource(initialLineIdx: 1)
        let before = source.currentGuess!
        source.nudge(by: 2)
        XCTAssertEqual(source.currentGuess?.shabadId, before.shabadId)
        XCTAssertEqual(source.currentGuess?.confidence, before.confidence)
        XCTAssertEqual(source.currentGuess?.isCommitted, before.isCommitted)
    }
}
