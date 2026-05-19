//
//  DemoCaptionSourceClockTests.swift
//  SangatAppTests — M5.6.x (carry-forward)
//
//  Two contracts, both surfaced during M5.6 sim testing:
//
//  Fix #2 (pause clock-stop) — Pre-M5.6.x, the scripted playback loop
//  kept ticking through delays during pause; steps whose sleeps
//  completed mid-pause were silently dropped. Result: pause → resume
//  jumped lines (e.g., 3 → 5). The fix holds each step at the gate
//  while paused. These tests pin that semantic.
//
//  Fix #3 (synthetic auto-advance) — Pre-M5.6.x, `manuallyCommit`
//  auto-paused but had nothing to "resume" because the scripted
//  timeline has line data for only one pre-baked shabad. Resume on a
//  manually-picked shabad was a no-op affordance. The fix kicks off a
//  synthetic loop that advances lineIdx by 1 every
//  `syntheticAdvanceInterval` while not paused. Tests use a short
//  interval (~120ms) so the suite stays fast.

import XCTest
@testable @_spi(Testing) import SangatApp
import GurbaniCaptioning

@MainActor
final class DemoCaptionSourceClockTests: XCTestCase {

    // MARK: - Fix #2: pause holds the script clock

    func testPauseHoldsScriptedTimelineUntilResume() async throws {
        // Build a 3-step script with short delays so the test runs
        // in well under a second. Each step commits a different
        // line, giving us a clear "did the next step apply?" signal.
        let script = DemoScript(
            name: "test.threeStep",
            shabadId: 1789,
            steps: [
                DemoStep(delay: 0.05, state: .committed(shabadId: 1789), guess: line(1789, 0)),
                DemoStep(delay: 0.05, guess: line(1789, 1)),
                DemoStep(delay: 0.05, guess: line(1789, 2))
            ]
        )
        let source = DemoCaptionSource(script: script)
        try await source.start()

        // Let step 0 fire.
        try await Task.sleep(nanoseconds: 80_000_000)
        XCTAssertEqual(source.currentGuess?.lineIdx, 0,
                       "First scripted step should have applied")

        // Pause BEFORE step 1's delay finishes.
        source.pause()

        // Wait through the original delays of steps 1 AND 2. Pre-fix,
        // both would have fired and been dropped by the apply-gate,
        // leaving us at lineIdx 0 — but with no way to recover them
        // on resume. Post-fix, both steps are held at the post-sleep
        // gate.
        try await Task.sleep(nanoseconds: 200_000_000)

        XCTAssertEqual(source.currentGuess?.lineIdx, 0,
                       "Paused: no further line advancement")

        // Resume — the next held step (step 1) should now apply.
        source.resume()
        try await Task.sleep(nanoseconds: 200_000_000)

        // We expect both step 1 and step 2 to fire in order. With the
        // 100ms pause-poll + 50ms step delays + 200ms wait, both have
        // ample time to clear. The contract: NO step was dropped — we
        // end on lineIdx 2, having visited 1 along the way.
        XCTAssertEqual(source.currentGuess?.lineIdx, 2,
                       "After resume, remaining steps fire in order — no drops")

        source.stop()
    }

    // MARK: - Fix #3: synthetic auto-advance after manualCommit

    func testManualCommitArmsSyntheticAutoAdvanceOnResume() async throws {
        let source = DemoCaptionSource(
            totalLinesProvider: { _ in 5 },
            syntheticAdvanceInterval: 0.12
        )
        try await source.start()

        source.manuallyCommit(shabadId: 1789)
        XCTAssertTrue(source.isPaused, "manualCommit auto-pauses (preserved contract)")
        XCTAssertEqual(source.currentGuess?.lineIdx, 0, "manualCommit lands on line 0")

        // While paused, the synthetic task waits at the gate. After
        // a generous interval we should still be on line 0.
        try await Task.sleep(nanoseconds: 200_000_000)
        XCTAssertEqual(source.currentGuess?.lineIdx, 0,
                       "Paused synthetic task must not advance")

        // Resume — synthetic task wakes; over the next ~300ms it
        // should tick at least twice (interval = 120ms).
        source.resume()
        try await Task.sleep(nanoseconds: 350_000_000)

        guard let after = source.currentGuess?.lineIdx else {
            return XCTFail("currentGuess unexpectedly nil after resume")
        }
        XCTAssertGreaterThanOrEqual(after, 1,
                                     "Resume must arm auto-advance; expected lineIdx >= 1, got \(after)")

        source.stop()
    }

    func testSyntheticAutoAdvanceWrapsAtTotalLines() async throws {
        let source = DemoCaptionSource(
            totalLinesProvider: { _ in 3 },
            syntheticAdvanceInterval: 0.08
        )
        try await source.start()

        source.manuallyCommit(shabadId: 1789)
        source.nudge(by: 2)  // start at line 2 of 3
        XCTAssertEqual(source.currentGuess?.lineIdx, 2)

        source.resume()
        try await Task.sleep(nanoseconds: 250_000_000)

        guard let after = source.currentGuess?.lineIdx else {
            return XCTFail("currentGuess unexpectedly nil")
        }
        // From line 2 (last line) → wrap to 0 → advance to 1.
        XCTAssertLessThan(after, 3, "Synthetic advance must wrap, not overflow lineCount")

        source.stop()
    }

    func testSyntheticAutoAdvanceHonorsPauseMidstream() async throws {
        let source = DemoCaptionSource(
            totalLinesProvider: { _ in 5 },
            syntheticAdvanceInterval: 0.08
        )
        try await source.start()

        source.manuallyCommit(shabadId: 1789)
        source.resume()
        try await Task.sleep(nanoseconds: 200_000_000)

        guard let snapshotA = source.currentGuess?.lineIdx else {
            return XCTFail("currentGuess unexpectedly nil after partial advance")
        }
        source.pause()
        try await Task.sleep(nanoseconds: 250_000_000)

        XCTAssertEqual(source.currentGuess?.lineIdx, snapshotA,
                       "Pause mid-stream must freeze lineIdx exactly where it was")

        source.stop()
    }

    func testNoProviderInjectedMeansSyntheticTaskIsANoOp() async throws {
        // Default DemoCaptionSource() — no provider passed. Synthetic
        // task still spins (so cancellation discipline is exercised)
        // but advanceOneLineSynthetic short-circuits on `nil` provider.
        let source = DemoCaptionSource(syntheticAdvanceInterval: 0.08)
        try await source.start()
        source.manuallyCommit(shabadId: 1789)
        source.resume()
        try await Task.sleep(nanoseconds: 250_000_000)
        XCTAssertEqual(source.currentGuess?.lineIdx, 0,
                       "Without a totalLinesProvider, synthetic mode holds position")
        source.stop()
    }

    func testStopCancelsSyntheticTask() async throws {
        let source = DemoCaptionSource(
            totalLinesProvider: { _ in 5 },
            syntheticAdvanceInterval: 0.08
        )
        try await source.start()
        source.manuallyCommit(shabadId: 1789)
        source.resume()
        try await Task.sleep(nanoseconds: 200_000_000)

        source.stop()
        guard let frozenAt = source.currentGuess?.lineIdx else {
            return XCTFail("currentGuess unexpectedly nil after stop")
        }
        try await Task.sleep(nanoseconds: 250_000_000)
        XCTAssertEqual(source.currentGuess?.lineIdx, frozenAt,
                       "stop() must cancel the synthetic task so no further advance fires")
    }

    // MARK: - Helpers

    private func line(_ shabadId: Int, _ idx: Int) -> LineGuess {
        LineGuess(
            chunk: AsrChunk(start: 0, end: 0, text: ""),
            shabadId: shabadId,
            lineIdx: idx,
            confidence: 100.0,
            isCommitted: true
        )
    }
}
