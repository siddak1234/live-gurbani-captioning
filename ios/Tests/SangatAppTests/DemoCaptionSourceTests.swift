//
//  DemoCaptionSourceTests.swift
//  SangatAppTests — M5.1 (Foundations)
//
//  Verifies the scripted timeline plays out in real time. Uses
//  `DemoScript.quickCommit` to keep waits short.

import XCTest
import GurbaniCaptioning
@testable import SangatApp

@MainActor
final class DemoCaptionSourceTests: XCTestCase {

    func testQuickCommitReachesCommittedState() async throws {
        let source = DemoCaptionSource(script: .quickCommit)
        try await source.start()

        // quickCommit schedules .committed at 0.5s. We give ourselves a
        // generous 2s budget so flaky CI hosts have headroom.
        try await waitFor(timeout: 2.0) { source.isCommitted }

        XCTAssertEqual(source.committedShabadId, 1789)
        XCTAssertNotNil(source.currentGuess)
        if let guess = source.currentGuess {
            XCTAssertEqual(guess.shabadId, 1789)
            XCTAssertTrue(guess.isCommitted)
        }
    }

    func testEventsStreamEmitsLifecycleEvents() async throws {
        let source = DemoCaptionSource(script: .quickCommit)

        let collected = Task { @MainActor [source] in
            var events: [CaptionSourceEvent] = []
            for await event in source.events {
                events.append(event)
                if case .stateChanged(.committed) = event { return events }
                if events.count > 50 { return events }  // safety
            }
            return events
        }

        try await source.start()
        let events = await collected.value

        XCTAssertTrue(
            events.contains(where: {
                if case .started = $0 { return true }
                return false
            }),
            "Expected a .started event in the stream"
        )
        XCTAssertTrue(
            events.contains(where: {
                if case .stateChanged = $0 { return true }
                return false
            }),
            "Expected at least one .stateChanged event"
        )
    }

    func testResetWhileRunningClearsGuess() async throws {
        let source = DemoCaptionSource(script: .quickCommit)
        try await source.start()
        try await waitFor(timeout: 2.0) { source.isCommitted }
        source.resetShabad()
        XCTAssertNil(source.currentGuess)
        XCTAssertTrue(source.runnerUps.isEmpty)
    }

    // MARK: - Helpers

    /// Poll `condition` every 50ms up to `timeout` seconds. Used because we
    /// don't want to hardcode exact sleep times in tests against a real
    /// scripted timeline.
    private func waitFor(
        timeout: TimeInterval,
        _ condition: @MainActor () -> Bool
    ) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        XCTFail("Timed out waiting for condition after \(timeout)s")
    }
}
