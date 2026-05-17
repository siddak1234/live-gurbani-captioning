//
//  CaptionSourceContractTests.swift
//  SangatAppTests — M5.1 (Foundations)
//
//  Contract tests every `CaptionSource` implementation must satisfy.
//  Today only `DemoCaptionSource` is exercised; when `LiveCaptionSource`
//  is wired in M5.6/M5.7 the same contract block runs against it (in a
//  conditional `#if WHISPERKIT_AVAILABLE` test) to guarantee parity at
//  the protocol boundary.

import XCTest
import GurbaniCaptioning
@testable import SangatApp

@MainActor
final class CaptionSourceContractTests: XCTestCase {

    // MARK: - Lifecycle

    func testDemoSourceStartsAndStops() async throws {
        let source = DemoCaptionSource(script: .quickCommit)
        XCTAssertFalse(source.isRunning)

        try await source.start()
        XCTAssertTrue(source.isRunning)

        source.stop()
        XCTAssertFalse(source.isRunning)
    }

    func testStartIsIdempotent() async throws {
        let source = DemoCaptionSource(script: .quickCommit)
        try await source.start()
        try await source.start()  // should not crash, should remain running
        XCTAssertTrue(source.isRunning)
    }

    func testStopIsSafeWhenNotRunning() {
        let source = DemoCaptionSource()
        source.stop()  // should be a no-op
        XCTAssertFalse(source.isRunning)
    }

    func testPrepareIsCallable() async throws {
        let source = DemoCaptionSource()
        try await source.prepare()  // should not throw for demo
    }

    // MARK: - Commands

    func testManualCommitChangesState() {
        let source = DemoCaptionSource()
        source.manuallyCommit(shabadId: 4242)
        XCTAssertEqual(source.committedShabadId, 4242)
        XCTAssertTrue(source.isCommitted)
    }

    func testResetReturnsToListening() async throws {
        let source = DemoCaptionSource()
        source.manuallyCommit(shabadId: 1)
        XCTAssertTrue(source.isCommitted)
        source.resetShabad()
        XCTAssertFalse(source.isCommitted)
        if case .listening = source.state {
            // ok
        } else {
            XCTFail("Expected .listening after resetShabad(), got \(source.state)")
        }
    }

    // MARK: - Convenience derived state

    func testIsCommittedDerivedCorrectly() {
        let source = DemoCaptionSource()
        XCTAssertFalse(source.isCommitted)
        source.manuallyCommit(shabadId: 7)
        XCTAssertTrue(source.isCommitted)
    }

    func testCommittedShabadIdDerivedCorrectly() {
        let source = DemoCaptionSource()
        XCTAssertNil(source.committedShabadId)
        source.manuallyCommit(shabadId: 9000)
        XCTAssertEqual(source.committedShabadId, 9000)
    }
}
