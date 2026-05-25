//
//  CorrectionAudioCaptureTests.swift
//  SangatAppTests — corrections feedback loop Phase 2b
//
//  Tests the consent + capability gating that decides whether a correction
//  gets an audio clip. The real capture (snapshot from a live WhisperKit
//  engine) is device-verified; here a stub `AudioClipCapturing` stands in so
//  the decision logic is covered without an engine.

import XCTest
@testable import SangatApp

@MainActor
final class CorrectionAudioCaptureTests: XCTestCase {

    /// Records whether capture was attempted and returns a canned path.
    private final class StubClipCapturer: AudioClipCapturing {
        var pathToReturn: String?
        private(set) var didCapture = false
        func captureCorrectionClip(id: UUID, seconds: Double, writer: AudioClipWriter) -> String? {
            didCapture = true
            return pathToReturn
        }
    }

    private func tempWriter() throws -> AudioClipWriter {
        try AudioClipWriter(directory: URL.temporaryDirectory.appending(path: "cac-\(UUID().uuidString)"))
    }

    func testCapturesWhenOptedInWithCapturerAndWriter() throws {
        let stub = StubClipCapturer()
        stub.pathToReturn = "/tmp/clip.m4a"
        let path = CorrectionAudioCapture.capture(
            id: UUID(), optedIn: true, capturer: stub, writer: try tempWriter()
        )
        XCTAssertEqual(path, "/tmp/clip.m4a")
        XCTAssertTrue(stub.didCapture)
    }

    func testNilAndNoAttemptWhenOptedOut() throws {
        let stub = StubClipCapturer()
        stub.pathToReturn = "/tmp/clip.m4a"
        let path = CorrectionAudioCapture.capture(
            id: UUID(), optedIn: false, capturer: stub, writer: try tempWriter()
        )
        XCTAssertNil(path)
        XCTAssertFalse(stub.didCapture, "opted-out users must never trigger capture")
    }

    func testNilWhenSourceCannotCapture() throws {
        let path = CorrectionAudioCapture.capture(
            id: UUID(), optedIn: true, capturer: nil, writer: try tempWriter()
        )
        XCTAssertNil(path)
    }

    func testNilWhenNoWriter() {
        let stub = StubClipCapturer()
        stub.pathToReturn = "/tmp/clip.m4a"
        let path = CorrectionAudioCapture.capture(
            id: UUID(), optedIn: true, capturer: stub, writer: nil
        )
        XCTAssertNil(path)
        XCTAssertFalse(stub.didCapture)
    }
}
