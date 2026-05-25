//
//  UploadOutcomeTests.swift
//  SangatAppTests — corrections feedback loop Phase 4
//
//  Locks the HTTP-status → outcome classification the outbox relies on.

import XCTest
@testable import SangatApp

final class UploadOutcomeTests: XCTestCase {

    func testSuccessCodes() {
        XCTAssertEqual(UploadOutcome.classify(statusCode: 201), .success)
        XCTAssertEqual(UploadOutcome.classify(statusCode: 200), .success)
        XCTAssertEqual(UploadOutcome.classify(statusCode: 204), .success)
    }

    func testConflictIsAlreadyUploaded() {
        XCTAssertEqual(UploadOutcome.classify(statusCode: 409), .alreadyUploaded)
    }

    func testServerAndThrottleCodesAreRetryable() {
        for code in [408, 425, 429, 500, 502, 503] {
            if case .retryable = UploadOutcome.classify(statusCode: code) { continue }
            XCTFail("expected \(code) to be retryable")
        }
    }

    func testClientErrorsArePermanent() {
        for code in [400, 401, 403, 422] {
            if case .permanent = UploadOutcome.classify(statusCode: code) { continue }
            XCTFail("expected \(code) to be permanent")
        }
    }
}
