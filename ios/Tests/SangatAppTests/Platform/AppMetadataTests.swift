//
//  AppMetadataTests.swift
//  SangatAppTests — corrections feedback loop Phase 0
//
//  Locks the version-stamping constants and asserts the bundle-derived fields
//  return non-empty strings under the test host.

import XCTest
@testable import SangatApp

final class AppMetadataTests: XCTestCase {

    func testSchemaVersionStartsAtOne() {
        XCTAssertEqual(AppMetadata.schemaVersion, 1)
    }

    func testModelVersionIsTheBundled6BitExport() {
        XCTAssertEqual(AppMetadata.modelVersion, "surt-small-v3-kirtan@6bit")
    }

    func testAppVersionAndBuildAreNonEmpty() {
        // Under the test host these fall back to the "0.0.0" / "0" defaults if
        // the bundle has no Info.plist keys; either way they are never empty.
        XCTAssertFalse(AppMetadata.appVersion.isEmpty)
        XCTAssertFalse(AppMetadata.buildNumber.isEmpty)
    }
}
