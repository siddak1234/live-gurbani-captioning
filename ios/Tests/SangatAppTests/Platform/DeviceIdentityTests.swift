//
//  DeviceIdentityTests.swift
//  SangatAppTests — corrections feedback loop Phase 0
//
//  Verifies the generate-once / persist / reuse / reset logic of
//  `DeviceIdentity` against an in-memory `IdentityStore`. The production
//  Keychain store is not exercised here — host-machine Keychain access under
//  `swift test` can require entitlements; the storage seam exists precisely so
//  the identity logic is testable without it.

import XCTest
@testable import SangatApp

/// In-memory `IdentityStore` for tests — a plain dictionary behind a lock.
private final class InMemoryIdentityStore: IdentityStore, @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String: Data] = [:]

    func read(key: String) -> Data? {
        lock.lock(); defer { lock.unlock() }
        return storage[key]
    }

    @discardableResult
    func write(_ data: Data, key: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        storage[key] = data
        return true
    }

    @discardableResult
    func delete(key: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        storage.removeValue(forKey: key)
        return true
    }
}

final class DeviceIdentityTests: XCTestCase {

    func testGeneratesAndCachesAStableIdWithinAnInstance() {
        let identity = DeviceIdentity(store: InMemoryIdentityStore())
        let first = identity.deviceId
        let second = identity.deviceId
        XCTAssertEqual(first, second, "deviceId must be stable across reads")
    }

    func testPersistsAcrossInstancesSharingAStore() {
        let store = InMemoryIdentityStore()
        let first = DeviceIdentity(store: store).deviceId
        // Simulate a fresh launch: new DeviceIdentity over the same store.
        let second = DeviceIdentity(store: store).deviceId
        XCTAssertEqual(first, second, "deviceId must survive across launches")
    }

    func testWritesGeneratedIdToStore() {
        let store = InMemoryIdentityStore()
        let id = DeviceIdentity(store: store).deviceId
        let raw = store.read(key: DeviceIdentity.storageKey)
        XCTAssertNotNil(raw)
        XCTAssertEqual(raw.flatMap { String(data: $0, encoding: .utf8) }, id.uuidString)
    }

    func testResetClearsStoreAndMintsNewId() {
        let store = InMemoryIdentityStore()
        let identity = DeviceIdentity(store: store)
        let original = identity.deviceId
        XCTAssertTrue(identity.reset())
        XCTAssertNil(store.read(key: DeviceIdentity.storageKey), "reset must clear the store")
        let regenerated = identity.deviceId
        XCTAssertNotEqual(original, regenerated, "a fresh id is minted after reset")
    }

    func testValidIdReturnedEvenWhenPersistenceFails() {
        // A store whose writes always fail: the app should still get a usable id.
        let identity = DeviceIdentity(store: FailingIdentityStore())
        let id = identity.deviceId
        XCTAssertEqual(id, identity.deviceId, "session id is stable even without persistence")
    }
}

/// `IdentityStore` whose writes always fail — exercises the degraded path.
private final class FailingIdentityStore: IdentityStore, @unchecked Sendable {
    func read(key: String) -> Data? { nil }
    @discardableResult func write(_ data: Data, key: String) -> Bool { false }
    @discardableResult func delete(key: String) -> Bool { false }
}
