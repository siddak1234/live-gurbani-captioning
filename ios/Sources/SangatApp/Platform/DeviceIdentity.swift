//
//  DeviceIdentity.swift
//  GurbaniCaptioningApp · SangatApp · Platform
//
//  Created for the Sangat iOS app, corrections feedback loop Phase 0
//  (Contracts & consent scaffolding). See
//  docs/corrections_feedback_loop_plan.md.
//
//  Anonymous, stable device identifier for attributing + deduping correction
//  uploads. Per locked decision #5 the id is **Keychain-persistent**: it
//  survives app launches and (because iOS Keychain items outlive the app
//  sandbox) app reinstalls, unless the user explicitly resets it via the
//  Phase 5 "Delete my data" path.
//
//  Privacy: this is a random UUID with no link to the user's identity, account,
//  or hardware. It exists only so the server can group a device's corrections
//  and honor a delete request.
//
//  Phase 0 status: defined and unit-tested, but **consumed by nobody yet** —
//  the sync engine (Phase 4) reads `deviceId`. No behavior change today.
//
//  Testability: storage is abstracted behind `IdentityStore` so the
//  generate-once / persist / reuse logic is unit-tested with an in-memory
//  store. Production uses `KeychainIdentityStore`; the host-machine Keychain is
//  not exercised in `swift test` (it can require entitlements there).

import Foundation
#if canImport(Security)
import Security
#endif

/// Minimal persistence seam for a single opaque blob keyed by string. Lets
/// `DeviceIdentity` swap the Keychain for an in-memory store in tests.
public protocol IdentityStore: Sendable {
    func read(key: String) -> Data?
    @discardableResult func write(_ data: Data, key: String) -> Bool
    @discardableResult func delete(key: String) -> Bool
}

/// Stable anonymous device id. Thread-safe (NSLock-guarded cache) rather than
/// actor-isolated, so the background sync coordinator (Phase 4) can read it
/// off the main thread without hopping actors.
public final class DeviceIdentity: @unchecked Sendable {

    /// Keychain account name for the stored id. Stable — do not change without
    /// a migration, or existing devices will re-roll their id.
    public static let storageKey = "com.sangat.app.deviceId"

    private let store: IdentityStore
    private let lock = NSLock()
    private var cached: UUID?

    public init(store: IdentityStore = KeychainIdentityStore()) {
        self.store = store
    }

    /// The device's stable anonymous id. Generated and persisted on first
    /// access; identical on every subsequent access (this launch or future
    /// ones). If persistence fails, returns a valid in-memory id for this
    /// session and logs the failure — the app stays functional.
    public var deviceId: UUID {
        lock.lock(); defer { lock.unlock() }
        if let cached { return cached }

        if let data = store.read(key: Self.storageKey),
           let string = String(data: data, encoding: .utf8),
           let existing = UUID(uuidString: string) {
            cached = existing
            return existing
        }

        let fresh = UUID()
        let ok = store.write(Data(fresh.uuidString.utf8), key: Self.storageKey)
        if !ok {
            AppLogger.sync.error("DeviceIdentity: failed to persist device id; using session-only id")
        }
        cached = fresh
        return fresh
    }

    /// Forget the stored id (Phase 5 "Delete my data"). The next `deviceId`
    /// access mints a fresh one. Returns whether the underlying delete succeeded.
    @discardableResult
    public func reset() -> Bool {
        lock.lock(); defer { lock.unlock() }
        cached = nil
        return store.delete(key: Self.storageKey)
    }
}

// MARK: - Keychain-backed store

/// `IdentityStore` over the iOS/macOS Keychain (`kSecClassGenericPassword`).
public struct KeychainIdentityStore: IdentityStore {

    private let service: String

    public init(service: String = "com.sangat.app.identity") {
        self.service = service
    }

    public func read(key: String) -> Data? {
        #if canImport(Security)
        var query = baseQuery(key: key)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess else { return nil }
        return item as? Data
        #else
        return nil
        #endif
    }

    @discardableResult
    public func write(_ data: Data, key: String) -> Bool {
        #if canImport(Security)
        // Delete-then-add keeps writes idempotent (avoids errSecDuplicateItem).
        SecItemDelete(baseQuery(key: key) as CFDictionary)
        var attributes = baseQuery(key: key)
        attributes[kSecValueData as String] = data
        attributes[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        return SecItemAdd(attributes as CFDictionary, nil) == errSecSuccess
        #else
        return false
        #endif
    }

    @discardableResult
    public func delete(key: String) -> Bool {
        #if canImport(Security)
        let status = SecItemDelete(baseQuery(key: key) as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
        #else
        return false
        #endif
    }

    #if canImport(Security)
    private func baseQuery(key: String) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
        ]
    }
    #endif
}
