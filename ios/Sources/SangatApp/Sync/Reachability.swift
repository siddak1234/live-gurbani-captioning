//
//  Reachability.swift
//  GurbaniCaptioningApp · SangatApp · Sync
//
//  Created for the Sangat iOS app, corrections feedback loop Phase 4
//  (Sync engine). See docs/corrections_feedback_loop_plan.md.
//
//  Network reachability for the outbox: is there a connection, and is it Wi-Fi
//  (so we can honor the `wifiOnlyUpload` preference and never spend cellular).

import Foundation
import Network

/// Current network state. Abstracted so the coordinator can be tested with a
/// fake that returns fixed values.
public protocol ReachabilityProviding: Sendable {
    var isOnline: Bool { get }
    var isOnWifi: Bool { get }
}

/// `NWPathMonitor`-backed reachability. Starts monitoring at init; thread-safe.
public final class NetworkReachability: ReachabilityProviding, @unchecked Sendable {

    private let monitor = NWPathMonitor()
    private let queue = DispatchQueue(label: "com.sangat.app.reachability")
    private let lock = NSLock()
    private var online = false
    private var wifi = false

    public init() {
        monitor.pathUpdateHandler = { [weak self] path in
            guard let self else { return }
            self.lock.lock()
            self.online = path.status == .satisfied
            self.wifi = path.usesInterfaceType(.wifi)
            self.lock.unlock()
        }
        monitor.start(queue: queue)
    }

    deinit { monitor.cancel() }

    public var isOnline: Bool { lock.lock(); defer { lock.unlock() }; return online }
    public var isOnWifi: Bool { lock.lock(); defer { lock.unlock() }; return wifi }
}
