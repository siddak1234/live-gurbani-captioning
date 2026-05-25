//
//  SyncCoordinator.swift
//  GurbaniCaptioningApp · SangatApp · Sync
//
//  Created for the Sangat iOS app, corrections feedback loop Phase 4
//  (Sync engine). See docs/corrections_feedback_loop_plan.md.
//
//  The outbox drain. On a foreground/online trigger it pushes pending durable
//  corrections to Supabase and advances their sync status. Resilience without a
//  background URLSession: an interrupted upload simply leaves the record
//  `pending` (or stale-`uploading`, requeued at the next sync), so the next
//  trigger retries — exactly-once is guaranteed server-side by the primary key.
//
//  Consent: nothing uploads unless `uploadOptIn` is on; cellular is skipped when
//  `wifiOnlyUpload` is on. Metadata-only for now — local audio clips are kept
//  (the audio-upload step is a later addition), not deleted on success.

import Foundation

@MainActor
public final class SyncCoordinator {

    private let log: DurableCorrectionLog
    private let uploader: any CorrectionsUploading
    private let reachability: any ReachabilityProviding
    private let preferences: Preferences
    private let deviceId: UUID
    private let batchSize: Int
    private var isSyncing = false

    public init(
        log: DurableCorrectionLog,
        uploader: any CorrectionsUploading,
        reachability: any ReachabilityProviding,
        preferences: Preferences,
        deviceId: UUID,
        batchSize: Int = 25
    ) {
        self.log = log
        self.uploader = uploader
        self.reachability = reachability
        self.preferences = preferences
        self.deviceId = deviceId
        self.batchSize = batchSize
    }

    /// Drain pending corrections to the server. Returns the number uploaded.
    /// Safe to call repeatedly; overlapping calls are coalesced.
    @discardableResult
    public func sync() async -> Int {
        guard !isSyncing else { return 0 }
        guard preferences.uploadOptIn else {
            AppLogger.sync.debug("Sync skipped — upload not opted in")
            return 0
        }
        guard reachability.isOnline else {
            AppLogger.sync.debug("Sync skipped — offline")
            return 0
        }
        if preferences.wifiOnlyUpload && !reachability.isOnWifi {
            AppLogger.sync.debug("Sync skipped — Wi-Fi only and not on Wi-Fi")
            return 0
        }

        isSyncing = true
        defer { isSyncing = false }

        // Reclaim any records stuck `uploading` from a prior interrupted run.
        log.requeueInFlight()

        let pending = log.pending(limit: batchSize)
        guard !pending.isEmpty else { return 0 }
        AppLogger.sync.info("Sync draining \(pending.count, privacy: .public) pending correction(s)")

        var uploaded = 0
        for event in pending {
            log.markStatus(.uploading, forId: event.id)
            let envelope = CorrectionEnvelope(event: event, deviceId: deviceId)
            switch await uploader.upload(envelope) {
            case .success, .alreadyUploaded:
                log.markStatus(.uploaded, forId: event.id)
                uploaded += 1
            case .retryable(let reason):
                AppLogger.sync.info("Correction \(event.id.uuidString, privacy: .public) retryable: \(reason, privacy: .public)")
                log.markStatus(.pending, forId: event.id)        // retry next trigger
            case .permanent(let reason):
                AppLogger.sync.error("Correction \(event.id.uuidString, privacy: .public) permanent failure: \(reason, privacy: .public)")
                log.markStatus(.failed, forId: event.id, error: reason)
            }
        }
        AppLogger.sync.info("Sync uploaded \(uploaded, privacy: .public)/\(pending.count, privacy: .public)")
        return uploaded
    }

    /// Right-to-delete: ask the server to remove every correction for this
    /// device (via the `delete-my-data` Edge Function). Best-effort; the caller
    /// also purges local data. Returns true on a 2xx. Works regardless of
    /// `uploadOptIn` — a user can always delete their data.
    @discardableResult
    public func deleteMyData() async -> Bool {
        guard let url = URL(string: SupabaseConfig.deleteFunctionEndpoint) else { return false }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(SupabaseConfig.publishableKey, forHTTPHeaderField: "apikey")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: ["device_id": deviceId.uuidString])
        } catch {
            return false
        }
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            AppLogger.sync.info("delete-my-data → http \(code, privacy: .public)")
            return (200...299).contains(code)
        } catch {
            AppLogger.sync.error("delete-my-data failed: \(error.localizedDescription, privacy: .public)")
            return false
        }
    }
}
