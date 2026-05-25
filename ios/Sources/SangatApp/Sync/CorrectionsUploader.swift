//
//  CorrectionsUploader.swift
//  GurbaniCaptioningApp · SangatApp · Sync
//
//  Created for the Sangat iOS app, corrections feedback loop Phase 4
//  (Sync engine). See docs/corrections_feedback_loop_plan.md.
//
//  Uploads a correction envelope to the dedicated Supabase project via a thin
//  PostgREST call (no SDK dependency). Honors the Phase-3-validated contract:
//  `Prefer: return=minimal` (anon has no SELECT grant) and treat HTTP 409 as
//  "already uploaded" (idempotency via the primary key).

import Foundation

/// Result of a single upload attempt, classified for the outbox.
public enum UploadOutcome: Equatable, Sendable {
    /// Stored server-side (HTTP 2xx).
    case success
    /// Row already exists (HTTP 409) — idempotent success.
    case alreadyUploaded
    /// Transient failure (network, 5xx, throttling) — keep pending, retry later.
    case retryable(String)
    /// Non-retryable rejection (validation/auth) — mark failed.
    case permanent(String)

    /// Map an HTTP status code to an outcome.
    public static func classify(statusCode: Int) -> UploadOutcome {
        switch statusCode {
        case 200, 201, 204:      return .success
        case 409:                return .alreadyUploaded
        case 408, 425, 429:      return .retryable("http \(statusCode)")
        case 500...599:          return .retryable("http \(statusCode)")
        case 400...499:          return .permanent("http \(statusCode)")
        default:                 return .retryable("http \(statusCode)")
        }
    }
}

/// Uploads one correction. Abstracted so the coordinator can be tested with a
/// fake (no network).
public protocol CorrectionsUploading: Sendable {
    func upload(_ envelope: CorrectionEnvelope) async -> UploadOutcome
}

/// PostgREST-backed uploader over `URLSession`. Stateless + `Sendable`.
public struct SupabaseRESTUploader: CorrectionsUploading {

    private let endpoint: String
    private let apiKey: String
    private let session: URLSession

    public init(
        endpoint: String = SupabaseConfig.correctionsEndpoint,
        apiKey: String = SupabaseConfig.publishableKey,
        session: URLSession = .shared
    ) {
        self.endpoint = endpoint
        self.apiKey = apiKey
        self.session = session
    }

    public func upload(_ envelope: CorrectionEnvelope) async -> UploadOutcome {
        guard let url = URL(string: endpoint) else { return .permanent("bad endpoint URL") }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(apiKey, forHTTPHeaderField: "apikey")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        // return=minimal: anon has no SELECT grant, so a representation insert
        // would 401. We already hold the row locally.
        request.setValue("return=minimal", forHTTPHeaderField: "Prefer")

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        do {
            request.httpBody = try encoder.encode(CorrectionRowDTO(envelope))
        } catch {
            return .permanent("encode failed: \(error.localizedDescription)")
        }

        do {
            let (_, response) = try await session.data(for: request)
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            let outcome = UploadOutcome.classify(statusCode: code)
            AppLogger.sync.info("Correction upload \(envelope.id.uuidString, privacy: .public) → http \(code, privacy: .public)")
            return outcome
        } catch {
            // Network-level error (offline, timeout) — retry later.
            return .retryable(error.localizedDescription)
        }
    }
}
