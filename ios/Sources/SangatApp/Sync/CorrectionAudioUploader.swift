//
//  CorrectionAudioUploader.swift
//  GurbaniCaptioningApp · SangatApp · Sync
//
//  Created for the Sangat iOS app, corrections feedback loop Phase 6a
//  (audio upload). See docs/corrections_feedback_loop_plan.md.
//
//  Uploads a local correction audio clip to the private `correction-audio`
//  Storage bucket via a thin Storage REST call. Uploaded before the metadata
//  row so the row can carry the storage key (anon has no UPDATE grant, so the
//  path can't be set after the fact). Idempotent via `x-upsert`.

import Foundation

/// Uploads one audio clip to Storage. Abstracted for testing with a fake.
public protocol CorrectionAudioUploading: Sendable {
    /// Upload the file at `fileURL` to object key `key` (relative to the bucket).
    func upload(fileURL: URL, toKey key: String) async -> UploadOutcome
}

/// Supabase Storage-backed uploader. Stateless + `Sendable`.
public struct SupabaseStorageUploader: CorrectionAudioUploading {

    private let baseURL: String
    private let apiKey: String
    private let bucket: String
    private let session: URLSession

    public init(
        baseURL: String = SupabaseConfig.url,
        apiKey: String = SupabaseConfig.publishableKey,
        bucket: String = "correction-audio",
        session: URLSession = .shared
    ) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.bucket = bucket
        self.session = session
    }

    public func upload(fileURL: URL, toKey key: String) async -> UploadOutcome {
        guard let url = URL(string: "\(baseURL)/storage/v1/object/\(bucket)/\(key)") else {
            return .permanent("bad storage URL")
        }
        guard let data = try? Data(contentsOf: fileURL) else {
            // Clip missing/unreadable — not retryable; metadata can still go up.
            return .permanent("clip unreadable at \(fileURL.lastPathComponent)")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(apiKey, forHTTPHeaderField: "apikey")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue(contentType(forKey: key), forHTTPHeaderField: "Content-Type")
        // No `x-upsert`: that does an UPDATE, which needs an UPDATE policy — but
        // anon is intentionally insert-only here (privacy). Idempotency on retry
        // instead comes from detecting Storage's duplicate marker below.
        request.httpBody = data

        do {
            let (responseData, response) = try await session.data(for: request)
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            AppLogger.sync.info("Audio upload \(key, privacy: .public) → http \(code, privacy: .public)")
            if (200...299).contains(code) { return .success }
            // Storage returns HTTP 400 with body {"statusCode":"409","error":"Duplicate"}
            // when the object already exists — treat as already-uploaded so a
            // retried record (audio ok, metadata failed) doesn't get stuck.
            if let err = try? JSONDecoder().decode(StorageErrorBody.self, from: responseData),
               err.statusCode == "409" || err.error == "Duplicate" {
                return .alreadyUploaded
            }
            return UploadOutcome.classify(statusCode: code)
        } catch {
            return .retryable(error.localizedDescription)
        }
    }

    private func contentType(forKey key: String) -> String {
        key.hasSuffix(".caf") ? "audio/x-caf" : "audio/mp4"
    }

    /// Supabase Storage error envelope (HTTP status is 400 even for "409 Duplicate").
    private struct StorageErrorBody: Decodable {
        let statusCode: String?
        let error: String?
    }
}
