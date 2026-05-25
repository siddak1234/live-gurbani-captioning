//
//  AudioClipWriter.swift
//  GurbaniCaptioningApp · SangatApp · Audio
//
//  Created for the Sangat iOS app, corrections feedback loop Phase 2
//  (Audio capture). See docs/corrections_feedback_loop_plan.md.
//
//  Encodes a window of microphone samples (16 kHz mono float, snapshotted from
//  the engine at correction time) into a small compressed clip on disk — the
//  trainable acoustic signal for the next surt fine-tune.
//
//  Codec: AAC by default. Decision #4 was Opus, but Apple's AVAudioFile Opus
//  path ignores `AVEncoderBitRateKey` (measured ~85 kbps → ~320 KB / 30 s),
//  while AAC honors it (~32 kbps mono → ~120 KB / 30 s). We default to AAC
//  (m4a) so the small-clip budget is met with real bitrate control; `.opus`
//  stays selectable (valid, just larger) and the server can transcode. A
//  low-bitrate Opus encoder via AVAssetWriter can be revisited later if clip
//  size matters more than simplicity.
//
//  Retention (resolves the §6 open decision): a total-bytes budget
//  (`maxStorageBytes`, default 50 MB ≈ hundreds of clips). After each write the
//  oldest clips are pruned until the directory is under budget, so an offline
//  backlog can never grow without bound.
//
//  Privacy: clips are written only when the user has opted in to audio capture
//  (`Preferences.audioCaptureOptIn`); the caller enforces that gate. Files are
//  device-local; nothing is uploaded until Phase 4 + the separate upload opt-in.

import Foundation
import AVFoundation

public struct AudioClipWriter: Sendable {

    public enum Codec: Sendable {
        case opus
        case aac
    }

    public enum WriterError: Error, LocalizedError {
        case emptySamples
        case formatUnavailable
        case bufferAllocationFailed

        public var errorDescription: String? {
            switch self {
            case .emptySamples:          return "AudioClipWriter: no samples to encode."
            case .formatUnavailable:     return "AudioClipWriter: could not build the PCM format."
            case .bufferAllocationFailed: return "AudioClipWriter: could not allocate the PCM buffer."
            }
        }
    }

    /// Default total on-disk budget for correction clips (50 MB).
    public static let defaultStorageBudgetBytes = 50 * 1024 * 1024

    /// Directory clips are written to (created on init).
    public let directory: URL
    public let codec: Codec
    /// Source sample rate of the snapshotted audio (WhisperKit captures 16 kHz).
    public let sampleRate: Double
    /// Total-bytes retention budget; oldest clips pruned past this after writes.
    public let maxStorageBytes: Int

    public init(
        directory: URL,
        codec: Codec = .aac,
        sampleRate: Double = 16000,
        maxStorageBytes: Int = AudioClipWriter.defaultStorageBudgetBytes
    ) throws {
        self.directory = directory
        self.codec = codec
        self.sampleRate = sampleRate
        self.maxStorageBytes = maxStorageBytes
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    /// `Application Support/CorrectionAudio/` — the default clip location.
    public static func defaultDirectory() throws -> URL {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true
        )
        return base.appendingPathComponent("CorrectionAudio", isDirectory: true)
    }

    public static func makeDefault(codec: Codec = .aac) throws -> AudioClipWriter {
        try AudioClipWriter(directory: try defaultDirectory(), codec: codec)
    }

    /// Container extension for the chosen codec.
    public var fileExtension: String {
        switch codec {
        case .opus: return "caf"   // Opus codec, CAF container
        case .aac:  return "m4a"
        }
    }

    // MARK: - Pure window helper

    /// Last `seconds` of `samples` (all of them if shorter). Pure; unit-tested.
    public func trimLastSeconds(_ samples: [Float], seconds: Double) -> [Float] {
        guard seconds > 0, sampleRate > 0 else { return [] }
        let wanted = Int(seconds * sampleRate)
        guard samples.count > wanted else { return samples }
        return Array(samples.suffix(wanted))
    }

    // MARK: - Encode

    /// Encode `samples` to `<directory>/<id>.<ext>`, prune to budget, return the URL.
    @discardableResult
    public func writeClip(id: UUID, samples: [Float]) throws -> URL {
        guard !samples.isEmpty else { throw WriterError.emptySamples }
        let url = directory.appendingPathComponent("\(id.uuidString).\(fileExtension)")
        try encode(samples: samples, to: url)
        pruneToBudget(maxBytes: maxStorageBytes)
        return url
    }

    private func encode(samples: [Float], to url: URL) throws {
        let settings: [String: Any]
        switch codec {
        case .opus:
            settings = [
                AVFormatIDKey: kAudioFormatOpus,
                AVSampleRateKey: sampleRate,
                AVNumberOfChannelsKey: 1,
                AVEncoderBitRateKey: 24_000,
            ]
        case .aac:
            settings = [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: sampleRate,
                AVNumberOfChannelsKey: 1,
                AVEncoderBitRateKey: 32_000,
            ]
        }

        // AVAudioFile converts our PCM buffer to the compressed file format on
        // write; the buffer must be in the file's processingFormat (PCM).
        let file = try AVAudioFile(forWriting: url, settings: settings)
        let format = file.processingFormat
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            throw WriterError.bufferAllocationFailed
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        guard let channel = buffer.floatChannelData else { throw WriterError.bufferAllocationFailed }
        samples.withUnsafeBufferPointer { src in
            if let base = src.baseAddress {
                channel[0].update(from: base, count: samples.count)
            }
        }
        try file.write(from: buffer)
    }

    // MARK: - Retention / cleanup

    /// Delete oldest clips until total bytes ≤ `maxBytes`.
    public func pruneToBudget(maxBytes: Int) {
        let fm = FileManager.default
        guard let urls = try? fm.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else { return }

        var clips: [(url: URL, size: Int, modified: Date)] = urls.compactMap { url in
            guard let values = try? url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey]),
                  let size = values.fileSize,
                  let modified = values.contentModificationDate else { return nil }
            return (url, size, modified)
        }

        var total = clips.reduce(0) { $0 + $1.size }
        guard total > maxBytes else { return }
        clips.sort { $0.modified < $1.modified } // oldest first
        // Never delete the newest clip — better to slightly exceed the budget
        // than discard the correction we just captured.
        for clip in clips.dropLast() {
            if total <= maxBytes { break }
            if (try? fm.removeItem(at: clip.url)) != nil {
                total -= clip.size
            }
        }
    }

    /// Remove a single clip by path (e.g. after confirmed upload).
    public func deleteClip(atPath path: String) {
        try? FileManager.default.removeItem(atPath: path)
    }

    /// Remove every clip (Phase 5 "Delete my data").
    public func deleteAll() {
        let fm = FileManager.default
        guard let urls = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else { return }
        for url in urls { try? fm.removeItem(at: url) }
    }

    /// Current total bytes used by clips (diagnostics / Settings display).
    public var totalBytesOnDisk: Int {
        let fm = FileManager.default
        guard let urls = try? fm.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]
        ) else { return 0 }
        return urls.reduce(0) { sum, url in
            sum + ((try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
        }
    }
}
