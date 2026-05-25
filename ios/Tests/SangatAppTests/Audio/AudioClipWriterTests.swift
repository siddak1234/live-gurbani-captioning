//
//  AudioClipWriterTests.swift
//  SangatAppTests — corrections feedback loop Phase 2
//
//  Verifies the trainable-audio primitives on the host: the pure window trim,
//  that an encoded clip is a valid, re-readable audio file within the size
//  budget (the "clip written / plays back / small" gate), and that retention
//  pruning keeps the directory under budget.

import XCTest
import AVFoundation
@testable import SangatApp

final class AudioClipWriterTests: XCTestCase {

    private let sampleRate = 16_000.0

    private func tempDir() -> URL {
        URL.temporaryDirectory.appending(path: "clips-\(UUID().uuidString)", directoryHint: .isDirectory)
    }

    /// A 440 Hz sine of `seconds` duration at 16 kHz mono.
    private func tone(seconds: Double) -> [Float] {
        let n = Int(seconds * sampleRate)
        return (0..<n).map { i in 0.25 * sinf(2 * .pi * 440 * Float(i) / Float(sampleRate)) }
    }

    // MARK: - Pure window trim

    func testTrimReturnsLastNSeconds() throws {
        let writer = try AudioClipWriter(directory: tempDir(), sampleRate: sampleRate)
        let samples = tone(seconds: 60)
        let trimmed = writer.trimLastSeconds(samples, seconds: 30)
        XCTAssertEqual(trimmed.count, 30 * 16_000)
        XCTAssertEqual(trimmed.last, samples.last, "trim keeps the most recent samples")
    }

    func testTrimReturnsAllWhenShorterThanWindow() throws {
        let writer = try AudioClipWriter(directory: tempDir(), sampleRate: sampleRate)
        let samples = tone(seconds: 5)
        XCTAssertEqual(writer.trimLastSeconds(samples, seconds: 30).count, samples.count)
    }

    // MARK: - Encode

    func testWritesAValidSmallClip() throws {
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let writer = try AudioClipWriter(directory: dir, codec: .aac, sampleRate: sampleRate)

        let url = try writer.writeClip(id: UUID(), samples: tone(seconds: 30))

        // Exists and is non-trivial.
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        let size = (try url.resourceValues(forKeys: [.fileSizeKey])).fileSize ?? 0
        XCTAssertGreaterThan(size, 0)
        // 30s of bitrate-controlled AAC must clear the small-clip budget.
        XCTAssertLessThan(size, 150_000, "30s clip should be small; got \(size) bytes")

        // "Plays back": re-open as audio and confirm it decodes to ~30s.
        let readBack = try AVAudioFile(forReading: url)
        let duration = Double(readBack.length) / readBack.fileFormat.sampleRate
        XCTAssertEqual(duration, 30, accuracy: 1.0, "decoded duration should be ~30s")
    }

    func testOpusClipIsAlsoAValidAudioFile() throws {
        // Opus is selectable even though it isn't the small-budget default —
        // confirm it still produces a decodable file.
        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let writer = try AudioClipWriter(directory: dir, codec: .opus, sampleRate: sampleRate)
        let url = try writer.writeClip(id: UUID(), samples: tone(seconds: 5))
        let readBack = try AVAudioFile(forReading: url)
        XCTAssertGreaterThan(readBack.length, 0)
    }

    func testEmptySamplesThrow() throws {
        let writer = try AudioClipWriter(directory: tempDir(), sampleRate: sampleRate)
        XCTAssertThrowsError(try writer.writeClip(id: UUID(), samples: []))
    }

    // MARK: - Retention

    func testPruneKeepsDirectoryUnderBudget() throws {
        // Measure one clip so the budget is sized realistically (clip size
        // depends on the platform encoder, not a guessed constant).
        let probeDir = tempDir()
        defer { try? FileManager.default.removeItem(at: probeDir) }
        let probe = try AudioClipWriter(directory: probeDir, codec: .aac, sampleRate: sampleRate, maxStorageBytes: .max)
        let probeURL = try probe.writeClip(id: UUID(), samples: tone(seconds: 30))
        let clipSize = (try probeURL.resourceValues(forKeys: [.fileSizeKey])).fileSize ?? 0
        XCTAssertGreaterThan(clipSize, 0)

        let dir = tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let budget = Int(2.5 * Double(clipSize)) // room for ~2 clips
        let writer = try AudioClipWriter(directory: dir, codec: .aac, sampleRate: sampleRate, maxStorageBytes: budget)

        for _ in 0..<6 {
            _ = try writer.writeClip(id: UUID(), samples: tone(seconds: 30))
        }
        XCTAssertLessThanOrEqual(writer.totalBytesOnDisk, budget, "prune must hold the byte budget")
        XCTAssertGreaterThanOrEqual(writer.totalBytesOnDisk, clipSize, "the newest clip always survives")
    }
}
