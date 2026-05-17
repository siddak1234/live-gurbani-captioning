//  Models.swift
//
//  Data types shared across the on-device caption pipeline. Mirror the
//  Python dataclasses in src/engine.py (Segment, PredictionResult) so the
//  iOS app's wire-format matches the Python reference verbatim.

import Foundation

/// A canonical SGGS line — one row from BaniDB. Embedded in the app bundle.
public struct ShabadLine: Codable, Identifiable, Equatable {
    public let lineIdx: Int
    public let verseId: String
    public let banidbGurmukhi: String
    /// Optional English transliteration used by the matcher.
    public let transliterationEnglish: String?

    public init(lineIdx: Int,
                verseId: String,
                banidbGurmukhi: String,
                transliterationEnglish: String? = nil) {
        self.lineIdx = lineIdx
        self.verseId = verseId
        self.banidbGurmukhi = banidbGurmukhi
        self.transliterationEnglish = transliterationEnglish
    }

    public var id: String { "\(verseId)#\(lineIdx)" }

    enum CodingKeys: String, CodingKey {
        case lineIdx = "line_idx"
        case verseId = "verse_id"
        case banidbGurmukhi = "banidb_gurmukhi"
        case transliterationEnglish = "transliteration_english"
    }
}

/// One shabad: a list of lines, indexed by lineIdx.
public struct Shabad: Codable, Identifiable, Equatable {
    public let shabadId: Int
    public let lines: [ShabadLine]

    public var id: Int { shabadId }

    enum CodingKeys: String, CodingKey {
        case shabadId = "shabad_id"
        case lines
    }
}

/// Output of the captioning pipeline — one contiguous time range mapped to a canonical line.
public struct CaptionSegment: Equatable {
    public let start: TimeInterval
    public let end: TimeInterval
    public let shabadId: Int
    public let lineIdx: Int
    public let verseId: String
    public let banidbGurmukhi: String

    public init(start: TimeInterval, end: TimeInterval,
                shabadId: Int, lineIdx: Int,
                verseId: String, banidbGurmukhi: String) {
        self.start = start
        self.end = end
        self.shabadId = shabadId
        self.lineIdx = lineIdx
        self.verseId = verseId
        self.banidbGurmukhi = banidbGurmukhi
    }
}

/// Streaming ASR chunk handed to the state machine — mirrors src.asr.AsrChunk.
public struct AsrChunk: Equatable {
    public let start: TimeInterval
    public let end: TimeInterval
    public let text: String

    public init(start: TimeInterval, end: TimeInterval, text: String) {
        self.start = start
        self.end = end
        self.text = text
    }
}
