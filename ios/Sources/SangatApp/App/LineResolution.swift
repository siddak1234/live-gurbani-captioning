//
//  LineResolution.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.2 (Sangat reading).
//
//  Bridge between a `LineGuess` (which carries just shabadId + lineIdx) and
//  the renderable display fields a reading view needs (Gurmukhi text,
//  transliteration, English meaning). Built as an extension on
//  `AppEnvironment` so views call `env.resolveLine(...)` consistently,
//  independent of whether the engine is the demo source or the live one.
//
//  Two backing data paths
//  ----------------------
//  1. **Demo path** (M5.1, M5.2): the demo `CaptionSource` produces guesses
//     for shabad 1789 (Tati Vao Na Lagai). Lookup falls through to
//     `PreviewData.line(forIndex:)` which has the full Gurmukhi + translit
//     + English meaning for those six lines.
//  2. **Live path** (M5.7+, when the corpus is bundled): we'll extend this
//     to consult `ShabadCorpus.line(shabadId:lineIdx:)` first. The corpus
//     today stores Gurmukhi + transliteration but no English; we'll need
//     a separate meaning store for that layer.
//
//  This is the single seam where presentation looks up canonical line
//  content. No reading view should ever construct shabad text directly.

import Foundation
import GurbaniCaptioning

/// Display-ready line content. The reading views render these three fields
/// (gating translit/english by user preference) and never touch
/// `ShabadLine` or `PreviewLine` directly.
public struct ResolvedLine: Sendable, Equatable {

    /// Canonical Gurmukhi text. Always present.
    public let gurmukhi: String

    /// Romanized transliteration. Present for demo data; the corpus stores
    /// it when available.
    public let transliteration: String?

    /// English meaning. Present for demo data; not yet bundled with the
    /// production corpus (waiting on the meanings pipeline).
    public let english: String?

    /// Stable BaniDB verse id, when known. Useful for analytics and
    /// correction-event provenance.
    public let verseId: String?

    public init(
        gurmukhi: String,
        transliteration: String? = nil,
        english: String? = nil,
        verseId: String? = nil
    ) {
        self.gurmukhi = gurmukhi
        self.transliteration = transliteration
        self.english = english
        self.verseId = verseId
    }
}

extension AppEnvironment {

    /// Resolve a canonical SGGS line for display.
    ///
    /// In M5.2 with the demo source, this always falls through to
    /// `PreviewData` (which mirrors the demo script's shabad). When the
    /// live source is wired in M5.7, the corpus lookup path is added in
    /// front.
    public func resolveLine(shabadId: Int, lineIdx: Int) -> ResolvedLine? {
        // Demo-source path: the only shabad demo produces is
        // `PreviewData.tatiVaoNaLagaiLines`. Lookup by lineIdx works for
        // any shabadId the demo emits.
        if let preview = PreviewData.line(forIndex: lineIdx) {
            return ResolvedLine(
                gurmukhi: preview.gurmukhi,
                transliteration: preview.translit,
                english: preview.english,
                verseId: preview.verseId
            )
        }
        return nil
    }

    /// Total number of lines in the given shabad. Used by reading views to
    /// compute progress strips and karaoke next-line guards.
    ///
    /// M5.2 returns the demo shabad's length (6). M5.7 will look up via
    /// the bundled corpus.
    public func totalLines(forShabadId shabadId: Int) -> Int {
        PreviewData.tatiVaoNaLagaiLines.count
    }

    /// Best-effort metadata header for the shabad currently on screen.
    /// M5.7 will populate from corpus metadata; today this is the demo
    /// shabad's known meta.
    public func shabadMeta(forShabadId shabadId: Int) -> ShabadMeta {
        ShabadMeta(
            raag: "Bilaaval",
            ang: 819,
            author: "Guru Arjan Dev Ji",
            authorGurmukhi: "ਗੁਰੂ ਅਰਜਨ ਦੇਵ ਜੀ"
        )
    }
}

/// Display metadata about a shabad.
public struct ShabadMeta: Sendable, Equatable {
    public let raag: String
    public let ang: Int
    public let author: String?
    public let authorGurmukhi: String?

    public init(raag: String, ang: Int, author: String? = nil, authorGurmukhi: String? = nil) {
        self.raag = raag
        self.ang = ang
        self.author = author
        self.authorGurmukhi = authorGurmukhi
    }
}
