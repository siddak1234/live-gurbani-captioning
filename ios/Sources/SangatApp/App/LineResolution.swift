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
    /// M5.3 dispatches by `shabadId` so the manual shabad picker
    /// actually changes the reading view — previously the resolve
    /// path ignored shabadId and always returned Tati Vao Na Lagai.
    /// Unknown shabad ids return `nil`; reading views render a "—"
    /// placeholder in that case. M5.7 wires the bundled `ShabadCorpus`.
    public func resolveLine(shabadId: Int, lineIdx: Int) -> ResolvedLine? {
        guard let preview = PreviewData.line(forShabadId: shabadId, lineIdx: lineIdx) else {
            return nil
        }
        return ResolvedLine(
            gurmukhi: preview.gurmukhi,
            transliteration: preview.translit,
            english: preview.english,
            verseId: preview.verseId
        )
    }

    /// Total number of lines in the given shabad. Drives the progress
    /// strip in `HeroLineView` and karaoke next-line guards. Returns
    /// `1` for catalog shabads that have only a single demo line so
    /// reading views don't divide-by-zero on the progress fraction.
    public func totalLines(forShabadId shabadId: Int) -> Int {
        let count = PreviewData.lineCount(forShabadId: shabadId)
        return max(count, 1)
    }

    /// Per-shabad metadata header — raag, ang, author. M5.3 dispatches
    /// through the same per-shabad table as the line resolver so the
    /// reading view's header matches whichever shabad was picked.
    /// Falls back to the Tati Vao defaults for unknown shabads to keep
    /// the header from blanking out.
    public func shabadMeta(forShabadId shabadId: Int) -> ShabadMeta {
        if let meta = PreviewData.meta(forShabadId: shabadId) {
            return ShabadMeta(
                raag: meta.raag,
                ang: meta.ang,
                author: meta.author,
                authorGurmukhi: meta.authorGurmukhi
            )
        }
        return ShabadMeta(
            raag: "—",
            ang: 0,
            author: "Unknown",
            authorGurmukhi: nil
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
