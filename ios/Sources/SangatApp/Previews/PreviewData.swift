//
//  PreviewData.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Hardcoded shabad lines used by SwiftUI Previews and unit tests. The
//  *only* place outside `DemoScripts.swift` where shabad text appears as
//  literal strings — production reads from `ShabadCorpus`.

import Foundation
import GurbaniCaptioning

/// Static shabad fixtures available to previews and tests.
public enum PreviewData {

    /// Lines of Tati Vao Na Lagai (Guru Arjan, Bilaaval, Ang 819). The same
    /// shabad used in `DemoScripts.tatiVaoNaLagai` so previews can show
    /// real lines without loading a corpus.
    public static let tatiVaoNaLagaiLines: [PreviewLine] = [
        PreviewLine(
            idx: 0,
            verseId: "40711",
            gurmukhi: "ਤਾਤੀ ਵਾਉ ਨ ਲਗਈ ਪਾਰਬ੍ਰਹਮ ਸਰਣਾਈ ॥",
            translit: "Tātī vā▫o na lag▫ī pārbarahm sarṇā▫ī.",
            english: "The hot wind cannot even touch one who is under the Protection of the Supreme Lord God."
        ),
        PreviewLine(
            idx: 1,
            verseId: "40712",
            gurmukhi: "ਚਉਗਿਰਦ ਹਮਾਰੈ ਰਾਮ ਕਾਰ ਦੁਖੁ ਲਗੈ ਨ ਭਾਈ ॥",
            translit: "Cha▫ugiraḏ hamārai rām kār ḏukẖ lagai na bẖā▫ī.",
            english: "On all four sides I am surrounded by the Lord's Circle of Protection; pain does not afflict me, O Siblings of Destiny."
        ),
        PreviewLine(
            idx: 2,
            verseId: "40713",
            gurmukhi: "ਸਤਿਗੁਰੁ ਪੂਰਾ ਭੇਟਿਆ ਜਿਨਿ ਬਣਤ ਬਣਾਈ ॥",
            translit: "Saṯgur pūrā bẖeti▫ā jin baṇaṯ baṇā▫ī.",
            english: "I have met the Perfect True Guru, who has done this deed."
        ),
        PreviewLine(
            idx: 3,
            verseId: "40714",
            gurmukhi: "ਰਾਮ ਨਾਮੁ ਅਉਖਧੁ ਦੀਆ ਏਕਾ ਲਿਵ ਲਾਈ ॥",
            translit: "Rām nām a▫ukẖaḏẖ ḏī▫ā ekā liv lā▫ī.",
            english: "He has given me the medicine of the Lord's Name, and I enshrine love for the One Lord."
        ),
        PreviewLine(
            idx: 4,
            verseId: "40715",
            gurmukhi: "ਰਾਖਿ ਲੀਏ ਤਿਨਿ ਰਖਨਹਾਰ ਸਭ ਬਿਆਧਿ ਮਿਟਾਈ ॥",
            translit: "Rākẖ lī▫e ṯin rakẖanhār sabẖ bi▫āḏẖ mitā▫ī.",
            english: "The Savior Lord has saved me, and erased all my sickness."
        ),
        PreviewLine(
            idx: 5,
            verseId: "40716",
            gurmukhi: "ਕਹੁ ਨਾਨਕ ਕਿਰਪਾ ਭਈ ਪ੍ਰਭ ਭਏ ਸਹਾਈ ॥੧॥",
            translit: "Kaho Nānak kirpā bẖa▫ī parabẖ bẖa▫e sahā▫ī. ||1||",
            english: "Says Nanak, He has granted His Grace; God has become my Help and Support. ||1||"
        ),
    ]

    /// Look up a preview line by index. Returns `nil` if out of bounds.
    public static func line(forIndex idx: Int) -> PreviewLine? {
        guard idx >= 0, idx < tatiVaoNaLagaiLines.count else { return nil }
        return tatiVaoNaLagaiLines[idx]
    }

    /// Build a library `ShabadLine` from a preview line — useful when a view
    /// wants the same shape as the corpus delivers.
    public static func asShabadLine(_ p: PreviewLine) -> ShabadLine {
        ShabadLine(
            lineIdx: p.idx,
            verseId: p.verseId,
            banidbGurmukhi: p.gurmukhi,
            transliterationEnglish: p.translit
        )
    }
}

/// Preview-only shabad line. Mirrors `ShabadLine` from the library but
/// includes the English meaning (which the corpus may or may not have).
public struct PreviewLine: Identifiable, Equatable, Sendable {
    public let idx: Int
    public let verseId: String
    public let gurmukhi: String
    public let translit: String
    public let english: String

    public var id: String { "\(verseId)#\(idx)" }

    public init(idx: Int, verseId: String, gurmukhi: String, translit: String, english: String) {
        self.idx = idx
        self.verseId = verseId
        self.gurmukhi = gurmukhi
        self.translit = translit
        self.english = english
    }
}
