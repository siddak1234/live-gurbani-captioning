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

    /// Look up a preview line by index in Tati Vao Na Lagai. Kept for
    /// existing callers that don't know which shabad they're looking
    /// up (the demo source always emits 1789); new callers should use
    /// `line(forShabadId:lineIdx:)` for accurate per-shabad dispatch.
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

    // MARK: - Per-shabad demo dispatch (M5.3)

    /// Minimal demo line content for catalog shabads other than the
    /// fully-detailed Tati Vao Na Lagai (1789). Lets the manual
    /// shabad picker actually change what the reading view shows.
    /// Production reads from `ShabadCorpus` and will retire this.
    public static let demoLinesByShabadId: [Int: [PreviewLine]] = [
        1789: tatiVaoNaLagaiLines,
        1: [
            PreviewLine(
                idx: 0,
                verseId: "mool-1",
                gurmukhi: "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ",
                translit: "Ik Oankār sat nām kartā purakẖ nirbẖa▫o nirvair",
                english: "One Universal Creator God. The Name Is Truth. Creative Being Personified. No Fear. No Hatred."
            ),
            PreviewLine(
                idx: 1,
                verseId: "mool-2",
                gurmukhi: "ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ ॥",
                translit: "Akāl mūraṯ ajūnī saibẖaʼn gur parsāḏ.",
                english: "Image Of The Undying. Beyond Birth. Self-Existent. By Guru's Grace."
            ),
            PreviewLine(
                idx: 2,
                verseId: "mool-3",
                gurmukhi: "॥ ਜਪੁ ॥",
                translit: "|| Jap ||",
                english: "Chant And Meditate."
            ),
            PreviewLine(
                idx: 3,
                verseId: "mool-4",
                gurmukhi: "ਆਦਿ ਸਚੁ ਜੁਗਾਦਿ ਸਚੁ ॥",
                translit: "Āḏ sacẖ jugāḏ sacẖ.",
                english: "True In The Primal Beginning. True Throughout The Ages."
            ),
            PreviewLine(
                idx: 4,
                verseId: "mool-5",
                gurmukhi: "ਹੈ ਭੀ ਸਚੁ ਨਾਨਕ ਹੋਸੀ ਭੀ ਸਚੁ ॥੧॥",
                translit: "Hai bẖī sacẖ Nānak hosī bẖī sacẖ. ||1||",
                english: "True Here And Now. O Nanak, Forever And Ever True. ||1||"
            )
        ],
        8: [
            PreviewLine(
                idx: 0,
                verseId: "sodar-1",
                gurmukhi: "ਸੋ ਦਰੁ ਕੇਹਾ ਸੋ ਘਰੁ ਕੇਹਾ ਜਿਤੁ ਬਹਿ ਸਰਬ ਸਮਾਲੇ ॥",
                translit: "So ḏar kehā so gẖar kehā jiṯ bahi sarab samāle.",
                english: "Where is that Gate, and where is that Dwelling, in which You sit and take care of all?"
            ),
            PreviewLine(
                idx: 1,
                verseId: "sodar-2",
                gurmukhi: "ਵਾਜੇ ਨਾਦ ਅਨੇਕ ਅਸੰਖਾ ਕੇਤੇ ਵਾਵਣਹਾਰੇ ॥",
                translit: "Vāje nāḏ anek asankẖā keṯe vāvaṇhāre.",
                english: "The Sound-current of the Naad vibrates there, and countless musicians play on all sorts of instruments there."
            ),
            PreviewLine(
                idx: 2,
                verseId: "sodar-3",
                gurmukhi: "ਕੇਤੇ ਰਾਗ ਪਰੀ ਸਿਉ ਕਹੀਅਨਿ ਕੇਤੇ ਗਾਵਣਹਾਰੇ ॥",
                translit: "Keṯe rāg parī si▫o kahī▫an keṯe gāvaṇhāre.",
                english: "So many Ragas, so many musicians singing there."
            )
        ],
        3: [
            PreviewLine(
                idx: 0,
                verseId: "jotudh-1",
                gurmukhi: "ਜੋ ਤੁਧੁ ਭਾਵੈ ਸਾਈ ਭਲੀ ਕਾਰ ॥",
                translit: "Jo ṯuḏẖ bẖāvai sā▫ī bẖalī kār.",
                english: "Whatever pleases You is the only good done."
            ),
            PreviewLine(
                idx: 1,
                verseId: "jotudh-2",
                gurmukhi: "ਤੂ ਸਦਾ ਸਲਾਮਤਿ ਨਿਰੰਕਾਰ ॥",
                translit: "Ŧū saḏā salāmaṯ nirankār.",
                english: "You, Eternal and Formless One."
            ),
            PreviewLine(
                idx: 2,
                verseId: "jotudh-3",
                gurmukhi: "ਹੁਕਮੀ ਹੋਵਨਿ ਆਕਾਰ ਹੁਕਮੁ ਨ ਕਹਿਆ ਜਾਈ ॥",
                translit: "Hukmī hovan ākār hukam na kahi▫ā jā▫ī.",
                english: "By His Command, bodies are created; His Command cannot be described."
            )
        ],
        660: [
            PreviewLine(
                idx: 0,
                verseId: "humaadmi-1",
                gurmukhi: "ਹਮ ਆਦਮੀ ਹਾਂ ਇਕ ਦਮੀ ਮੁਹਲਤਿ ਮੁਹਤੁ ਨ ਜਾਣਾ ॥",
                translit: "Ham āḏmī hāʼn ik ḏamī muhlaṯ muhaṯ na jāṇā.",
                english: "We are mortal beings of a single breath; we do not know the appointed time of our departure."
            ),
            PreviewLine(
                idx: 1,
                verseId: "humaadmi-2",
                gurmukhi: "ਨਾਨਕੁ ਬਿਨਵੈ ਤਿਸੈ ਸਰੇਵਹੁ ਜਾ ਕੇ ਜੀਅ ਪਰਾਣਾ ॥",
                translit: "Nānak binvai ṯisai sarevhu jā ke jī▫a parāṇā.",
                english: "Nanak humbly prays — serve the One to whom belong our soul and our breath."
            ),
            PreviewLine(
                idx: 2,
                verseId: "humaadmi-3",
                gurmukhi: "ਅੰਧੇ ਜੀਵਨਾ ਵੀਚਾਰਿ ਦੇਖਿ ਕੇਤੇ ਕੇ ਦਿਨਾ ॥੧॥",
                translit: "Anḏẖe jīvnā vīcẖār ḏekẖ keṯe ke ḏinā. ||1||",
                english: "O blind one, reflect on your life — see, how many days do you have left? ||1||"
            )
        ],
        4900: [
            PreviewLine(
                idx: 0,
                verseId: "mitr-1",
                gurmukhi: "ਮਿਤ੍ਰ ਪਿਆਰੇ ਨੂੰ ਹਾਲ ਮੁਰੀਦਾਂ ਦਾ ਕਹਿਣਾ ॥",
                translit: "Miṯar pi▫āre nūʼn hāl murīḏāʼn ḏā kahiṇā.",
                english: "Tell the state of the disciples to the Beloved Friend."
            ),
            PreviewLine(
                idx: 1,
                verseId: "mitr-2",
                gurmukhi: "ਤੁਧੁ ਬਿਨੁ ਰੋਗੁ ਰਜਾਈਆਂ ਦਾ ਓਢਣ ਨਾਗ ਨਿਵਾਸਾਂ ਦੇ ਰਹਿਣਾ ॥",
                translit: "Ŧuḏẖ bin rog rajā▫ī▫āʼn ḏā odẖaṇ nāg nivāsāʼn ḏe rahiṇā.",
                english: "Without You, soft blankets feel like disease, and living at home is like dwelling among snakes."
            ),
            PreviewLine(
                idx: 2,
                verseId: "mitr-3",
                gurmukhi: "ਸੂਲ ਸੁਰਾਹੀ ਖੰਜਰੁ ਪਿਆਲਾ ਬਿੰਗ ਕਸਾਈਆਂ ਦਾ ਸਹਿਣਾ ॥",
                translit: "Sūl surāhī kẖanjar pi▫ālā bing kasā▫ī▫āʼn ḏā sahiṇā.",
                english: "The flask becomes a spike, the cup a dagger; without You, life is enduring the cuts of a butcher."
            )
        ],
        4901: [
            PreviewLine(
                idx: 0,
                verseId: "deh-1",
                gurmukhi: "ਦੇਹ ਸਿਵਾ ਬਰੁ ਮੋਹਿ ਇਹੈ ਸੁਭ ਕਰਮਨ ਤੇ ਕਬਹੂੰ ਨ ਟਰੋਂ ॥",
                translit: "Ḏeh sivā bar mohi ihai subẖ karman ṯe kabahūʼn na taroʼn.",
                english: "Grant me this boon, O God: that I may never shrink from righteous deeds."
            ),
            PreviewLine(
                idx: 1,
                verseId: "deh-2",
                gurmukhi: "ਨ ਡਰੋਂ ਅਰਿ ਸੋਂ ਜਬ ਜਾਇ ਲਰੋਂ ਨਿਸਚੈ ਕਰਿ ਅਪਨੀ ਜੀਤ ਕਰੋਂ ॥",
                translit: "Na daroʼn ar soʼn jab jā▫e laroʼn niscẖai kar apnī jīṯ karoʼn.",
                english: "May I not fear the foe when I go to fight, and with certainty win my victory."
            ),
            PreviewLine(
                idx: 2,
                verseId: "deh-3",
                gurmukhi: "ਅਰੁ ਸਿਖ ਹੌਂ ਆਪਨੇ ਹੀ ਮਨ ਕੌ ਇਹ ਲਾਲਚ ਹਉ ਗੁਨ ਤਉ ਉਚਰੋਂ ॥",
                translit: "Ar sikẖ haⁿ̃ āpne hī man kau ih lālacẖ ha▫o gun ṯa▫o ucẖroʼn.",
                english: "And teach my own mind only this longing: that I may forever sing Your praises."
            ),
            PreviewLine(
                idx: 3,
                verseId: "deh-4",
                gurmukhi: "ਜਬ ਆਵ ਕੀ ਅਉਧ ਨਿਦਾਨ ਬਨੈ ਅਤਿ ਹੀ ਰਨ ਮੈ ਤਬ ਜੂਝ ਮਰੋਂ ॥੨੩੧॥",
                translit: "Jab āv kī a▫oḏẖ niḏān banai aṯ hī ran mai ṯab jūjẖ maroʼn. ||231||",
                english: "And when the final hour of my life comes, may I die fighting in the field of battle. ||231||"
            )
        ]
    ]

    /// Look up a preview line by shabad id + line index. Used by
    /// `AppEnvironment.resolveLine` so the manual shabad picker
    /// actually changes the reading view.
    public static func line(forShabadId shabadId: Int, lineIdx: Int) -> PreviewLine? {
        guard let lines = demoLinesByShabadId[shabadId],
              lineIdx >= 0,
              lineIdx < lines.count
        else { return nil }
        return lines[lineIdx]
    }

    /// Total line count for the given shabad in the demo data, or 0 if
    /// unknown. Drives the progress strip in `HeroLineView`.
    public static func lineCount(forShabadId shabadId: Int) -> Int {
        demoLinesByShabadId[shabadId]?.count ?? 0
    }

    /// Minimal demo meta per catalog shabad — raag, ang, author. Used
    /// by `AppEnvironment.shabadMeta(forShabadId:)`.
    public static func meta(forShabadId shabadId: Int) -> (raag: String, ang: Int, author: String, authorGurmukhi: String)? {
        switch shabadId {
        case 1789:
            return ("Bilaaval", 819, "Guru Arjan Dev Ji", "ਗੁਰੂ ਅਰਜਨ ਦੇਵ ਜੀ")
        case 1:
            return ("Mool Mantar", 1, "Guru Nanak Dev Ji", "ਗੁਰੂ ਨਾਨਕ ਦੇਵ ਜੀ")
        case 8:
            return ("Aasaa", 8, "Guru Nanak Dev Ji", "ਗੁਰੂ ਨਾਨਕ ਦੇਵ ਜੀ")
        case 3:
            return ("Japji Sahib", 3, "Guru Nanak Dev Ji", "ਗੁਰੂ ਨਾਨਕ ਦੇਵ ਜੀ")
        case 660:
            return ("Dhanaasaree", 660, "Guru Nanak Dev Ji", "ਗੁਰੂ ਨਾਨਕ ਦੇਵ ਜੀ")
        case 4900:
            return ("Dasam Granth", 0, "Guru Gobind Singh Ji", "ਗੁਰੂ ਗੋਬਿੰਦ ਸਿੰਘ ਜੀ")
        case 4901:
            return ("Dasam Granth", 0, "Guru Gobind Singh Ji", "ਗੁਰੂ ਗੋਬਿੰਦ ਸਿੰਘ ਜੀ")
        default:
            return nil
        }
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
