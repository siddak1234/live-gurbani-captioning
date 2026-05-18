//
//  ShabadPickerViewModel.swift
//  GurbaniCaptioningApp · SangatApp · Features/Sevadar
//
//  Filter + section logic for `ShabadPickerView`. Held as a `@State`
//  on the view so the search query lives within the sheet's lifetime
//  (cancel → discard).
//
//  The demo catalog mirrors the seven shabads listed in
//  `assets/v1-paper.jsx` (V1ShabadPicker, lines 836-882): Tati Vao Na
//  Lagai, So Dar Keha, Jo Tudh Bhavai, Ek Onkar, Hum Aadmi, Mitr
//  Pyaare, Deh Siva. When the corpus is bundled (M5.7), the picker
//  will source entries from `ShabadCorpus` instead of this static list.

import Foundation
import Observation

/// One row in the picker. Reuses `PickRow`'s `titleGurmukhi + meta`
/// surface so we don't grow a parallel atom.
public struct ShabadPickerEntry: Identifiable, Equatable, Sendable {
    /// Shabad id — drives `manuallyCommit` when the user picks.
    public let shabadId: Int

    /// First-line Gurmukhi shown as the row title.
    public let firstLineGurmukhi: String

    /// Compact "Guru Arjan · Ang 819" meta line.
    public let meta: String

    public var id: Int { shabadId }

    public init(shabadId: Int, firstLineGurmukhi: String, meta: String) {
        self.shabadId = shabadId
        self.firstLineGurmukhi = firstLineGurmukhi
        self.meta = meta
    }
}

@MainActor
@Observable
public final class ShabadPickerViewModel {

    public var searchQuery: String = ""

    /// Currently-playing shabad's id, if any. Drives the "Now playing"
    /// section and the active highlight.
    public let nowPlayingShabadId: Int?

    /// Catalog of all shabads available to pick. M5.3 uses a static
    /// demo list; M5.7 wires this to `ShabadCorpus.allShabadIds`.
    public let catalog: [ShabadPickerEntry]

    /// Recent shabads from `SessionHistoryStore`. Caller passes pre-
    /// mapped entries so the view model doesn't need a store reference.
    public let recents: [ShabadPickerEntry]

    public init(
        nowPlayingShabadId: Int? = nil,
        catalog: [ShabadPickerEntry] = .demoCatalog,
        recents: [ShabadPickerEntry] = []
    ) {
        self.nowPlayingShabadId = nowPlayingShabadId
        self.catalog = catalog
        self.recents = recents
    }

    // MARK: - Derived sections

    /// Entry for the currently-playing shabad, if any, looked up from
    /// the catalog. Shown in its own section above the others.
    public var nowPlayingEntry: ShabadPickerEntry? {
        guard let id = nowPlayingShabadId else { return nil }
        return catalog.first { $0.shabadId == id }
    }

    /// Recent-today entries excluding whatever's currently playing, so
    /// the same row doesn't appear in two sections.
    public var recentEntries: [ShabadPickerEntry] {
        recents.filter { $0.shabadId != nowPlayingShabadId }
    }

    /// "All matches" — the catalog filtered by the search query, with
    /// now-playing and recents removed so each row appears in at most
    /// one section. Empty `searchQuery` matches every catalog entry.
    public var allMatches: [ShabadPickerEntry] {
        let trimmed = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        let recentIds = Set(recents.map(\.shabadId))
        let filtered = catalog.filter { entry in
            entry.shabadId != nowPlayingShabadId &&
            !recentIds.contains(entry.shabadId) &&
            matches(entry, query: trimmed)
        }
        return filtered
    }

    /// Total count for the "X results" caption in the search bar.
    /// Counts everything visible across the three sections.
    public var totalResultsCount: Int {
        var n = nowPlayingEntry == nil ? 0 : 1
        n += recentEntries.count
        n += allMatches.count
        return n
    }

    // MARK: - Filter

    private func matches(_ entry: ShabadPickerEntry, query: String) -> Bool {
        guard !query.isEmpty else { return true }
        // Gurmukhi prefix is the primary match (matches the design's
        // "ਤਾਤੀ ਵਾ" search example); fall back to a contains check so
        // typing mid-line still hits.
        if entry.firstLineGurmukhi.hasPrefix(query) { return true }
        return entry.firstLineGurmukhi.contains(query)
    }
}

// MARK: - Demo catalog

extension Array where Element == ShabadPickerEntry {

    /// Static catalog mirroring V1ShabadPicker's recents + browse arrays
    /// in `assets/v1-paper.jsx`. Replaced by a corpus-backed list at M5.7.
    public static let demoCatalog: [ShabadPickerEntry] = [
        ShabadPickerEntry(
            shabadId: 1789,
            firstLineGurmukhi: "ਤਾਤੀ ਵਾਉ ਨ ਲਗਈ",
            meta: "Guru Arjan · Ang 819"
        ),
        ShabadPickerEntry(
            shabadId: 8,
            firstLineGurmukhi: "ਸੋ ਦਰੁ ਕੇਹਾ ਸੋ ਘਰੁ ਕੇਹਾ",
            meta: "Guru Nanak · Ang 8"
        ),
        ShabadPickerEntry(
            shabadId: 3,
            firstLineGurmukhi: "ਜੋ ਤੁਧੁ ਭਾਵੈ ਸਾਈ ਭਲੀ ਕਾਰ",
            meta: "Guru Nanak · Ang 3"
        ),
        ShabadPickerEntry(
            shabadId: 1,
            firstLineGurmukhi: "ਏਕ ਓਅੰਕਾਰ ਸਤਿਨਾਮੁ",
            meta: "Mool Mantar · Ang 1"
        ),
        ShabadPickerEntry(
            shabadId: 660,
            firstLineGurmukhi: "ਹਮ ਆਦਮੀ ਹਾਂ ਇਕ ਦਮੀ",
            meta: "Guru Nanak · Ang 660"
        ),
        ShabadPickerEntry(
            shabadId: 4900,
            firstLineGurmukhi: "ਮਿਤ੍ਰ ਪਿਆਰੇ ਨੂੰ ਹਾਲ ਮੁਰੀਦਾਂ",
            meta: "Guru Gobind Singh · Dasam"
        ),
        ShabadPickerEntry(
            shabadId: 4901,
            firstLineGurmukhi: "ਦੇਹ ਸਿਵਾ ਬਰੁ ਮੋਹਿ ਇਹੈ",
            meta: "Guru Gobind Singh · Dasam"
        )
    ]

    /// Two-entry demo "recent today" list. Used by the picker preview
    /// and by `SevadarDock` when SessionHistoryStore is empty so the
    /// "Recent today" section isn't blank.
    public static let demoRecents: [ShabadPickerEntry] = [
        ShabadPickerEntry(
            shabadId: 8,
            firstLineGurmukhi: "ਸੋ ਦਰੁ ਕੇਹਾ ਸੋ ਘਰੁ ਕੇਹਾ",
            meta: "Guru Nanak · Ang 8"
        ),
        ShabadPickerEntry(
            shabadId: 3,
            firstLineGurmukhi: "ਜੋ ਤੁਧੁ ਭਾਵੈ ਸਾਈ ਭਲੀ ਕਾਰ",
            meta: "Guru Nanak · Ang 3"
        )
    ]
}
