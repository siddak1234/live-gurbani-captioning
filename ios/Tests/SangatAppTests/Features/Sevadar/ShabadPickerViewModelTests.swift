//
//  ShabadPickerViewModelTests.swift
//  SangatAppTests — M5.3 (Sevadar surfaces)
//
//  Filter + section logic for the manual shabad picker. The view
//  model has no I/O so all tests run in-process.

import XCTest
@testable import SangatApp

@MainActor
final class ShabadPickerViewModelTests: XCTestCase {

    // MARK: - Default catalog

    func testDemoCatalogContainsSevenShabads() {
        XCTAssertEqual(Array<ShabadPickerEntry>.demoCatalog.count, 7)
    }

    func testDemoCatalogIncludesTatiVaoNaLagaiAsFirstEntry() {
        let first = Array<ShabadPickerEntry>.demoCatalog.first
        XCTAssertEqual(first?.shabadId, 1789)
        XCTAssertEqual(first?.firstLineGurmukhi, "ਤਾਤੀ ਵਾਉ ਨ ਲਗਈ")
    }

    // MARK: - Sectioning

    func testNoNowPlayingMeansNilEntry() {
        let vm = ShabadPickerViewModel()
        XCTAssertNil(vm.nowPlayingEntry)
    }

    func testNowPlayingResolvesAgainstCatalog() {
        let vm = ShabadPickerViewModel(nowPlayingShabadId: 1789)
        XCTAssertEqual(vm.nowPlayingEntry?.shabadId, 1789)
    }

    func testUnknownNowPlayingIdResolvesToNil() {
        let vm = ShabadPickerViewModel(nowPlayingShabadId: 99999)
        XCTAssertNil(vm.nowPlayingEntry)
    }

    func testRecentEntriesExcludeNowPlaying() {
        // Recent contains shabad 1789 AND shabad 8 — but 1789 is now-
        // playing so it must be filtered out of the recent section.
        let recents: [ShabadPickerEntry] = [
            ShabadPickerEntry(shabadId: 1789, firstLineGurmukhi: "ਤਾਤੀ", meta: "x"),
            ShabadPickerEntry(shabadId: 8, firstLineGurmukhi: "ਸੋ", meta: "y")
        ]
        let vm = ShabadPickerViewModel(
            nowPlayingShabadId: 1789,
            recents: recents
        )
        XCTAssertEqual(vm.recentEntries.map(\.shabadId), [8])
    }

    func testAllMatchesExcludesBothNowPlayingAndRecents() {
        let recents: [ShabadPickerEntry] = [
            ShabadPickerEntry(shabadId: 8, firstLineGurmukhi: "ਸੋ", meta: "x")
        ]
        let vm = ShabadPickerViewModel(
            nowPlayingShabadId: 1789,
            recents: recents
        )
        let allMatchIds = vm.allMatches.map(\.shabadId)
        XCTAssertFalse(allMatchIds.contains(1789), "now-playing must not appear in all matches")
        XCTAssertFalse(allMatchIds.contains(8), "recent entries must not appear in all matches")
    }

    // MARK: - Search filter

    func testEmptyQueryReturnsEverythingMinusExclusions() {
        let vm = ShabadPickerViewModel()
        XCTAssertEqual(vm.searchQuery, "")
        XCTAssertEqual(vm.allMatches.count, 7)
    }

    func testGurmukhiPrefixSearchFiltersAllMatches() {
        let vm = ShabadPickerViewModel()
        vm.searchQuery = "ਤਾਤੀ"
        XCTAssertEqual(vm.allMatches.count, 1)
        XCTAssertEqual(vm.allMatches.first?.shabadId, 1789)
    }

    func testMidStringSearchAlsoMatchesViaContains() {
        // "ਆਦਮੀ" appears mid-line in "ਹਮ ਆਦਮੀ ਹਾਂ ਇਕ ਦਮੀ" — the
        // contains() fallback must catch it even though it's not a
        // prefix.
        let vm = ShabadPickerViewModel()
        vm.searchQuery = "ਆਦਮੀ"
        let matches = vm.allMatches.map(\.shabadId)
        XCTAssertTrue(matches.contains(660), "should match shabad #660 (Hum Aadmi)")
    }

    func testNonMatchingQueryReturnsEmpty() {
        let vm = ShabadPickerViewModel()
        vm.searchQuery = "zzznotfound"
        XCTAssertTrue(vm.allMatches.isEmpty)
    }

    func testQueryTrimsWhitespace() {
        let vm = ShabadPickerViewModel()
        vm.searchQuery = "  ਤਾਤੀ  "
        XCTAssertEqual(vm.allMatches.count, 1)
    }

    // MARK: - Results count

    func testTotalResultsCountSumsAllSections() {
        let recents: [ShabadPickerEntry] = [
            ShabadPickerEntry(shabadId: 8, firstLineGurmukhi: "ਸੋ", meta: "x"),
            ShabadPickerEntry(shabadId: 3, firstLineGurmukhi: "ਜੋ", meta: "y")
        ]
        let vm = ShabadPickerViewModel(
            nowPlayingShabadId: 1789,
            recents: recents
        )
        // 1 (now playing) + 2 (recents) + 4 (catalog minus three)
        XCTAssertEqual(vm.totalResultsCount, 1 + 2 + 4)
    }

    func testEmptyQueryWithNoExclusionsTotals7() {
        let vm = ShabadPickerViewModel()
        XCTAssertEqual(vm.totalResultsCount, 7)
    }
}
