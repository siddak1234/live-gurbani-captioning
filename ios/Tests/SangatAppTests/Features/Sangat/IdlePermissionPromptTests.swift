//
//  IdlePermissionPromptTests.swift
//  SangatAppTests — M5.4.1 (Enable microphone affordance)
//
//  Pure-logic tests for IdleView's permission-driven decisions:
//  pill visibility, pill copy/icon, and the Listen disc's action
//  routing. Each helper is a `static func` on `IdleView`, so these
//  tests need no view environment, no MainActor dance, and no
//  AVFoundation. The view itself dispatches into these helpers, so
//  every assertion here is binding on production behavior.
//
//  Real interaction tests (system-dialog flow, scenePhase re-read,
//  Settings.app deep-link) are simulator/device walkthroughs — see
//  `ios/AUDIT-M5.4.1.md` § 5.

import XCTest
@testable import SangatApp

@MainActor
final class IdlePermissionPromptTests: XCTestCase {

    // MARK: - shouldShowMicPrompt

    func testPromptShownForNotDetermined() {
        XCTAssertTrue(IdleView.shouldShowMicPrompt(for: .notDetermined))
    }

    func testPromptShownForDenied() {
        XCTAssertTrue(IdleView.shouldShowMicPrompt(for: .denied))
    }

    func testPromptHiddenForGranted() {
        XCTAssertFalse(IdleView.shouldShowMicPrompt(for: .granted))
    }

    func testPromptHiddenForRestricted() {
        // Restricted (parental controls): user has no recourse, so
        // showing the pill would just frustrate them. The Listen
        // disc still bounces to Settings for visibility.
        XCTAssertFalse(IdleView.shouldShowMicPrompt(for: .restricted))
    }

    func testPromptHiddenForUnavailable() {
        // macOS test host / no-mic platform.
        XCTAssertFalse(IdleView.shouldShowMicPrompt(for: .unavailable))
    }

    // MARK: - listenAction

    func testListenActionGrantedStartsCapture() {
        XCTAssertEqual(IdleView.listenAction(for: .granted), .startCapture)
    }

    func testListenActionNotDeterminedRequestsThenStarts() {
        XCTAssertEqual(IdleView.listenAction(for: .notDetermined), .requestThenStart)
    }

    func testListenActionDeniedOpensSettings() {
        XCTAssertEqual(IdleView.listenAction(for: .denied), .openSettings)
    }

    func testListenActionRestrictedOpensSettings() {
        XCTAssertEqual(IdleView.listenAction(for: .restricted), .openSettings)
    }

    func testListenActionUnavailableIsNoop() {
        XCTAssertEqual(IdleView.listenAction(for: .unavailable), .noop)
    }

    // MARK: - pill / icon copy by status

    func testIconNameForDeniedIsGear() {
        // Denial routes to Settings — the gear cue signals that.
        XCTAssertEqual(IdleView.micPromptIconName(for: .denied), "gearshape.fill")
    }

    func testIconNameForNotDeterminedIsMic() {
        // Pre-grant prompts in-app; the mic cue signals that.
        XCTAssertEqual(IdleView.micPromptIconName(for: .notDetermined), "mic.fill")
    }

    func testTitleForDeniedMentionsSettings() {
        let title = IdleView.micPromptTitle(for: .denied)
        XCTAssertTrue(title.contains("Settings"),
                      "Denied-state title should signal Settings — got \"\(title)\"")
    }

    func testTitleForNotDeterminedIsEnableCopy() {
        XCTAssertEqual(IdleView.micPromptTitle(for: .notDetermined), "Enable microphone")
    }

    func testSubtitleForDeniedSpellsOutPath() {
        // The denied subtitle is the user's only breadcrumb to the
        // pane that re-grants — assert the path is in the string.
        let subtitle = IdleView.micPromptSubtitle(for: .denied)
        XCTAssertTrue(subtitle.contains("Sangat"),
                      "Denied subtitle should name the app — got \"\(subtitle)\"")
        XCTAssertTrue(subtitle.contains("Microphone"),
                      "Denied subtitle should name the permission — got \"\(subtitle)\"")
    }

    func testSubtitleForNotDeterminedExplainsWhy() {
        let subtitle = IdleView.micPromptSubtitle(for: .notDetermined)
        XCTAssertFalse(subtitle.isEmpty)
        XCTAssertFalse(subtitle.contains("Settings"),
                       "Pre-grant subtitle shouldn't reference Settings — that's the post-deny copy. Got \"\(subtitle)\"")
    }

    // MARK: - cross-helper invariant

    func testEveryVisibleStateHasACoherentAction() {
        // If the pill is visible, the Listen disc must have a non-noop
        // action — otherwise tapping Listen would do nothing while the
        // pill stands there promising recovery.
        let statuses: [MicPermissionStatus] = [
            .notDetermined, .granted, .denied, .restricted, .unavailable
        ]
        for status in statuses where IdleView.shouldShowMicPrompt(for: status) {
            let action = IdleView.listenAction(for: status)
            XCTAssertNotEqual(action, .noop,
                              "Status \(status) shows the pill but Listen is a no-op — UX dead-end.")
        }
    }
}
