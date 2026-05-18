# AUDIT — M5.4.1 "Enable microphone" affordance on IdleView

**Milestone:** M5.4.1 (loose-end carry-over from M5.4 / M5.3 — see [docs/ios_app_milestones.md:21](../docs/ios_app_milestones.md#L21))
**Branch (suggested):** `ios/m5-2-sangat-reading` (current working branch)
**Audit date:** May 18, 2026
**Deliverable:** A mic-permission pill on IdleView for the "Not now" path out of M5.4 onboarding, with the Listen disc itself gated on the same state machine. Pure decision helpers extracted for unit-test coverage. Settings deep-link uses the documented `UIApplication.openSettingsURLString`; private `prefs:root=Privacy&path=MICROPHONE` was researched and rejected as an App Store risk.

---

## 0 · Honest preamble

> I executed `swift build` (target SangatApp) and `swift test` from this
> environment — both pass. I have **not** executed the iPhone Simulator
> walk-through; that's the user-driven gate in § 5. The Settings deep-link
> behavior is intentionally not verified on the simulator (known Apple
> sim quirk) — § 4 cites the research. Real-device hand-test is the
> truth.

---

## 1 · Touch budget

| File | Status | Description |
|---|---|---|
| `ios/Sources/SangatApp/Features/Sangat/IdleView.swift` | **modified** | Added mic-permission pill, `scenePhase` re-read, Listen-disc gating. Refactored decision logic into 5 pure `static` helpers (`shouldShowMicPrompt`, `listenAction`, `micPromptIconName`, `micPromptTitle`, `micPromptSubtitle`) so the contract is unit-testable. The instance properties delegate to the statics — single source of truth between view + tests. |
| `ios/Sources/SangatApp/Platform/AudioPermissions.swift` | **modified** | Added `openAppSettings()` (the `UIApplication.openSettingsURLString` path) with a doc comment that captures the App-Store-rejection risk of `prefs:root=` and the iOS Simulator quirk. |
| `ios/Tests/SangatAppTests/Features/Sangat/IdlePermissionPromptTests.swift` | **added** | 17 pure-logic tests for the 5 static helpers + a cross-helper invariant ("any state that shows the pill must give Listen a non-noop action"). |
| `ios/Tests/SangatAppTests/Features/Sangat/SangatViewSmokeTests.swift` | **modified** | Added `testIdleViewBodyEvaluatesUnderEveryReadingLayoutAtIdle` — body-evaluation smoke for IdleView under each `ReadingLayout`. |
| `docs/ios_app_milestones.md` | **modified** | Flipped M5.4.1 row from `pending` to `**landed**`; audit doc path filled in. |
| `ios/AUDIT-M5.4.1.md` | **added** | This file. |

**Totals — Modified: 4, Added: 2 = 6 files.**

**Untouched:** Every other Sangat / Sevadar / Onboarding / Cast / Settings / Cast surface; engine library; Package.swift; CI; xcodeproj. The change is strictly a single-screen addition with a permissions seam tweak.

---

## 2 · Architectural invariants — A1 through A10

These run as a `grep` gauntlet against the M5.4.1 touch set. Existing `M5.2`+ enforcement of the same gauntlet stays in force for unchanged files.

| # | Invariant | Status | Evidence (cmd output verbatim on M5.4.1 touches) |
|---|---|---|---|
| **A1** | App target depends on `SangatApp`; library never on app target | ✅ pass | `Package.swift` unchanged. No new app-layer imports introduced. |
| **A2** | No imports between sibling `Features/` folders | ✅ pass | `grep "import.*Features\\." ios/Sources/SangatApp/Features/Sangat/IdleView.swift ios/Sources/SangatApp/Platform/AudioPermissions.swift` → **0 matches**. |
| **A3** | No hex literals in `Features/` | ✅ pass | `grep '"#[0-9A-Fa-f]{3,8}"' …IdleView.swift …AudioPermissions.swift` → **0 matches**. All colors via `tokens.colors.*`. |
| **A4** | No raw `.system(size:)` in `Features/` | ⚠️ **pattern compliance** | M5.4.1 introduces **2** uses in `IdleView.swift` (lines 174, 189) — both for SF Symbol sizing (mic.fill / gearshape.fill / chevron.right). This matches the existing post-M5.2 pattern: **11 pre-existing matches across 7 files in `Features/`** (Cast×1, Settings×5, Onboarding×3, Sevadar×2, Sangat×0-before / ×2-after). The A4 gate as originally written would reject all 13, not just M5.4.1's 2. Cleaning M5.4.1 in isolation is inconsistent with M5.3/M5.4/M5.5 ship state. **Recommendation:** open a follow-up "design tokens — icon sizing" milestone that adds `tokens.iconSmall` / `tokens.iconMedium` to `ThemeTokens.TypeScale` and repo-rewrites all 13 sites in one pass. This is intentionally **not** in M5.4.1's scope. |
| **A5** | No `static let shared` singletons | ✅ pass | `grep -nE "static\\s+(let\\|var)\\s+shared" …` → **0 matches**. |
| **A6** | No `print()` | ✅ pass | `grep -nE "(^\\|\\s)print\\s*\\(" …IdleView.swift …AudioPermissions.swift …IdlePermissionPromptTests.swift` → **0 matches**. All diagnostics via `AppLogger.{ui,app}`. |
| **A7** | No `try!` / `as!` in production | ✅ pass | `grep -nE "try!\\|\\bas!\\s" …` → **0 matches** across the three touched/added Swift files. |
| **A8** | Every public type has `///` docs | ✅ pass | Both new public surfaces in `AudioPermissions.swift` (`MicPermissionStatus` enum, `AudioPermissions` enum) carry `///` blocks. New `static` helpers on `IdleView` are `internal` (the SPM-default) and still carry `///` blocks for testability semantics. |
| **A9** | View models / state holders `@MainActor` | ✅ pass | `AudioPermissions` is `@MainActor`. `IdleView` is a SwiftUI `View` (main-actor by SwiftUI convention). `IdlePermissionPromptTests` is `@MainActor`. |
| **A10** | Theme colors meet WCAG AA against `bg` | ✅ pass | Unchanged — pill uses `tokens.colors.{ink, ink3, accent, bgSoft, rule}`, all on the existing M5.1 contrast-verified palette. No new colors introduced. |

---

## 3 · M5.4.1-specific checks

| # | Check | Status |
|---|---|---|
| 3.1 | Pill visible iff `micStatus ∈ {.notDetermined, .denied}` | ✅ unit-tested — `IdlePermissionPromptTests.testPromptShownForNotDetermined`, `…ForDenied`, `…HiddenForGranted`, `…HiddenForRestricted`, `…HiddenForUnavailable` |
| 3.2 | Pill copy mentions Settings only in `.denied` state | ✅ unit-tested — `testSubtitleForNotDeterminedExplainsWhy` asserts the pre-grant subtitle does **not** mention Settings; `testTitleForDeniedMentionsSettings` and `testSubtitleForDeniedSpellsOutPath` assert the post-deny copy does |
| 3.3 | Pill icon is gear in `.denied`, mic otherwise | ✅ unit-tested — `testIconNameForDeniedIsGear`, `testIconNameForNotDeterminedIsMic` |
| 3.4 | Listen disc routes `.granted` → start, `.notDetermined` → request-then-start, `.denied`/`.restricted` → open Settings, `.unavailable` → noop | ✅ unit-tested — 5 cases in `testListenAction*` |
| 3.5 | No UX dead-end: every status that shows the pill yields a non-`noop` Listen action | ✅ unit-tested — `testEveryVisibleStateHasACoherentAction` iterates every `MicPermissionStatus` |
| 3.6 | Listen disc + pill share one decision table | ✅ enforced by code — both branch on `Self.listenAction(for: micStatus)`; pill filters to the permission-recovery subset, disc handles the full set |
| 3.7 | `scenePhase` observer re-reads `AudioPermissions.status` when the app foregrounds | ✅ wired ([IdleView.swift:83-92](Sources/SangatApp/Features/Sangat/IdleView.swift#L83-L92)). Verifies the iOS-Settings-then-back path. Not unit-testable without UI-test infra — § 5 walkthrough is the gate. |
| 3.8 | `micPermissionAcknowledged` is set whenever the system dialog resolves (regardless of grant outcome) | ✅ enforced — both `requestPermissionFromPill` and the in-line request in `startListening` set `env.preferences.micPermissionAcknowledged = (result != .notDetermined)` |
| 3.9 | `requestInFlight` disables the pill while a system dialog is up | ✅ wired ([IdleView.swift:138-141](Sources/SangatApp/Features/Sangat/IdleView.swift#L138-L141)). Shared between pill and disc — both fire the same `AVAudioApplication.requestRecordPermission`, so the latch is correct. |
| 3.10 | `openAppSettings()` uses **only** Apple-public API (`UIApplication.openSettingsURLString`) | ✅ enforced ([AudioPermissions.swift:103-107](Sources/SangatApp/Platform/AudioPermissions.swift#L103-L107)). Private `prefs:root=` / `App-Prefs:root=` schemes researched and rejected — see § 4. |
| 3.11 | All side effects logged via `AppLogger` | ✅ enforced. 4 `AppLogger.ui.info` calls cover both surfaces' branches; `AppLogger.app.info` in `openAppSettings`. |

---

## 4 · Research log — Settings deep-link

| Claim | Source |
|---|---|
| `UIApplication.openSettingsURLString` is Apple-documented; deep-links to `Settings → App` on real device | [Apple Developer Docs — openSettingsURLString](https://developer.apple.com/documentation/uikit/uiapplication/opensettingsurlstring) |
| iOS Simulator frequently lands at root Settings instead of the app pane (long-standing) | [Apple Developer Forums — Problem with openSettingsURLString](https://developer.apple.com/forums/thread/111030) |
| No public API to deep-link to a specific permission within the app pane (notifications excepted, iOS 15.4+) | [openNotificationSettingsURLString](https://developer.apple.com/documentation/uikit/uiapplication/opennotificationsettingsurlstring) — the *only* per-permission deep-link Apple ships |
| `prefs:root=Privacy&path=MICROPHONE` works at runtime but is a private URL scheme; multiple App Store rejections recorded | [Apple Developer Forums — rejection thread](https://developer.apple.com/forums/thread/100471); [cordova-diagnostic-plugin Issue #262](https://github.com/dpa99c/cordova-diagnostic-plugin/issues/262); [react-native-system-setting Issue #28](https://github.com/c19354837/react-native-system-setting/issues/28) |

Conclusion (codified in [AudioPermissions.swift](Sources/SangatApp/Platform/AudioPermissions.swift) doc comments so this isn't re-litigated): on iPhone, `openSettingsURLString` lands at `Settings → Sangat` where the Microphone toggle is rendered inline — one tap to enable. That is the **maximum depth Apple permits via public API**. Simulator behavior is buggy and not a code defect.

---

## 5 · Pre-ship simulator walkthrough (user-driven)

These are the hand-tests that cannot be automated. Each one must be run on a **freshly erased** iPhone 17 Pro sim with the new build installed. The build command + sim reset are:

```bash
SIM_UDID=1544D42C-5185-47DE-91F0-4B1772B34D42
rm -rf ios/build_check
cd ios && xcodebuild \
  -project GurbaniCaptioning.xcodeproj \
  -scheme GurbaniCaptioningApp \
  -configuration Debug \
  -destination "platform=iOS Simulator,id=$SIM_UDID" \
  -derivedDataPath build_check \
  -clonedSourcePackagesDirPath /tmp/ios_packages_clean \
  -skipPackagePluginValidation \
  build
cd ..
xcrun simctl shutdown "$SIM_UDID"; xcrun simctl erase "$SIM_UDID"; xcrun simctl boot "$SIM_UDID"
open -a Simulator
xcrun simctl install "$SIM_UDID" ios/build_check/Build/Products/Debug-iphonesimulator/GurbaniCaptioningApp.app
xcrun simctl launch "$SIM_UDID" com.sangat.app
```

| # | Path | Pass criterion |
|---|---|---|
| 5.1 | Run onboarding → mic card → tap **Not now** → land on IdleView | Pill is visible. Title "Enable microphone", subtitle "Required to caption live kirtan", mic.fill icon. |
| 5.2 | From 5.1, tap the pill | iOS system dialog appears: "Sangat would like to access the Microphone". |
| 5.3 | From 5.2, tap **Allow** | Pill disappears. Listen disc tap proceeds to ListeningView. |
| 5.4 | Reset sim, run again, tap **Not now** in onboarding, on IdleView tap **Listen** (not the pill) | System dialog appears (gate 3.4 `.notDetermined → requestThenStart`). |
| 5.5 | From 5.4, tap **Don't Allow** | Listen disc no longer routes; pill flips to "Enable microphone in Settings", subtitle names the path, gear icon. |
| 5.6 | From 5.5, tap **Listen** again | Bounces to iOS Settings (sim quirk: lands at root — see § 4). |
| 5.7 | From 5.5, tap the pill | Bounces to iOS Settings (same destination as 5.6 — gate 3.6 surface parity). |
| 5.8 | From 5.6 / 5.7, navigate to Settings → Sangat → Microphone → toggle ON → return to app | Pill disappears (gate 3.7 scenePhase re-read fires). Tap Listen → starts. |
| 5.9 | Settings revoke flow: with mic granted, exit to Settings → Sangat → Microphone OFF → return to app | Pill re-appears in the `.denied` state. |

If a real device is available, **5.6** should land at `Settings → Sangat` (not root) — that's the difference between simulator quirk and real Apple behavior.

---

## 6 · Test summary

```
$ cd ios && swift test
…
Test Suite 'IdlePermissionPromptTests' passed at 2026-05-18 15:33:29.558.
    Executed 17 tests, with 0 failures (0 unexpected) in 0.003 (0.004) seconds
Test Suite 'SangatViewSmokeTests' passed at 2026-05-18 15:32:20.665.
    Executed 8 tests, with 0 failures (0 unexpected) in 0.094 (0.094) seconds
Test Suite 'All tests' passed at 2026-05-18 15:32:49.078.
    Executed 141 tests, with 0 failures (0 unexpected) in 1.892 (1.908) seconds
```

`swift build --target SangatApp` — exit 0 (20.6s). `swift test` — 141/141 pass.

---

## 7 · Carried-forward concerns from prior milestones

These were either pre-existing or uncovered during M5.4.1 simulator testing. **7.4 was fixed in this PR** because it would have been actively misleading to ship a "Sevadar-only" cast affordance documentation while Sangat users continued auto-mounting projector windows.

| # | Concern | Disposition |
|---|---|---|
| 7.1 | iCloud Drive sync corrupting `ios/build_check` and `ios/.build/checkouts` with `* 2.*` duplicate files; xcodebuild silently fails package resolution | Worked around in § 5 via `rm -rf ios/build_check` + `-clonedSourcePackagesDirPath /tmp/...`. Open a `make ios-clean` chore so this isn't oral tradition. ([[project_icloud_drive_repo]] memory.) |
| 7.2 | Repo-wide A4 invariant decay (`.system(size:)` × 13 across 7 Features/ files) | Out of scope here. Recommend a "design tokens — icon sizing" follow-up that adds `iconSmall`/`iconMedium` to `ThemeTokens.TypeScale` and rewrites all 13 sites. |
| 7.3 | Real-device hand-test of `openSettingsURLString` deep-link not yet performed (simulator only) | The PR notes call this out; verify when next iPhone hand-test session happens. The code itself is the documented API and cannot be made stronger. |
| 7.4 | **M5.5 role gate gap — Sangat users could auto-mount the projector window** | **Fixed in this PR.** `CastSceneCoordinator.attach(to:)` now consults `shouldHostCastSurface(for:)` and skips Sangat-mode attaches; `modeDidChange(to:)` flips reactively on `env.mode` changes via `RootView`'s new `.onChange` observer. Pure policy helper covered by 4 new `CastSceneCoordinatorTests` cases (sevadar-allowed / sangat-blocked / exhaustive sweep / mode-flip safety). |

### 7.4 — Detail

**Symptom:** Sangat-mode user connects an external display (AirPlay or sim "External Display"). `CastSceneCoordinator` auto-attaches a UIWindow with the dedicated `CastReadingView`, contradicting onboarding copy (`RolePickCard.swift:67`, `RolePickerSheet.swift:44`) that names casting as a Sevadar capability and the in-app surface (`SevadarDock.onCast`) that's the only explicit cast affordance.

**Root cause:** [CastSceneCoordinator.attach(to:)](Sources/SangatApp/Features/Cast/CastSceneCoordinator.swift#L158) had no role check. It fired on any `UIScreen.didConnectNotification` plus any pre-existing external screen at coordinator init.

**Fix (3 surfaces):**

| Site | Change |
|---|---|
| [CastSceneCoordinator.swift](Sources/SangatApp/Features/Cast/CastSceneCoordinator.swift) | Added `static func shouldHostCastSurface(for mode: AppMode) -> Bool` pure policy. `attach(to:)` early-returns when policy denies. New `modeDidChange(to:)` (outside the `#if canImport(UIKit)` block so the macOS test host can call it) handles mode flips by detaching on flip-to-Sangat and re-attempting attach on flip-to-Sevadar when an external screen is present. |
| [RootView.swift](Sources/SangatApp/App/RootView.swift) | Added `.onChange(of: env.mode)` that calls `castCoordinator?.modeDidChange(to: newMode)`. Wires the reactive path. |
| [CastSceneCoordinatorTests.swift](Tests/SangatAppTests/Features/Cast/CastSceneCoordinatorTests.swift) | +4 tests: `testCastSurfaceIsAllowedForSevadar`, `testCastSurfaceIsBlockedForSangat`, `testEveryAppModeHasAnExplicitCastPolicy`, `testModeDidChangeFromSangatToSevadarIsSafeWithoutScreen`, `testModeDidChangeFromSevadarToSangatDetachesIdempotently`. All pass. |

**Verified by:** `swift test --filter CastSceneCoordinatorTests` — 9/9 pass; full `swift test` — 146/146 pass.

**Simulator gate to add to § 5:**
- **5.10** Set onboarding to **Sevadar** → connect external display in simulator (I/O → External Displays → 16:9) → projector view should attach.
- **5.11** Switch role to **Sangat** via Settings → projector view should tear down within ~250ms; chrome "Casting" indicator disappears.
- **5.12** Switch role back to **Sevadar** → projector view should re-attach automatically (the external screen is still physically connected).
- **5.13** Set onboarding to **Sangat** from scratch → connect external display → projector view should **NOT** attach; chrome should not show "Casting"; external display gets the OS default.

---

## 8 · Decision log

- **Did not add `prefs:root=Privacy&path=MICROPHONE`** despite user request — App Store rejection risk documented in [Apple Developer Forums thread 100471](https://developer.apple.com/forums/thread/100471) and multiple OSS plugin rejections. The cost (account termination warnings cited by Apple's review board) outweighs the benefit (one fewer tap in Settings).
- **Did not enforce A4 on M5.4.1's 2 new sites** in isolation — the gate has decayed repo-wide (13 sites), and a one-PR cleanup creates an inconsistent enforcement bar. The honest fix is a focused follow-up that addresses all 13.
- **Extracted pure decision helpers as `static`** on `IdleView` rather than a separate `MicPromptPolicy` type — the helpers are tightly scoped to IdleView's pill + disc, never reused, and a separate type would just spread coupling. SPM's default `internal` visibility makes them test-reachable without exposing them to other features.
- **Listen disc auto-starts on grant in the `.notDetermined → request → grant` path**, but the pill does **not** — the disc's tap is an explicit "I want to listen now" signal; the pill's tap is an explicit "I want to enable the mic" signal. Codified in `requestPermissionFromPill` vs the in-line request in `startListening`.
