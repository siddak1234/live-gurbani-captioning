# AUDIT — M5.6 Correction loop surfaces (+ M5.6.x carry-forward)

**Milestone:** M5.6 (Correction loop surfaces — see [docs/ios_app_milestones.md:23](../docs/ios_app_milestones.md#L23))
**Branch:** `ios/m5-6-corrections` (off `main` at `0d8ef46`)
**Audit date:** May 18–19, 2026
**Deliverable:** Touchpoints in Sangat (long-press → `WrongShabadSheet`) and Sevadar (picker mismatch, dock nudge, runner-up tap, retroactive history flag) emitting `CorrectionEvent`s through the existing `CorrectionLog` seam. Pure `CorrectionEventBuilder` factory. `CorrectionsSettingsView` exposes the `correctionsOptIn` trust hinge.

**Carry-forward fixes (M5.6.x, May 19) surfaced during the M5.6 walkthrough:**
1. Production log swapped from `NoopCorrectionLog` to `InMemoryCorrectionLog` so Settings count actually moves. Durable cross-launch persistence stays M5.8.
2. `DemoCaptionSource` pause now stops the script clock instead of dropping steps (fixes 3 → 5 jump on resume).
3. `manuallyCommit` arms a synthetic auto-advance task so Resume actually advances lines within a manually-picked shabad (instead of being a dead affordance).

---

## 0 · Honest preamble

> I executed `swift build --target SangatApp` and `swift test` from this
> environment — both pass. The 4 new test suites are exercised by both
> the filtered and the unfiltered run. I have **not** executed the
> simulator walkthrough; that's the user-driven gate in § 5. The
> `softPos` kind is deliberately not wired in M5.6 (timer-driven signal,
> belongs with the live engine) — documented in § 8.

---

## 1 · Touch budget

| File | Status | Description |
|---|---|---|
| `ios/Sources/SangatApp/App/AppEnvironment.swift` | **modified** | Added `sessionId: UUID` (`@Observable`, default-init at construction) + `startNewSession()` mutator. Added optional `correctionLog:` parameter to `preview(...)` so tests can inject a spy without touching production code. |
| `ios/Sources/SangatApp/App/RootView.swift` | **modified** | Sangat long-press → `WrongShabadSheet` host (`.sheet(item:)` driven by `PendingShabad`). Sevadar picker `onPick` records `hardNegPos` when picked id mismatches commit. Dock `nudgeLine` records `lineNudge` when the move is non-zero. SessionStart "Begin" calls `env.startNewSession()`. Added 4 correction-emission helpers funnelled through `recordIfOptedIn`. |
| `ios/Sources/SangatApp/Features/Sangat/ReadingHost.swift` | **modified** | Long-press (0.55s) on the committed reading view (Hero/Karaoke/Full) fires `onRequestWrongShabad(currentShabadId)`. Accessibility action mirrors the gesture so VoiceOver users have the same affordance. |
| `ios/Sources/SangatApp/Features/Sevadar/ConfidenceView.swift` | **modified** | Added optional `onEndorseRunnerUp: ((Int) -> Void)?` init parameter. `CandidateRow` extended with `onSelect:` — wraps the row in a `Button` when non-nil, leaves it display-only when nil (committed row). |
| `ios/Sources/SangatApp/Features/Sevadar/HistoryView.swift` | **modified** | Added optional `onFlagWrongShabad: ((SessionEntry) -> Void)?` init parameter. Each `HistoryRow` exposes a `.contextMenu` "Flag as wrong shabad" item when the closure is provided. |
| `ios/Sources/SangatApp/Features/Sevadar/ShabadPickerView.swift` | **modified** | Added optional `headerTitle: String = "Pick shabad"` and `headerSubtitle: String? = nil` params so M5.6's `WrongShabadSheet` reuses the same picker with reframed copy ("Wrong shabad? / Pick the correct one"). Existing Sevadar call site default-preserved. |
| `ios/Sources/SangatApp/Features/Settings/SettingsView.swift` | **modified** | New `improveDetectionSection` linking to `CorrectionsSettingsView`. `ConfidenceView` sheet wires `onEndorseRunnerUp` → records `runnerUpEndorsed` + manualCommit. `HistoryView` sheet wires `onFlagWrongShabad` → opens a follow-up `WrongShabadSheet` whose `onCorrect` records `retroactive`. |
| `ios/Sources/SangatApp/Features/Corrections/CorrectionEventBuilder.swift` | **added** | Pure factory: 4 `make<Kind>(...)` methods, each accepting deterministic `now:` and `id:` for test reproducibility. `engineStateRaw(_:)` stringifier centralizes the `ShabadState` → log-field mapping. |
| `ios/Sources/SangatApp/Features/Corrections/WrongShabadSheet.swift` | **added** | Sangat-side mistake-recovery surface. Delegates the picker to `ShabadPickerView` with reframed copy; defers commit + emission to the host (`RootView`) so this view stays presentation-only. |
| `ios/Sources/SangatApp/Features/Corrections/CorrectionsSettingsView.swift` | **added** | Trust-hinge surface. Toggle backed by `preferences.correctionsOptIn`. Storage section reads `correctionLog.approximateCount` (always 0 for `NoopCorrectionLog`; honest about it). Clear button. Three explainer rows describing the privacy contract. |
| `ios/Tests/SangatAppTests/Features/Corrections/CorrectionLogSpy.swift` | **added** | `@unchecked Sendable` test double with a `recorded` accessor. Used by the gate and wiring tests. |
| `ios/Tests/SangatAppTests/Features/Corrections/CorrectionEventBuilderTests.swift` | **added** | 11 tests verifying every kind's field mapping, including the privacy invariant ("every M5.6 factory must leave `audioBufferPath` nil") and the session-id stamping invariant. |
| `ios/Tests/SangatAppTests/Features/Corrections/CorrectionGateTests.swift` | **added** | 4 tests exercising the opt-in gate at the level RootView/SettingsView use it: off-by-default, on records, toggle-mid-flight produces only the on emission, `startNewSession` propagates to subsequent emissions' `sessionId`. |
| `ios/Tests/SangatAppTests/Features/Corrections/CorrectionsViewSmokeTests.swift` | **added** | 3 body-evaluation smokes (`UIHostingController` rootView assignment) for `WrongShabadSheet`, `CorrectionsSettingsView` in both opt-in states. |
| `ios/Tests/SangatAppTests/Features/Corrections/SessionIdTests.swift` | **added** | 3 tests on the env-level session contract. |
| `docs/ios_app_milestones.md` | **modified** | M5.6 row flipped from `pending` to `**landed**`; audit doc path filled in. |
| `ios/AUDIT-M5.6.md` | **added** | This file. |

**Totals — Modified: 8 source + 1 doc, Added: 4 source + 4 tests + 1 audit = 18 files.**

---

## 2 · Architectural invariants — A1 through A10

Grep gauntlet against the M5.6 touch set:

| # | Invariant | Status | Evidence |
|---|---|---|---|
| **A1** | App target depends on `SangatApp`; library never on app target | ✅ pass | No package-graph changes. |
| **A2** | No imports between sibling `Features/` folders | ✅ pass | `grep -rn "import .*Features\\." Features/Corrections/` → **0 matches**. `WrongShabadSheet` references `ShabadPickerView` and `ShabadPickerViewModel` via the SangatApp module namespace (both are public), not a cross-feature import. |
| **A3** | No hex literals in `Features/` | ✅ pass | 0 matches in the M5.6 touch set. |
| **A4** | No raw `.system(size:)` in `Features/` | ⚠️ **pattern compliance (same as M5.4.1)** | M5.6 adds 2 new sites in `CorrectionsSettingsView.swift` (lines 89, 132 — SF Symbol sizing for the close button and the explainer-row icon). Matches the established post-M5.2 pattern (13+ sites across 7+ files). Same disposition as AUDIT-M5.4.1 § 2 — repo-wide cleanup belongs in a focused token-extraction milestone, not piecemeal here. |
| **A5** | No `static let shared` singletons | ✅ pass | 0 matches in the M5.6 touch set. |
| **A6** | No `print()` | ✅ pass | 0 matches in `Features/Corrections/` and in the M5.6 modifications to other files (verified via grep). All diagnostics use `AppLogger.{app,corrections}`. |
| **A7** | No `try!` / `as!` in production | ✅ pass | 0 matches in the M5.6 touch set. |
| **A8** | Every public type has `///` docs | ✅ pass | `CorrectionEventBuilder`, `WrongShabadSheet`, `CorrectionsSettingsView`, `PendingShabad`, `PendingRetroactiveFlag` — all carry `///` blocks. The new `static func`s on `CorrectionEventBuilder` are `public` and each has a doc block explaining its kind. |
| **A9** | View models / state holders `@MainActor` | ✅ pass | `AppEnvironment` (existing `@MainActor`) houses the new `sessionId` + `startNewSession()`. `CorrectionLogSpy` is `@unchecked Sendable` with an `NSLock` because the protocol is `Sendable` and the test pattern allows it. All view surfaces are SwiftUI `View`s (main-actor by SwiftUI convention). |
| **A10** | Theme colors meet WCAG AA against `bg` | ✅ pass | Unchanged — `CorrectionsSettingsView` uses `tokens.colors.{ink, ink2, ink3, accent, surface, rule, ruleSoft}` from the M5.1 contrast-verified palette. No new colors introduced. |

---

## 3 · M5.6-specific checks

| # | Check | Status |
|---|---|---|
| 3.1 | Every `CorrectionKind` in the public enum has an emission site OR a documented deferral | ✅ — `hardNegPos` (Sevadar picker + Sangat long-press), `runnerUpEndorsed` (ConfidenceView tap), `lineNudge` (Sevadar dock), `retroactive` (HistoryView flag → follow-up sheet). `softPos` deferred (see § 8). |
| 3.2 | `correctionLog.record(...)` is called from exactly one centralized gate in each call site | ✅ — `grep -rn "correctionLog\\.record"` returns 2 sites, both inside `recordIfOptedIn(...)` (in `RootView` and `SettingsView`). No emission bypasses the opt-in gate. |
| 3.3 | Opt-in is **off by default** | ✅ — `Preferences.correctionsOptIn` is a `Bool` defaulting to `false` (M5.1 contract preserved). Verified by `CorrectionGateTests.testOptInOffDoesNotRecord`. |
| 3.4 | `sessionId` regenerates on Let's Begin transition | ✅ — `RootView.SessionStartView { env.startNewSession(); didStartSession = true }`. Verified by `SessionIdTests.testStartNewSessionRollsTheId` + `CorrectionGateTests.testStartNewSessionChangesStampedSessionIdOnSubsequentEmissions`. |
| 3.5 | `audioBufferPath` is **never** set in M5.6 emissions (privacy boundary) | ✅ — `CorrectionEventBuilderTests.testEveryFactoryClearsAudioBufferPath` covers all 4 kinds. M5.6 ships the opt-in toggle and the schema field; audio capture is M5.8. |
| 3.6 | No `URLSession` / network calls in `Corrections/` | ✅ — `grep -rn "URLSession\\|.dataTask\\|.uploadTask"` over `Features/Corrections/` and `Corrections/` → **0 matches**. Privacy boundary held. |
| 3.7 | Sangat long-press doesn't collide with any reading-view gesture | ✅ — `HeroLineView`, `KaraokeView`, `FullShabadView` have zero pre-existing gestures (verified by grep). Long-press attaches at `ReadingHost`-level with `.contentShape(Rectangle())` so the entire view is hit-testable. |
| 3.8 | Sevadar picker emits only when the picked id mismatches commit | ✅ — `recordSevadarPickerCorrectionIfMismatch` short-circuits on `predictedId != pickedShabadId`. A confirm-the-engine pick is not a correction. |
| 3.9 | Line nudge emits only when the move is non-zero (clamping doesn't fire spurious events) | ✅ — `nudgeLine` checks `guard actualDelta != 0` before recording. |
| 3.10 | Same `ShabadPickerView` atom serves both M5.3 Sevadar manual-pick and M5.6 Sangat wrong-shabad surfaces | ✅ — Header strings are parameterized via optional `headerTitle` / `headerSubtitle`; existing Sevadar call site is unchanged thanks to defaults. `WrongShabadSheet` is a thin wrapper. |
| 3.11 | Retroactive flag is a 2-step UX (flag row → supply correct shabad), not a one-tap mistake | ✅ — `HistoryView` `.contextMenu` → `SettingsView`'s `pendingRetroactiveFlag` state → second `WrongShabadSheet` → emit. Confirmed by reading the wired chain. |
| 3.12 | `softPos` is **not** wired (would require timer infra that belongs with the live engine) | ✅ — Documented deferral in § 8. `CorrectionKind.softPos` exists in the enum (M5.1 contract) but no factory or emission site uses it in M5.6. |
| 3.13 | All side effects logged via `AppLogger` | ✅ — `AppLogger.corrections.debug` when gate blocks; `AppLogger.app.info` on `startNewSession` and on opt-in flip; `AppLogger.app.info` on Clear. |
| 3.14 | Accessibility action mirror for long-press | ✅ — `ReadingHost.committedView` attaches `.accessibilityAction(named: Text("Flag wrong shabad"))` next to the gesture so VoiceOver users have the same path. |

---

## 4 · Test summary

```
$ swift test
…
Test Suite 'CorrectionEventBuilderTests' passed at 2026-05-18 23:06:03
    Executed 11 tests, with 0 failures (0 unexpected)
Test Suite 'CorrectionGateTests' passed at 2026-05-18 23:06:03
    Executed 4 tests, with 0 failures (0 unexpected)
Test Suite 'CorrectionsViewSmokeTests' passed at 2026-05-18 23:06:03
    Executed 3 tests, with 0 failures (0 unexpected)
Test Suite 'SessionIdTests' passed at 2026-05-18 23:06:03
    Executed 3 tests, with 0 failures (0 unexpected)
Test Suite 'All tests' passed
    Executed 166 tests, with 0 failures (0 unexpected) in 2.009 (2.028) seconds
```

`swift build --target SangatApp` — exit 0 (≈20s). Full suite was 146 before M5.6; now 166 (+20 new tests).

---

## 5 · Pre-ship simulator walkthrough (user-driven)

Build with the same iCloud-safe incantation used by M5.4.1 § 5:

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
| 5.1 | Onboarding → Sangat role → Allow mic → Begin → Listen → wait for engine to commit a shabad | Reading view (Hero/Karaoke/Full per Settings) renders normally. |
| 5.2 | Long-press the reading view (~0.55s) | Haptic fires; `WrongShabadSheet` appears with title "Wrong shabad?" and subtitle "Pick the correct one." |
| 5.3 | Pick a different shabad from the picker | Sheet dismisses; reading view updates to the new shabad. With **opt-in OFF** (default), `CorrectionsSettingsView` count stays 0. |
| 5.4 | Settings → Improve detection → toggle "Help improve detection" ON, return | Subtitle on the Settings row updates to "On · corrections saved on this device". |
| 5.5 | Long-press → pick different shabad again | (NoopCorrectionLog: count still 0 because the noop discards; in Xcode console you should see "Noop correction recorded" log line from `AppLogger.corrections.info`.) |
| 5.6 | Switch to Sevadar role in Settings → reading view → Sevadar dock visible at bottom | Dock shows: ◀ ▶ Pause Pick Cast Layers buttons. |
| 5.7 | Tap dock ◀ or ▶ | Line changes; `AppLogger.corrections.info` logs `kind=lineNudge` (assuming opt-in ON). |
| 5.8 | Dock → Pick → choose a different shabad | Commit changes; logs `kind=hardNegPos`. Same-shabad pick: no log. |
| 5.9 | Settings → Sevadar tools → Engine confidence → tap a non-committed candidate row | Reading view re-commits to picked shabad; logs `kind=runnerUpEndorsed`. |
| 5.10 | Settings → Sevadar tools → Session history → long-press a row → "Flag as wrong shabad" | Context menu shows the destructive flag item. Tap it → follow-up `WrongShabadSheet` appears. Pick correct shabad → logs `kind=retroactive`. |
| 5.11 | Settings → Improve detection → toggle OFF → repeat any of 5.7-5.10 | No correction-log line in Xcode console. Other behavior (commit/nudge/etc) unchanged. |
| 5.12 | Settings → Improve detection → Clear corrections | With `NoopCorrectionLog` the button is permanently disabled (count is always 0). The button visually dims and is non-interactive. (Real test fires when M5.8 ships a durable log.) |

---

## 6 · Out-of-scope confirmations

These must **NOT** happen in M5.6 — if any do, the audit fails:

- [ ] **No** `URLSession` import anywhere under `Sources/SangatApp/Corrections/` or `Sources/SangatApp/Features/Corrections/`. Verified by grep (§ 3.6).
- [ ] **No** writes to `event.audioBufferPath` anywhere in the M5.6 builder or callers. Verified by test (`testEveryFactoryClearsAudioBufferPath`) + grep (`grep -rn "audioBufferPath" Sources/SangatApp/` returns only the field-clear-to-nil reads in the builder).
- [ ] **No** `softPos` emission. Verified by grep: `grep -rn "softPos" Sources/SangatApp/` returns only the enum definition (`CorrectionKind.softPos` at `Corrections/CorrectionKind.swift`) — zero callers.
- [ ] Production `CorrectionLog` is still `NoopCorrectionLog`. Verified at [AppEnvironment.swift:135-137](Sources/SangatApp/App/AppEnvironment.swift#L135-L137) (production constructor) and the lone `correctionLog ?? NoopCorrectionLog()` in `preview(...)`.

---

## 7 · Carried-forward concerns

| # | Concern | Disposition |
|---|---|---|
| 7.1 | A4 invariant decay (raw `.system(size:)` for SF Symbol sizing) — M5.6 adds 2 more sites for a total of ~15 across 8 features. M5.6.x adds **0** more sites. | Same as M5.4.1 § 7.2. Open a "design tokens — icon sizing" follow-up. |
| 7.2 | iCloud Drive `* 2.*` derived-data poison | Same workaround documented in M5.4.1 § 5 build script. `rm -rf ios/build_check` + `-clonedSourcePackagesDirPath /tmp/...`. |
| 7.3 | Real-device hand-test of the long-press gesture's reliability (especially on Full layout's scrollable content) | Run before final ship to confirm 0.55s is the right threshold. Adjust if it conflicts with reading-scroll instinct. |
| 7.4 | **Settings → Improve detection count stayed at 0 with `NoopCorrectionLog`** (surfaced May 19 walkthrough) | **Fixed in M5.6.x.** New `InMemoryCorrectionLog` ships as the production + preview default; the noop class stays available as the explicit-discard option for tests. Settings count + Clear button now reflect real activity within a session. Durable cross-launch persistence remains M5.8. |
| 7.5 | **Sevadar pause → resume jumped 3 → 5** (M5.3-era bug surfaced May 19 walkthrough) | **Fixed in M5.6.x.** `DemoCaptionSource.makeScriptedPlaybackTask()` holds each step at a post-sleep `while isPaused { … }` gate. Steps whose timer expires during a pause window are no longer silently dropped. New test `testPauseHoldsScriptedTimelineUntilResume` pins the semantic. |
| 7.6 | **After Pick shabad, Resume did nothing** (surfaced May 19 walkthrough) | **Fixed in M5.6.x.** `manuallyCommit` cancels the scripted task and starts `makeSyntheticPlaybackTask` — a wall-clock loop that advances `currentGuess.lineIdx` by 1 every `syntheticAdvanceInterval` (default 4s), wrapping at `totalLinesProvider(shabadId)`. Provider is injected by `AppEnvironment.makeCaptionSource` from `PreviewData.lineCount(forShabadId:)`. 5 new tests cover the contract. |

### 7.4 / 7.5 / 7.6 — Detail + call-site audit (no-break verification)

**Files changed (M5.6.x):**
- [`ios/Sources/SangatApp/Corrections/InMemoryCorrectionLog.swift`](Sources/SangatApp/Corrections/InMemoryCorrectionLog.swift) — **added**
- [`ios/Sources/SangatApp/App/AppEnvironment.swift`](Sources/SangatApp/App/AppEnvironment.swift) — `production()` injects `InMemoryCorrectionLog()`; `preview()` defaults to it; `makeCaptionSource` passes `totalLinesProvider`.
- [`ios/Sources/SangatApp/Sources/DemoCaptionSource.swift`](Sources/SangatApp/Sources/DemoCaptionSource.swift) — init gains 2 optional params; playback split into `makeScriptedPlaybackTask()` + `makeSyntheticPlaybackTask()`; `manuallyCommit` switches modes.
- [`ios/Tests/SangatAppTests/Features/Corrections/InMemoryCorrectionLogTests.swift`](Tests/SangatAppTests/Features/Corrections/InMemoryCorrectionLogTests.swift) — **added** (7 tests).
- [`ios/Tests/SangatAppTests/Features/Sevadar/DemoCaptionSourceClockTests.swift`](Tests/SangatAppTests/Features/Sevadar/DemoCaptionSourceClockTests.swift) — **added** (6 tests).

**Call-site audit (pre vs. post snapshot):**

| Symbol | Pre | Post | Verdict |
|---|---|---|---|
| `DemoCaptionSource(` total callers | 26 | 33 | +7 from the new `DemoCaptionSourceClockTests`. **All 26 originals unchanged** — both new init params have defaults. |
| `AppEnvironment.preview(` callers | 102 | 102 | No churn — added `correctionLog:` already existed from M5.6. |
| `AppEnvironment.production()` callers | 1 (`RootView.init`) | 1 | Same site; behavior change is observable count, no API change. |
| `correctionLog.record(...)` production call sites | 2 (RootView + SettingsView, both inside `recordIfOptedIn`) | 2 | **Gate unbypassed.** |
| `CorrectionEventBuilder.make<Kind>` emission sites | 5 (3 in RootView + 2 in SettingsView) | 5 | Unchanged. |
| `softPos` production callers | 0 (enum def only) | 0 | Still deferred per § 8. |
| `audioBufferPath` non-nil writes | 0 (4 explicit `nil` writes in builder) | 0 | **Privacy boundary intact.** |
| `URLSession`/network in `Corrections/` or `Sources/` | 0 | 0 | **No network.** |
| `NoopCorrectionLog` class definition | exists | exists | **Not deleted** — still available for explicit-discard test scenarios; remains protocol-conformant. |
| New raw `.system(size:)` in M5.6.x sources | — | 0 | A4 invariant not regressed by these fixes (M5.6 itself still has 2 in `CorrectionsSettingsView`, documented in § 2). |

### Behavior-change verification

| Change | What could have broken | Verified by |
|---|---|---|
| `production()` default log change | Anything reading `approximateCount == 0` as a stable invariant | Audited: 1 production reader (`CorrectionsSettingsView.storageSection`) reads it live on appear + after Clear — works unchanged with non-zero counts. |
| `preview()` default log change | Tests that assert `env.correctionLog.approximateCount == 0` post-emission | Audited: only `CorrectionGateTests` inspects emissions, and it injects `CorrectionLogSpy` explicitly. Other tests don't read the log. |
| `DemoCaptionSource.init` added params | The 26 existing callers | All use `()` or `(script:)` — both compile against the new init with defaulted new params. Confirmed by `swift build` + `swift test` exit-0. |
| `manuallyCommit` task swap | `testManuallyCommitAutoPausesTheEngine`, `testPausePreservesCommittedStateAndGuess`, `testNudgeStillWorksWhilePaused` (existing M5.3 contracts) | All three pass: isPaused immediately true after manualCommit; state/guess preserved; nudge still works because synthetic task starts paused. |
| Pause gate added inside scripted loop | `testPauseSetsIsPausedTrue` and existing M5.3 pause tests | Pass — flag-flip semantics unchanged; what changed is *what the loop does* with the flag. |
| `totalLinesProvider` injection | Anything calling DemoCaptionSource without the provider | `nil` provider → synthetic mode short-circuits without advancing. Test `testNoProviderInjectedMeansSyntheticTaskIsANoOp` pins the safe-no-op behavior. |

### Walkthrough additions (append to § 5)

| # | Path | Pass criterion |
|---|---|---|
| 5.13 | Opt-in ON → long-press → pick different shabad → check Settings → Improve detection | **Saved corrections** count is now ≥ 1 (was always 0 with noop). Repeat → count increments. |
| 5.14 | Opt-in ON → Sevadar dock Pick shabad → choose different → Settings → Improve detection | Count ≥ 1 (this surface emits `hardNegPos` same as long-press). |
| 5.15 | After 5.13/5.14: tap Clear corrections in Settings | Count drops to 0; button becomes disabled until next emission. |
| 5.16 | Sevadar dock: tap Pause mid-script → wait ~2× the average step interval → tap Resume | Reading view should advance to the **immediately next** scripted line, not skip ahead. No 3 → 5 jump. |
| 5.17 | Sevadar dock: Pick shabad → choose Mool Mantar → reading view lands on line 1 → tap Resume | Within ~4 seconds (`syntheticAdvanceInterval` default), reading view advances to line 2. Within ~12 seconds, advances through lines 3, 4, 5 → wraps to 1. |
| 5.18 | During 5.17, mid-advance: tap Pause | Line freezes. Wait another full interval → line still frozen. Tap Resume → advance resumes from the held line. |
| 5.19 | Nudge ◀ / ▶ during synthetic auto-advance | Nudge takes effect immediately; next synthetic tick starts from the nudged position (not the position before nudge). |

---

## 8 · Decision log

- **`softPos` is deferred to M5.7/M5.8.** The signal definition is "user read for ≥60s on a committed shabad without correcting". That requires (a) a per-shabad-commit timer, (b) a notion of "real time spent on the shabad" (pause-aware), and (c) certainty the engine is *actually running* — which only the live engine path provides reliably. Adding it on top of `DemoCaptionSource`'s scripted-time would produce events that don't match production semantics.
- **Sangat surface is `WrongShabadSheet` only.** Sangat users don't see ConfidenceView, the dock, or HistoryView, so the only correction path they encounter is the long-press → pick. This matches the role-gate philosophy from M5.4.1 / M5.5.
- **`audioBufferPath` is opt-in **plus** future capability.** Even with the opt-in ON, M5.6's emissions leave the field `nil` — capturing a rolling audio buffer needs the live engine's mic pipeline (M5.7) and the privacy-decision finalization (M5.8). The current toggle is a contract: "I accept that my corrections may eventually include audio buffers." Honest in both directions.
- **Two separate `recordIfOptedIn` helpers** (one in RootView, one in SettingsView) rather than a single shared one. Same body, different call sites; centralizing it would require either (a) a new `AppEnvironment` method (couples a layer that doesn't need the gate) or (b) a free function (loses the `env` capture). The duplication is two lines; the lock-step audit is `grep -rn "correctionLog\\.record" → 2 sites, both inside `recordIfOptedIn`'.
- **`ShabadPickerView` parameterization over forking a separate sheet.** One picker, one search engine, one row atom. The optional header strings keep the change surgical and the default arguments preserve every existing Sevadar call site.
- **`WrongShabadSheet` re-uses the picker rather than building its own.** Same atoms, single point of UX truth for "pick a shabad". The Sangat-side framing change is purely typographic.
