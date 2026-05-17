# iOS app milestones

The iOS application (Layer 3 frontend per `docs/architecture.md`, plus the
detail in `docs/ios_app_architecture.md`) ships in six audited milestones.
Each lands as its own branch off `main`, scoped strictly to the `ios/`
directory and a small set of additive doc files; no milestone touches
training-machine territory (`src/`, `scripts/`, `configs/`, `eval_data/`,
`submissions/`, `lora_adapters/`).

The cadence is **sequential gating**: I do not start milestone N+1 until
the audit report for N is reviewed and approved.

## Milestone roster

| # | Name | Scope | Audit doc | Status |
|---|------|-------|-----------|--------|
| M5.1 | Foundations | Theme, tokens, atoms, `CaptionSource` + Demo, `AppEnvironment`, RootView placeholder, Corrections protocol + Noop, test target | `ios/AUDIT-M5.1.md` | **landed** |
| M5.2 | Sangat reading | `Features/Sangat/` — Idle, Listening, Tentative, ReadingHost, Hero, Karaoke, FullShabad | `ios/AUDIT-M5.2.md` | pending |
| M5.3 | Sevadar surfaces | `Features/Sevadar/` — Dock, Picker, Confidence, History; AppMode switching | `ios/AUDIT-M5.3.md` | pending |
| M5.4 | Onboarding + Settings | `Features/Onboarding/` + `Features/Settings/`; mic permission flow | `ios/AUDIT-M5.4.md` | pending |
| M5.5 | Cast / AirPlay | `Features/Cast/`; UIScreen route handling | `ios/AUDIT-M5.5.md` | pending |
| M5.6 | Correction loop surfaces | Touchpoints in Sangat + Sevadar; `CorrectionsSettingsView`; still writes to `NoopCorrectionLog` | `ios/AUDIT-M5.6.md` | pending |

After M5.6 the codebase is feature-complete for everything except the
WhisperKit wire (M5.7, blocked on model export) and real correction
persistence (M5.8, blocked on user-trust decision).

## Architectural invariants — apply to every milestone

These run as part of every milestone's audit. See
`docs/ios_app_architecture.md` § "Architectural invariants" for the full
list.

| # | Check | How verified |
|---|---|---|
| A1 | No file outside `ios/` modified (except permitted `docs/ios_app_*.md`) | `git diff --name-only` |
| A2 | No file in `ios/Sources/GurbaniCaptioning/` modified | same |
| A3 | `GurbaniCaptioning` library never imports `SangatApp` | grep |
| A4 | No import between sibling `Features/` folders | SwiftLint |
| A5 | No hex strings or `Color(red:green:blue:)` in `Features/` | SwiftLint |
| A6 | No `static let shared` singletons (Logger exempted) | SwiftLint |
| A7 | No `print(` | SwiftLint |
| A8 | No `try!` / `as!` outside Tests + Previews | SwiftLint |
| A9 | All public types carry `///` doc comments | review |
| A10 | All view models / state holders `@MainActor` | review |

## Per-milestone audit gates

Audit reports follow this template — see `ios/AUDIT-M5.1.md` for the
canonical example.

```
1. Touch budget          — modified files list
2. Files added           — new files list with totals per layer
3. Invariants            — A1–A10 pass/fail
4. Per-milestone checks  — checks specific to this milestone's deliverable
5. Test summary          — XCTest pass/fail per test target
6. Known gaps            — what's deliberately deferred and to which milestone
7. Sign-off              — your approval marker
```

## What lands in M5.4 (onboarding)

Replaces the M5.1 placeholder Welcome screen with the four-card flow from
design canvas section 03 (`V1Onb01_Welcome` → `V1Onb04_RolePick` in
`assets/v1-paper.jsx`). Settings already shipped in a slice ahead of M5.4
(commit `e101517` — theme + reading layout + reading layers); M5.4 is
focused on first-launch onboarding only.

Files added: 7 source files + 3 test files, all under `ios/`.

```
ios/Sources/SangatApp/Features/Onboarding/
  OnboardingFlow.swift            container, card index state, swipe + button navigation
  OnboardingViewModel.swift       state machine: { welcome, howItWorks, mic, role } + completion
  WelcomeCard.swift               Card 1 — "ਸ੍ਰਵਣ ਕਰੋ" hero + body lines + Begin pill
  HowItWorksCard.swift            Card 2 — 3-step amber/amber/saffron timeline + Back/Continue
  MicPermissionCard.swift         Card 3 — saffronSoft disc + AVAudioApplication.requestRecordPermission
  RolePickCard.swift              Card 4 — Sangat default selected, Sevadar requires-unlock copy

ios/Sources/SangatApp/DesignSystem/Components/
  OnboardingMetaHeader.swift      "Live Gurbani · Sangat" top strip shared by all 4 cards

ios/Tests/SangatAppTests/Features/Onboarding/
  OnboardingFlowTests.swift       card sequencing + back navigation
  OnboardingViewModelTests.swift  state transitions; completion writes Preferences
  MicPermissionCardTests.swift    Allow path + Not-now path both complete the flow
```

Files modified: `ios/Sources/SangatApp/App/RootView.swift` (single-block
swap of `OnboardingPlaceholderView` for `OnboardingFlow()`),
`ios/Sources/SangatApp/App/AppEnvironment.swift` (re-introduces persistence
for `hasCompletedOnboarding` now that onboarding is substantive),
`ios/Tests/SangatAppTests/AppEnvironmentTests.swift` (flip the test back
from session-only to persistent).

Persistence: **fully persistent** — after the user picks a role and taps
Continue, `hasCompletedOnboarding`, `mode`, and `micPermissionAcknowledged`
all land in `Preferences` (UserDefaults). Subsequent cold starts skip
onboarding entirely. Reset by uninstall + reinstall during testing. The
session-only behavior introduced in commit `b63bd38` was a stopgap for the
M5.1 placeholder; M5.4 reverts it deliberately because re-asking for mic
permission every launch is hostile.

Behavior contract:

- Forward navigation: Welcome → HowItWorks → Mic → Role → done.
- Back navigation: HowItWorks/Mic/Role have a Back button per the design;
  Welcome has only Begin.
- Skip mic ("Not now"): advances to Role; `micPermissionAcknowledged`
  records `false`. The "Enable microphone" affordance on IdleView for that
  case is deferred to M5.4.1 / M5.3.
- Role pick: tapping a card selects it; CTA label tracks selection
  ("Continue as Sangat" / "Continue as Sevadar"). On Continue,
  `env.mode = selected` and `env.hasCompletedOnboarding = true`.

Audit gates beyond invariants A1–A10:

| # | Check |
|---|---|
| 4.1 | First launch shows Welcome; `hasCompletedOnboarding` flips on Role Continue |
| 4.2 | Mic card calls `AVAudioApplication.requestRecordPermission` (real path) |
| 4.3 | "Not now" advances and `micPermissionAcknowledged = false` is recorded |
| 4.4 | Settings (theme/layout/layers) still persists across launches (regression) |
| 4.5 | Theme change in Settings is still live, no relaunch (regression) |
| 4.6 | Simulator walkthrough: all 4 cards render in `paper`, `darbar`, `mool` |

Design fidelity gates: 8 visual checks against `assets/v1-paper.jsx` line
ranges 150-345 — see the M5.4 audit doc once written.

Pre-ship grep gauntlet (carried forward from M5.2): no explicit `return`
in `#Preview`/`@ViewBuilder`, no `print(`, no force-unwraps in `Sources/`,
no raw `Color(red:…)`/hex literals in `Features/`, no `Bundle.module` as a
public default arg, no `static let shared`.

Not in M5.4: the IdleView "Enable microphone" affordance for users who
tapped Not now (lands in M5.3 / M5.4.1), localized onboarding strings,
debug-skip gestures. Real Sevadar behavior is M5.3.

## What lands in M5.1 (foundations) — for reference

Files added: **37 source files** in `ios/Sources/SangatApp/` + **6 test
files** in `ios/Tests/SangatAppTests/`. Touch budget: **2 modified files**
(`ios/Package.swift` adds 2 targets; `ios/Sources/GurbaniCaptioningApp/ContentView.swift`
simplified to redirect to `RootView`). Adds **3 new docs**:
`docs/ios_app_architecture.md`, this file, and the app-target README.

Demoable artifact: launch the app, the `DemoCaptionSource` scripts
`listening → tentative → committed`. The state pill and committed line
render in the placeholder root view. Theme switching works (in code; no UI
to toggle yet — that lands in M5.4 Settings).

Not in M5.1: real reading views, real Sevadar surfaces, onboarding,
Settings, cast, correction touchpoints. Those are M5.2–M5.6.

## Branch and PR strategy

For every milestone:

1. Branch off `main` named `ios/m5-<N>-<short-name>`.
2. Develop entirely under `ios/` (plus permitted `docs/ios_app_*.md`).
3. Open PR with the audit report attached (`ios/AUDIT-M5.<N>.md`).
4. CI runs `swift build && swift test --filter SangatAppTests`.
5. After audit approval, squash-merge into `main`.

Branches do not collide with the training pipeline because training files
are entirely outside `ios/`. The single shared file is `Makefile` (root)
which we do not touch.
