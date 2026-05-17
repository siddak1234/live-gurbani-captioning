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
