# iOS app architecture

The on-device captioning **engine** is the subject of `docs/architecture.md` —
that document describes the three-layer engine decoupling
(primitives / library / frontends). This document picks up where that one
ends: how the **iOS app** (a Layer 3 frontend) is structured internally.

These two documents are deliberately separate so the engine layers can
evolve without re-architecting the app, and vice versa.

## Goals

1. **Build and demo without the model.** The UI must run end-to-end with no
   `.mlpackage` loaded, no microphone access, and no `WhisperKit` link.
   Otherwise designers, reviewers, and TestFlight users are blocked on the
   training pipeline.
2. **Testability through library boundaries.** App-layer code lives in a
   library target (`SangatApp`) so unit tests can `@testable import` it.
   The executable target is a 6-line shim.
3. **Substitutable seams.** Anything the UI binds to is a protocol with at
   least two implementations (a real one and a test/demo one). The seams
   are: `CaptionSource`, `CorrectionLog`, `SessionHistoryStore`,
   `HapticsService`.
4. **One design system.** No view in `Features/` references a hex color, a
   raw point size, or a hardcoded margin. Everything routes through
   `Theme.tokens`.
5. **No singletons.** Every service is constructor-injected via
   `AppEnvironment`. The only exception is `AppLogger`, which is a
   namespaced wrapper over `os.Logger` (a process-wide system facility).

## Four layers

```
┌─────────────────────────────────────────────────────────┐
│  Layer 4 · Features                                     │
│  Onboarding · Sangat · Sevadar · Cast · Settings        │
│  (M5.2 → M5.5 land these)                               │
└────────────────────────┬────────────────────────────────┘
                         │ binds to
┌────────────────────────▼────────────────────────────────┐
│  Layer 3 · App state                                    │
│  App/{AppEnvironment, AppMode, RootView, FeatureFlags}  │
└────────────────────────┬────────────────────────────────┘
                         │ observes / commands
┌────────────────────────▼────────────────────────────────┐
│  Layer 2 · CaptionSource seam                           │
│  protocol → {Demo, Live, Replay} implementations        │
└────────────────────────┬────────────────────────────────┘
                         │ wraps (when wired)
┌────────────────────────▼────────────────────────────────┐
│  Layer 1 · GurbaniCaptioning library (UNCHANGED)        │
│  CaptionEngine, ShabadStateMachine, FuzzyMatcher        │
└─────────────────────────────────────────────────────────┘
```

### Layer 1 — engine library

Owned by the existing `ios/Sources/GurbaniCaptioning/` target. We do not
touch it. The Layer 2 seam wraps it.

### Layer 2 — caption source seam

`CaptionSource` (protocol) is the single contract the UI binds to. Three
concrete implementations:

| Implementation | Used for | Status |
|---|---|---|
| `LiveCaptionSource` | Production. Wraps `CaptionEngine` + WhisperKit. | Skeleton (M5.1) → wired (M5.7, after model export) |
| `DemoCaptionSource` | Design review, screenshots, App Store demo. Drives the UI from a scripted timeline. | **Complete (M5.1)** |
| `ReplayCaptionSource` | Replays saved sessions for parity tests. | Deferred (post-M5.6) |

The protocol exposes:

- Synchronous reads for current value (`state`, `currentGuess`,
  `runnerUps`) — safe because every conformer is `@MainActor`.
- An `AsyncStream<CaptionSourceEvent>` for change notification.
- Lifecycle: `prepare()`, `start()`, `stop()`.
- Commands: `resetShabad()`, `manuallyCommit(shabadId:)`.

#### Why a protocol + AsyncStream rather than @Observable?

The Observation framework's `@Observable` macro requires a concrete class
declaration; it can't be applied to a protocol. To preserve
substitutability we keep the protocol on values + `AsyncStream` and wrap
the stream in a single `@Observable` adapter — `CaptionSourceModel`. Views
observe the model, never the source directly. This is the standard
"service + observable adapter" pattern.

### Layer 3 — app state

- **`AppEnvironment`** — composition root. Holds every service as `let` and
  every user-facing observable bit as `var`. Constructed once at app launch
  via `AppEnvironment.production()` (or `.preview(...)` for tests/previews).
- **`AppMode`** — `sangat | sevadar`. Drives which surfaces are visible.
- **`RootView`** — switches between onboarding and the reading host.
- **`FeatureFlags`** — compile-time + runtime gates. Today's flags:
  `correctionLoopEnabled`, `castEnabled`, `developerMenuEnabled`,
  `useDemoSource`.

### Layer 4 — features

Each feature folder owns its screens. The strict rules:

- A feature folder **cannot** import another feature folder.
  (Shared code goes in `DesignSystem/Components/` or `App/`.)
- Every view binds to `@Environment(AppEnvironment.self)`; the env holds
  the services.
- Every view's body must use only `DesignSystem/` atoms for shabad text,
  state pills, list rows, etc. No raw `Text` for shabad content; no
  `.system(size:)` for fonts.

## Architectural invariants

These are non-negotiable. SwiftLint enforces what it can; the rest are
manual-review gates. Every PR audit references these codes.

| # | Invariant | Enforcement |
|---|---|---|
| A1 | App target depends on `SangatApp` library; library never depends on app target | review |
| A2 | No imports between sibling `Features/` folders | SwiftLint `no_cross_feature_import` |
| A3 | No hex literals in `Features/` | SwiftLint `no_hex_colors_in_features` |
| A4 | No raw `.system(size:)` fonts in `Features/` | SwiftLint `no_raw_system_font_in_features` |
| A5 | No `static let shared` singletons (Logger exempted) | SwiftLint `no_singleton_shared` |
| A6 | No `print()` — use `AppLogger.<category>` | SwiftLint `no_print` |
| A7 | No `try!` / `as!` in production code | SwiftLint `no_force_try`, `no_force_cast` |
| A8 | Every `public` type has a `///` doc comment | review |
| A9 | All view models / state holders annotated `@MainActor` | review |
| A10 | Theme colors meet WCAG AA against their `bg` | `ThemeTokensTests` |

## Theme system

Three tokens-only directions: `paper`, `darbar`, `mool` (see
`Theme.swift`). Each maps to a `ThemeTokens` value with four sub-bags:

- `colors: ColorPalette` — semantic names (`ink`, `bg`, `accent`, `amber`,
  `sevadar`), never literal names (`saffron`, `cream`).
- `type: TypeScale` — Gurmukhi at 3 sizes + serif + sans + mono.
- `spacing: SpacingScale` — `xs/sm/md/lg/xl/xxl/edge`.
- `radii: RadiusScale` — `xs/sm/md/lg/xl/pill`.

Views read tokens via `@Environment(\.themeTokens)`. The root applies the
theme via `.environment(\.theme, theme)`, which auto-propagates the tokens
through the env key dependency.

## Correction loop (plumbed today, no-op behind the seam)

`CorrectionLog` (protocol) defines `record`, `recent`, `clear`, and
`approximateCount`. M5.1 ships `NoopCorrectionLog` — accepts and discards.
M5.6 introduces the real touchpoints (long-press meta header in committed
reading view, pick-shabad action, line nudge). When the user opts in via
Settings, `NoopCorrectionLog` is swapped for a JSONL-backed implementation —
the change is a single line in `AppEnvironment`.

`CorrectionEvent` (value type) carries everything the off-device fine-tune
pipeline needs: predicted shabad, ground-truth shabad, runner-ups,
recent ASR chunks, engine state, optional audio buffer path, free-text
notes. Mirrors the Python `submissions/<run>/notes.md` convention.

### Privacy invariant

The README's "audio never leaves the phone" guarantee is preserved by:

- `audioBufferPath` is *optional* and only populated when the user opts in
  to audio retention (Settings → Help improve detection, default off).
- The path is always device-local. There is no sync path in M5.1–M5.6.
- The opt-in toggle is the only way to enable retention; the screen
  surfacing it lands in M5.4.

## Build, test, run

```bash
# from ios/ directory
swift build                  # build all targets
swift test                   # run all test targets
swift test --filter SangatAppTests   # app-layer tests only
swift test --filter GurbaniCaptioningTests   # engine library tests only
```

Open `Package.swift` in Xcode to develop with the simulator. The first
build is slow because Xcode compiles the bundled `.mlpackage` for the
device's specific Neural Engine version.

## Where to go from here

See `docs/ios_app_milestones.md` for the M5.1 → M5.6 build-out plan and
audit gates.
