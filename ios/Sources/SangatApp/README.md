# SangatApp — application library

The bulk of the Sangat iOS app lives in this Swift Package library target.
The executable target (`Sources/GurbaniCaptioningApp/`) is a thin `@main` +
`ContentView` shim that just instantiates `RootView` from here.

Why a library?

- The `@testable import` workflow needs a library — SwiftPM cannot reliably
  test against an executable target.
- A future watch app, Mac Catalyst build, or App Clip can depend on the
  same library without duplicating code.
- The library boundary forces the public/internal access discipline that
  keeps each file from being a kitchen sink.

## Directory map

```
SangatApp/
├── App/             Composition root, routing, modes, feature flags
├── Sources/         CaptionSource protocol + Demo/Live impls + observable adapter
├── DesignSystem/    Theme, tokens, environment, and reusable atoms (Components/)
├── Features/        (M5.2+) Onboarding, Sangat, Sevadar, Settings, Cast
├── Corrections/     CorrectionEvent + CorrectionLog protocol + Noop impl
├── Persistence/     Preferences facade, in-memory session history
├── Platform/        OSLog wrapper, haptics, audio permissions
└── Previews/        SwiftUI preview helpers + static shabad fixtures
```

## Architectural rules (enforced by `ios/.swiftlint.yml`)

| Code | Rule | Enforced |
|------|------|----------|
| A1 | App target depends on `SangatApp`; library never the other way | manual review |
| A2 | No imports between sibling `Features/` folders | custom rule `no_cross_feature_import` |
| A3 | No hex literals in `Features/`; route through `Theme.tokens.colors` | custom rule `no_hex_colors_in_features` |
| A4 | No `.system(size:)` in `Features/`; route through `Theme.tokens.type` | custom rule `no_raw_system_font_in_features` |
| A5 | No `static let shared` singletons (Logger is the one exception) | custom rule `no_singleton_shared` |
| A6 | No `print()` — use `AppLogger.<category>` | custom rule `no_print` |
| A7 | No `try!` / `as!` in production code | custom rules `no_force_try`, `no_force_cast` |
| A8 | All public types have `///` doc comments | manual review |

## Adding a new screen

1. Pick a `Features/<Area>/` folder (creates one if needed).
2. Build the view using only `DesignSystem/Components/` atoms.
3. Bind to `@Environment(AppEnvironment.self)` and read state from
   `env.captionModel`.
4. Mutate state by calling `env.captionModel` methods, never `env.captionSource` directly.
5. Add a `#Preview` block — use `PreviewSupport.env(...)` for an in-memory
   `AppEnvironment`, or `previewTheme(...)` if your view only needs theme tokens.

## Adding a new caption source

Conform to `CaptionSource`, ship a unit-test conforming to the contract in
`CaptionSourceContractTests`, and add a factory case in
`AppEnvironment.makeCaptionSource(flags:)` gated by a feature flag.

## Running tests

```bash
cd ios
swift test --filter SangatAppTests
```

Tests live in `ios/Tests/SangatAppTests/`. Add new tests there, not in
`Tests/GurbaniCaptioningTests/` (that target is for the engine library).
