//  ContentView.swift
//
//  Thin redirect to the real app root. Implemented in the `SangatApp`
//  library target; this file stays in the executable target only because
//  `@main GurbaniCaptioningApp` (in `GurbaniCaptioningApp.swift`) currently
//  references it by name.
//
//  History
//  -------
//  The original "first-light" version of this file held `CaptionViewModel`
//  + `CaptionEngineBridge` directly. As of M5.1 (Foundations) those types
//  are superseded by the new architecture — `CaptionSource` +
//  `CaptionSourceModel` + `AppEnvironment` (all in `SangatApp`).
//  See `docs/ios_app_architecture.md` and `docs/ios_app_milestones.md`.

import SwiftUI
import SangatApp

struct ContentView: View {
    var body: some View {
        RootView()
    }
}

#Preview {
    ContentView()
}
