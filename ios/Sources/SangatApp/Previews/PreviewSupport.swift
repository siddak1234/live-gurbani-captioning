//
//  PreviewSupport.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Helpers for SwiftUI Previews. Builds preview environments and devices
//  without any I/O.

import SwiftUI

#if DEBUG

/// Standard preview environments. Use the `.preview` env when an entire
/// `AppEnvironment` is needed; use `withPreviewTheme(_:)` for atom-only
/// previews that just need theme tokens.
public enum PreviewSupport {

    /// Build a fresh preview env. Default script lands on `.committed`
    /// quickly so previews aren't stuck at "Listening".
    @MainActor
    public static func env(
        theme: Theme = .default,
        mode: AppMode = .sangat,
        script: DemoScript = .quickCommit
    ) -> AppEnvironment {
        AppEnvironment.preview(
            captionSource: DemoCaptionSource(script: script),
            theme: theme,
            mode: mode
        )
    }
}

extension View {

    /// Apply a theme + tokens environment for an atom-level preview without
    /// constructing a full `AppEnvironment`. Use `.preview()` from
    /// `PreviewSupport` when the view depends on services.
    public func previewTheme(_ theme: Theme = .default) -> some View {
        self
            .environment(\.theme, theme)
            .environment(\.themeTokens, theme.tokens)
            .preferredColorScheme(theme.isDark ? .dark : .light)
            .background(theme.tokens.colors.bg)
    }
}

#endif
