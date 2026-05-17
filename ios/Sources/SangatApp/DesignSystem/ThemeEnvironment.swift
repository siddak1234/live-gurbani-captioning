//
//  ThemeEnvironment.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  SwiftUI Environment plumbing so any view can read the active theme
//  without a global. Views read tokens via:
//
//      @Environment(\.themeTokens) private var t
//      Text("…").foregroundStyle(t.colors.ink)
//
//  The root of the app injects the theme exactly once via
//  `.appTheme(themeBinding)` (see `RootView`); a binding rather than a
//  value so theme changes in Settings update the entire tree live.

import SwiftUI

// MARK: - Environment keys

private struct ThemeKey: EnvironmentKey {
    static let defaultValue: Theme = .default
}

private struct ThemeTokensKey: EnvironmentKey {
    static let defaultValue: ThemeTokens = Theme.default.tokens
}

extension EnvironmentValues {

    /// The active `Theme` enum case. Use this if you need to branch on
    /// `theme.isDark` or read `theme.displayName`; read `themeTokens`
    /// to get the concrete color/font tokens.
    public var theme: Theme {
        get { self[ThemeKey.self] }
        set {
            self[ThemeKey.self] = newValue
            self[ThemeTokensKey.self] = newValue.tokens
        }
    }

    /// The concrete `ThemeTokens` for the active theme. This is the value
    /// 99% of views need — color palette, type scale, spacing, radii.
    public var themeTokens: ThemeTokens {
        get { self[ThemeTokensKey.self] }
        set { self[ThemeTokensKey.self] = newValue }
    }
}

// MARK: - View modifier

extension View {

    /// Apply a theme to this view subtree. Pass a `Theme` value for a fixed
    /// theme (Previews) or use the variant that takes a binding from
    /// `RootView` for the live app.
    public func appTheme(_ theme: Theme) -> some View {
        environment(\.theme, theme)
            .preferredColorScheme(theme.isDark ? .dark : .light)
            .background(theme.tokens.colors.bg.ignoresSafeArea())
    }
}
