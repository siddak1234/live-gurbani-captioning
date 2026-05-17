//
//  ThemeTokens.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  The concrete design token bag for one theme. A `ThemeTokens` value is
//  *the* design system — every color, font, spacing unit, and radius that
//  any view in the app references must live here. Hex literals and
//  `Color(red:green:blue:)` in `Features/` are caught by the SwiftLint
//  custom rule defined in `ios/.swiftlint.yml` (invariant A6).
//
//  Concrete preset values live in `ThemeTokens+Presets.swift` — separated so
//  this file stays a clean type declaration and presets stay reviewable as
//  pure data.

import SwiftUI

/// Concrete tokens for one theme — color palette, type scale, spacing, radii.
///
/// Views never construct this directly; they read it via
/// `\.themeTokens` from the SwiftUI Environment (see `ThemeEnvironment.swift`).
public struct ThemeTokens: Sendable {

    public let colors: ColorPalette
    public let type: TypeScale
    public let spacing: SpacingScale
    public let radii: RadiusScale

    public init(
        colors: ColorPalette,
        type: TypeScale,
        spacing: SpacingScale = .default,
        radii: RadiusScale = .default
    ) {
        self.colors = colors
        self.type = type
        self.spacing = spacing
        self.radii = radii
    }

    // MARK: - Color palette

    /// Color tokens for one theme. Names are semantic (`ink`, `bg`, `accent`)
    /// rather than literal (`saffron`, `cream`) so the same view code works
    /// across all three themes.
    public struct ColorPalette: Sendable {
        /// Page background — the canvas every screen sits on.
        public let bg: Color
        /// Slightly raised background — secondary surfaces (chips, soft cards).
        public let bgSoft: Color
        /// Primary foreground text — headlines, primary copy, active state.
        public let ink: Color
        /// Secondary foreground text — translit, descriptions, body.
        public let ink2: Color
        /// Tertiary foreground text — meta, hints, disabled, captions.
        public let ink3: Color
        /// Dividers and rule lines.
        public let rule: Color
        /// Faint dividers — list-row separators, light section breaks.
        public let ruleSoft: Color
        /// Single primary accent — saffron / wheat / terracotta by theme.
        public let accent: Color
        /// Accent at low alpha — for tinted backgrounds (committed pill, active row).
        public let accentSoft: Color
        /// Listening / tentative state color — amber, distinct from accent.
        public let amber: Color
        /// Sevadar mode indicator color — operator-state distinct from sangat.
        public let sevadar: Color
        /// Raised surface — bottom-sheets, modal cards, the Sevadar dock.
        public let surface: Color

        public init(
            bg: Color, bgSoft: Color,
            ink: Color, ink2: Color, ink3: Color,
            rule: Color, ruleSoft: Color,
            accent: Color, accentSoft: Color,
            amber: Color,
            sevadar: Color,
            surface: Color
        ) {
            self.bg = bg
            self.bgSoft = bgSoft
            self.ink = ink
            self.ink2 = ink2
            self.ink3 = ink3
            self.rule = rule
            self.ruleSoft = ruleSoft
            self.accent = accent
            self.accentSoft = accentSoft
            self.amber = amber
            self.sevadar = sevadar
            self.surface = surface
        }
    }

    // MARK: - Typography

    /// Type tokens for one theme. All view fonts route through these — never
    /// `.system(size: 17)` directly inside a feature view (invariant A6).
    ///
    /// `gurmukhi*` use `Noto Serif Gurmukhi` (or `Noto Sans Gurmukhi` for `.mool`)
    /// when bundled; we fall back to the system Gurmukhi font otherwise.
    public struct TypeScale: Sendable {
        /// Body Gurmukhi — list rows, settings labels.
        public let gurmukhi: Font
        /// Sub-hero Gurmukhi — karaoke prev/next, full-shabad inactive lines.
        public let gurmukhiLarge: Font
        /// Hero Gurmukhi — the committed line in `HeroLineView`.
        public let gurmukhiHero: Font
        /// English / translit body.
        public let serif: Font
        /// English translit italic — used for transliteration rows.
        public let serifItalic: Font
        /// UI chrome — buttons, labels, body copy in chrome.
        public let sans: Font
        /// Small UI text — captions, secondary chrome.
        public let sansSmall: Font
        /// All-caps chrome label — section headers, state pills.
        public let sansCaps: Font
        /// Numbers, verse IDs, confidence values — monospace, tabular.
        public let mono: Font

        public init(
            gurmukhi: Font, gurmukhiLarge: Font, gurmukhiHero: Font,
            serif: Font, serifItalic: Font,
            sans: Font, sansSmall: Font, sansCaps: Font,
            mono: Font
        ) {
            self.gurmukhi = gurmukhi
            self.gurmukhiLarge = gurmukhiLarge
            self.gurmukhiHero = gurmukhiHero
            self.serif = serif
            self.serifItalic = serifItalic
            self.sans = sans
            self.sansSmall = sansSmall
            self.sansCaps = sansCaps
            self.mono = mono
        }
    }

    // MARK: - Spacing

    /// Spacing tokens. Views use these for any padding, gap, or margin.
    /// Raw numeric literals in `Features/` are an audit failure.
    public struct SpacingScale: Sendable {
        public let xs: CGFloat
        public let sm: CGFloat
        public let md: CGFloat
        public let lg: CGFloat
        public let xl: CGFloat
        public let xxl: CGFloat
        /// Standard horizontal edge padding for a screen — keep consistent across views.
        public let edge: CGFloat

        public init(
            xs: CGFloat = 4, sm: CGFloat = 8, md: CGFloat = 16,
            lg: CGFloat = 24, xl: CGFloat = 32, xxl: CGFloat = 48,
            edge: CGFloat = 22
        ) {
            self.xs = xs; self.sm = sm; self.md = md
            self.lg = lg; self.xl = xl; self.xxl = xxl
            self.edge = edge
        }

        public static let `default` = SpacingScale()
    }

    // MARK: - Radii

    /// Corner radius tokens. `pill` is "fully rounded" — use it for any pill
    /// or fully-rounded surface so we render consistently.
    public struct RadiusScale: Sendable {
        public let xs: CGFloat
        public let sm: CGFloat
        public let md: CGFloat
        public let lg: CGFloat
        public let xl: CGFloat
        public let pill: CGFloat

        public init(
            xs: CGFloat = 6, sm: CGFloat = 10, md: CGFloat = 14,
            lg: CGFloat = 18, xl: CGFloat = 22, pill: CGFloat = 999
        ) {
            self.xs = xs; self.sm = sm; self.md = md
            self.lg = lg; self.xl = xl; self.pill = pill
        }

        public static let `default` = RadiusScale()
    }
}
