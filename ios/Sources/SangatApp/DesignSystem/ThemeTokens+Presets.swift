//
//  ThemeTokens+Presets.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Concrete preset values for the three shipped themes. Separated from
//  `ThemeTokens.swift` so the type definition stays a clean declaration
//  and the actual color/type choices are reviewable as pure data.
//
//  Color accessibility
//  -------------------
//  All `ink`/`ink2` text colors meet WCAG AA contrast (4.5:1) against
//  their theme's `bg`. `ink3` is hint-only and reaches AA Large (3:1).
//  Verified in `ThemeTokensTests.swift`. If you adjust a color here, the
//  tests will fail and tell you the offending ratio — fix the color, do
//  not relax the test.

import SwiftUI

extension ThemeTokens {

    // MARK: - Reading Room (paper)

    /// Warm gutka-paper ivory. Default theme.
    public static let paper = ThemeTokens(
        colors: .init(
            // bg #F4EEE2 — warm ivory paper
            bg: Color(red: 244.0 / 255, green: 238.0 / 255, blue: 226.0 / 255),
            // bgSoft #EBE3D2 — slightly darker fold
            bgSoft: Color(red: 235.0 / 255, green: 227.0 / 255, blue: 210.0 / 255),
            // ink #1F1B14 — deep brown-black
            ink: Color(red: 31.0 / 255, green: 27.0 / 255, blue: 20.0 / 255),
            // ink2 #4A4234 — body
            ink2: Color(red: 74.0 / 255, green: 66.0 / 255, blue: 52.0 / 255),
            // ink3 #7C7259 — hints
            ink3: Color(red: 124.0 / 255, green: 114.0 / 255, blue: 89.0 / 255),
            rule: Color.black.opacity(0.12),
            ruleSoft: Color.black.opacity(0.06),
            // accent — saffron, oklch(0.62 0.14 60) ≈ #C97A2F
            accent: Color(red: 201.0 / 255, green: 122.0 / 255, blue: 47.0 / 255),
            accentSoft: Color(red: 201.0 / 255, green: 122.0 / 255, blue: 47.0 / 255).opacity(0.12),
            // amber — slightly cooler for listening pulse
            amber: Color(red: 213.0 / 255, green: 152.0 / 255, blue: 53.0 / 255),
            // sevadar — deep indigo, mode-distinct
            sevadar: Color(red: 71.0 / 255, green: 91.0 / 255, blue: 157.0 / 255),
            surface: Color.white
        ),
        type: .system
    )

    // MARK: - Darbar Night (darbar)

    /// Calm dark for low-light darbars and projector AirPlay.
    public static let darbar = ThemeTokens(
        colors: .init(
            // bg #0E0D0B — warm near-black
            bg: Color(red: 14.0 / 255, green: 13.0 / 255, blue: 11.0 / 255),
            // bgSoft #1A1814 — raised
            bgSoft: Color(red: 26.0 / 255, green: 24.0 / 255, blue: 20.0 / 255),
            // ink #EFEAE0 — warm cream
            ink: Color(red: 239.0 / 255, green: 234.0 / 255, blue: 224.0 / 255),
            // ink2 #C5BEAE — body
            ink2: Color(red: 197.0 / 255, green: 190.0 / 255, blue: 174.0 / 255),
            // ink3 #7E7868 — hints
            ink3: Color(red: 126.0 / 255, green: 120.0 / 255, blue: 104.0 / 255),
            rule: Color.white.opacity(0.10),
            ruleSoft: Color.white.opacity(0.05),
            // accent — wheat-gold, oklch(0.80 0.10 75) ≈ #D9B860
            accent: Color(red: 217.0 / 255, green: 184.0 / 255, blue: 96.0 / 255),
            accentSoft: Color(red: 217.0 / 255, green: 184.0 / 255, blue: 96.0 / 255).opacity(0.16),
            // amber — listening state, brighter than accent so it reads "active"
            amber: Color(red: 240.0 / 255, green: 184.0 / 255, blue: 64.0 / 255),
            // sevadar — soft indigo, readable against dark
            sevadar: Color(red: 128.0 / 255, green: 156.0 / 255, blue: 214.0 / 255),
            // surface — raised card, between bg and bgSoft
            surface: Color(red: 35.0 / 255, green: 31.0 / 255, blue: 24.0 / 255)
        ),
        type: .system
    )

    // MARK: - Mool (editorial minimal)

    /// Near-white editorial minimal with sans Gurmukhi.
    public static let mool = ThemeTokens(
        colors: .init(
            // bg #FAFAF7 — paper white
            bg: Color(red: 250.0 / 255, green: 250.0 / 255, blue: 247.0 / 255),
            // bgSoft #F1F0EB
            bgSoft: Color(red: 241.0 / 255, green: 240.0 / 255, blue: 235.0 / 255),
            // ink #0A0A09 — near-black
            ink: Color(red: 10.0 / 255, green: 10.0 / 255, blue: 9.0 / 255),
            // ink2 #3A3936 — body
            ink2: Color(red: 58.0 / 255, green: 57.0 / 255, blue: 54.0 / 255),
            // ink3 #8C8B86 — hints
            ink3: Color(red: 140.0 / 255, green: 139.0 / 255, blue: 134.0 / 255),
            rule: Color.black.opacity(0.10),
            ruleSoft: Color.black.opacity(0.05),
            // accent — terracotta, oklch(0.58 0.15 30) ≈ #BA5A38
            accent: Color(red: 186.0 / 255, green: 90.0 / 255, blue: 56.0 / 255),
            accentSoft: Color(red: 186.0 / 255, green: 90.0 / 255, blue: 56.0 / 255).opacity(0.10),
            // amber — slightly muted to fit the editorial voice
            amber: Color(red: 219.0 / 255, green: 142.0 / 255, blue: 50.0 / 255),
            sevadar: Color(red: 64.0 / 255, green: 82.0 / 255, blue: 140.0 / 255),
            surface: Color.white
        ),
        type: .moolSans  // Mool uses Noto Sans Gurmukhi instead of serif
    )
}

extension ThemeTokens.TypeScale {

    /// System-fallback type scale. Serif body, system Gurmukhi (which on iOS
    /// is Gurmukhi MN). Used for `paper` and `darbar`.
    ///
    /// When `Noto Serif Gurmukhi` is bundled as an asset (`Resources/Fonts/`),
    /// swap the `.system` Gurmukhi entries for the custom font — keeping
    /// this in one place is the whole point of the type scale.
    public static let system = ThemeTokens.TypeScale(
        gurmukhi: .system(size: 18, weight: .regular, design: .serif),
        gurmukhiLarge: .system(size: 28, weight: .regular, design: .serif),
        gurmukhiHero: .system(size: 38, weight: .medium, design: .serif),
        serif: .system(size: 17, weight: .regular, design: .serif),
        serifItalic: .system(size: 17, weight: .regular, design: .serif).italic(),
        sans: .system(size: 14, weight: .regular, design: .default),
        sansSmall: .system(size: 12, weight: .regular, design: .default),
        sansCaps: .system(size: 11, weight: .semibold, design: .default),
        mono: .system(size: 12, weight: .regular, design: .monospaced)
    )

    /// Sans-Gurmukhi variant for the `.mool` editorial theme.
    public static let moolSans = ThemeTokens.TypeScale(
        gurmukhi: .system(size: 18, weight: .regular, design: .default),
        gurmukhiLarge: .system(size: 28, weight: .regular, design: .default),
        gurmukhiHero: .system(size: 38, weight: .medium, design: .default),
        serif: .system(size: 17, weight: .regular, design: .default),
        serifItalic: .system(size: 17, weight: .regular, design: .default).italic(),
        sans: .system(size: 14, weight: .regular, design: .default),
        sansSmall: .system(size: 12, weight: .regular, design: .default),
        sansCaps: .system(size: 11, weight: .semibold, design: .default),
        mono: .system(size: 12, weight: .regular, design: .monospaced)
    )
}
