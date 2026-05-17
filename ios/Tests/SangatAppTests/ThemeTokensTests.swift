//
//  ThemeTokensTests.swift
//  SangatAppTests — M5.1 (Foundations)
//
//  Verifies each shipped theme provides a complete `ThemeTokens` value and
//  that `ink` / `ink2` text colors meet WCAG AA contrast against their `bg`.
//  If you adjust a color in `ThemeTokens+Presets.swift` and the contrast
//  check fails, fix the color — do not relax the threshold.

import XCTest
import SwiftUI
@testable import SangatApp

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

final class ThemeTokensTests: XCTestCase {

    // MARK: - Instantiation

    func testAllThemesInstantiateTokens() {
        for theme in Theme.allCases {
            let tokens = theme.tokens
            // Sanity: tokens exist and the ink is not the same as bg.
            XCTAssertNotEqual(
                describe(tokens.colors.ink),
                describe(tokens.colors.bg),
                "Theme \(theme.rawValue): ink == bg — text would be invisible"
            )
        }
    }

    func testAllThemesProvideAccentDifferentFromInk() {
        for theme in Theme.allCases {
            let tokens = theme.tokens
            XCTAssertNotEqual(
                describe(tokens.colors.accent),
                describe(tokens.colors.ink),
                "Theme \(theme.rawValue): accent == ink — accent would be indistinguishable"
            )
        }
    }

    // MARK: - WCAG AA contrast — primary text (ink against bg)

    func testInkOnBgMeetsWCAGAA() {
        for theme in Theme.allCases {
            let tokens = theme.tokens
            guard let ratio = contrastRatio(tokens.colors.ink, tokens.colors.bg) else {
                XCTFail("Theme \(theme.rawValue): could not compute contrast ratio")
                continue
            }
            XCTAssertGreaterThanOrEqual(
                ratio, 4.5,
                "Theme \(theme.rawValue): ink-on-bg contrast \(ratio) below WCAG AA (4.5:1)"
            )
        }
    }

    // MARK: - WCAG AA Large contrast — secondary + tertiary text

    func testInk2OnBgMeetsWCAGAALarge() {
        for theme in Theme.allCases {
            let tokens = theme.tokens
            guard let ratio = contrastRatio(tokens.colors.ink2, tokens.colors.bg) else {
                XCTFail("Theme \(theme.rawValue): could not compute contrast for ink2")
                continue
            }
            XCTAssertGreaterThanOrEqual(
                ratio, 3.0,
                "Theme \(theme.rawValue): ink2-on-bg contrast \(ratio) below WCAG AA Large (3:1)"
            )
        }
    }

    // MARK: - Cross-cut sanity

    func testRadiusScaleIsMonotonic() {
        let r = ThemeTokens.RadiusScale.default
        XCTAssertLessThan(r.xs, r.sm)
        XCTAssertLessThan(r.sm, r.md)
        XCTAssertLessThan(r.md, r.lg)
        XCTAssertLessThan(r.lg, r.xl)
        XCTAssertLessThan(r.xl, r.pill)
    }

    func testSpacingScaleIsMonotonic() {
        let s = ThemeTokens.SpacingScale.default
        XCTAssertLessThan(s.xs, s.sm)
        XCTAssertLessThan(s.sm, s.md)
        XCTAssertLessThan(s.md, s.lg)
        XCTAssertLessThan(s.lg, s.xl)
        XCTAssertLessThan(s.xl, s.xxl)
    }

    // MARK: - Helpers

    /// WCAG 2.x contrast ratio. Returns `nil` if either color's RGB
    /// components can't be resolved on the host platform.
    private func contrastRatio(_ a: Color, _ b: Color) -> Double? {
        guard let ar = rgb(a), let br = rgb(b) else { return nil }
        let la = relativeLuminance(r: ar.r, g: ar.g, b: ar.b)
        let lb = relativeLuminance(r: br.r, g: br.g, b: br.b)
        let lighter = max(la, lb)
        let darker = min(la, lb)
        return (lighter + 0.05) / (darker + 0.05)
    }

    private func relativeLuminance(r: Double, g: Double, b: Double) -> Double {
        func channel(_ c: Double) -> Double {
            c <= 0.03928 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
        }
        return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
    }

    private func rgb(_ color: Color) -> (r: Double, g: Double, b: Double)? {
        #if canImport(UIKit)
        let ui = UIColor(color)
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        guard ui.getRed(&r, green: &g, blue: &b, alpha: &a) else { return nil }
        return (Double(r), Double(g), Double(b))
        #elseif canImport(AppKit)
        guard let ns = NSColor(color).usingColorSpace(.sRGB) else { return nil }
        return (Double(ns.redComponent), Double(ns.greenComponent), Double(ns.blueComponent))
        #else
        return nil
        #endif
    }

    private func describe(_ color: Color) -> String {
        if let rgb = rgb(color) {
            return String(format: "rgb(%.3f, %.3f, %.3f)", rgb.r, rgb.g, rgb.b)
        }
        return String(describing: color)
    }
}
