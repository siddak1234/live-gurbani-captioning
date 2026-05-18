//
//  SevadarButton.swift
//  GurbaniCaptioningApp · SangatApp · DesignSystem/Components
//
//  Dock control button used by `SevadarDock` (Line ±1, Pause, Pick,
//  Cast, Lock). Mirrors `SevBtn` in `assets/v1-paper.jsx` (lines 823-834):
//  a rounded rectangle with sans-semibold label, white-ish fill for the
//  default variant, ink fill for the primary variant. `flex` controls
//  horizontal stretch when multiple buttons share a row.

import SwiftUI

public struct SevadarButton: View {

    @Environment(\.themeTokens) private var tokens

    public enum Style: Sendable {
        case secondary
        case primary
    }

    public let title: String
    public let style: Style
    public let isEnabled: Bool
    public let action: () -> Void

    public init(
        _ title: String,
        style: Style = .secondary,
        isEnabled: Bool = true,
        action: @escaping () -> Void
    ) {
        self.title = title
        self.style = style
        self.isEnabled = isEnabled
        self.action = action
    }

    public var body: some View {
        Button(action: action) {
            Text(title)
                .font(tokens.type.sans.weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.7)
                .frame(maxWidth: .infinity)
                .padding(.vertical, tokens.spacing.sm + 3)
                .padding(.horizontal, tokens.spacing.sm)
                .foregroundStyle(foreground)
                .background(background, in: RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous)
                        .stroke(borderColor, lineWidth: borderWidth)
                )
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .opacity(isEnabled ? 1 : 0.4)
        .accessibilityIdentifier("sevadar.button.\(title.lowercased().replacingOccurrences(of: " ", with: "."))")
    }

    private var background: Color {
        switch style {
        case .primary: return tokens.colors.ink
        case .secondary: return tokens.colors.bg
        }
    }

    private var foreground: Color {
        switch style {
        case .primary: return tokens.colors.bg
        case .secondary: return tokens.colors.ink2
        }
    }

    private var borderColor: Color {
        switch style {
        case .primary: return .clear
        case .secondary: return tokens.colors.rule
        }
    }

    private var borderWidth: CGFloat {
        switch style {
        case .primary: return 0
        case .secondary: return 1
        }
    }
}

/// Icon-only sibling of `SevadarButton`, used for the dock's nudge
/// arrows where a 1-character label would look starved inside a wide
/// equal-width frame. Same height + corner radius + border treatment
/// as `SevadarButton` so the row stays visually consistent.
public struct SevadarIconButton: View {

    @Environment(\.themeTokens) private var tokens

    public let systemName: String
    public let style: SevadarButton.Style
    public let isEnabled: Bool
    public let action: () -> Void

    public init(
        systemName: String,
        style: SevadarButton.Style = .secondary,
        isEnabled: Bool = true,
        action: @escaping () -> Void
    ) {
        self.systemName = systemName
        self.style = style
        self.isEnabled = isEnabled
        self.action = action
    }

    public var body: some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 17, weight: .semibold))
                .frame(maxWidth: .infinity)
                .padding(.vertical, tokens.spacing.sm + 3)
                .padding(.horizontal, tokens.spacing.sm)
                .foregroundStyle(foreground)
                .background(background, in: RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous)
                        .stroke(borderColor, lineWidth: borderWidth)
                )
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .opacity(isEnabled ? 1 : 0.4)
        .accessibilityIdentifier("sevadar.icon.\(systemName.replacingOccurrences(of: ".", with: "-"))")
    }

    // Style helpers mirror SevadarButton's, kept inline so the two
    // atoms don't depend on shared internals.

    private var background: Color {
        switch style {
        case .primary: return tokens.colors.ink
        case .secondary: return tokens.colors.bg
        }
    }

    private var foreground: Color {
        switch style {
        case .primary: return tokens.colors.bg
        case .secondary: return tokens.colors.ink2
        }
    }

    private var borderColor: Color {
        switch style {
        case .primary: return .clear
        case .secondary: return tokens.colors.rule
        }
    }

    private var borderWidth: CGFloat {
        switch style {
        case .primary: return 0
        case .secondary: return 1
        }
    }
}

#Preview("SevadarButton variants · paper") {
    VStack(spacing: 12) {
        HStack(spacing: 8) {
            SevadarIconButton(systemName: "arrow.left") {}
            SevadarButton("Pause auto", style: .primary) {}
            SevadarIconButton(systemName: "arrow.right") {}
        }
        HStack(spacing: 8) {
            SevadarButton("Pick shabad") {}
            SevadarButton("Cast") {}
        }
        SevadarButton("Disabled", isEnabled: false) {}
    }
    .padding()
    .previewTheme(.paper)
}

#Preview("SevadarButton variants · darbar") {
    VStack(spacing: 12) {
        HStack(spacing: 8) {
            SevadarButton("← Line −1") {}
            SevadarButton("Pause auto", style: .primary) {}
            SevadarButton("Line +1 →") {}
        }
    }
    .padding()
    .previewTheme(.darbar)
}
