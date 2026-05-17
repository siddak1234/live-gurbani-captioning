//
//  GurbaniText.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Renders shabad content. Single source of truth for how the three
//  language layers (Gurmukhi / translit / English) compose visually.
//  Every reading view uses this — never a raw `Text` for shabad content
//  (invariant A6 enforces in feature folders via SwiftLint).

import SwiftUI

public struct GurbaniText: View {

    public enum Size: Sendable {
        case body      // list rows, sub-hero
        case karaoke   // prev/next karaoke neighbors
        case large     // emphasized but not hero
        case hero      // committed-line star
    }

    @Environment(\.themeTokens) private var tokens

    public let gurmukhi: String
    public let translit: String?
    public let english: String?
    public let size: Size
    public let textAlignment: TextAlignment

    public init(
        gurmukhi: String,
        translit: String? = nil,
        english: String? = nil,
        size: Size = .body,
        textAlignment: TextAlignment = .center
    ) {
        self.gurmukhi = gurmukhi
        self.translit = translit
        self.english = english
        self.size = size
        self.textAlignment = textAlignment
    }

    public var body: some View {
        VStack(spacing: spacingBetweenLayers) {
            Text(gurmukhi)
                .font(gurmukhiFont)
                .foregroundStyle(tokens.colors.ink)
                .multilineTextAlignment(textAlignment)
                .accessibilityAddTraits(.isHeader)
                .accessibilityLabel(Text(gurmukhi))

            if let translit, !translit.isEmpty {
                Text(translit)
                    .font(tokens.type.serifItalic)
                    .foregroundStyle(tokens.colors.ink2)
                    .multilineTextAlignment(textAlignment)
                    .accessibilityLabel(Text("Transliteration: \(translit)"))
            }

            if let english, !english.isEmpty {
                Rectangle()
                    .fill(tokens.colors.rule)
                    .frame(maxWidth: 220, maxHeight: 1)
                    .padding(.vertical, tokens.spacing.xs)
                Text(english)
                    .font(tokens.type.serif)
                    .foregroundStyle(tokens.colors.ink2)
                    .multilineTextAlignment(textAlignment)
                    .accessibilityLabel(Text("Meaning: \(english)"))
            }
        }
        .frame(maxWidth: .infinity, alignment: frameAlignment)
    }

    // MARK: - Resolution

    private var gurmukhiFont: Font {
        switch size {
        case .body:    return tokens.type.gurmukhi
        case .karaoke: return tokens.type.gurmukhiLarge
        case .large:   return tokens.type.gurmukhiLarge
        case .hero:    return tokens.type.gurmukhiHero
        }
    }

    private var spacingBetweenLayers: CGFloat {
        switch size {
        case .body, .karaoke: return tokens.spacing.sm
        case .large, .hero:   return tokens.spacing.md
        }
    }

    private var frameAlignment: Alignment {
        switch textAlignment {
        case .leading:  return .leading
        case .trailing: return .trailing
        case .center:   return .center
        }
    }
}

#Preview("GurbaniText · sizes") {
    ScrollView {
        VStack(spacing: 32) {
            ForEach(Theme.allCases) { theme in
                VStack(spacing: 24) {
                    Text(theme.displayName.uppercased())
                        .font(.system(size: 10, weight: .semibold))
                        .tracking(0.6)
                        .foregroundStyle(.secondary)
                    GurbaniText(
                        gurmukhi: "ਤਾਤੀ ਵਾਉ ਨ ਲਗਈ ਪਾਰਬ੍ਰਹਮ ਸਰਣਾਈ ॥",
                        translit: "Tātī vā▫o na lag▫ī pārbarahm sarṇā▫ī.",
                        english: "The hot wind cannot even touch one who is under the Protection of the Supreme Lord God.",
                        size: .hero
                    )
                    GurbaniText(
                        gurmukhi: "ਚਉਗਿਰਦ ਹਮਾਰੈ ਰਾਮ ਕਾਰ ਦੁਖੁ ਲਗੈ ਨ ਭਾਈ ॥",
                        size: .karaoke
                    )
                }
                .padding(20)
                .background(theme.tokens.colors.bg)
                .environment(\.themeTokens, theme.tokens)
            }
        }
    }
}
