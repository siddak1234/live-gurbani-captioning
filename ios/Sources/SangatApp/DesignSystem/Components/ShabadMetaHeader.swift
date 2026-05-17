//
//  ShabadMetaHeader.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Header shown above every reading view: raag · ang · author. Tappable
//  in the Sevadar build (M5.6) — long-press surface for the correction
//  loop. M5.1 ships the static surface only.

import SwiftUI

public struct ShabadMetaHeader: View {

    @Environment(\.themeTokens) private var tokens

    public let raag: String
    public let ang: Int
    public let author: String?
    public let authorGurmukhi: String?
    public let compact: Bool

    public init(
        raag: String,
        ang: Int,
        author: String? = nil,
        authorGurmukhi: String? = nil,
        compact: Bool = false
    ) {
        self.raag = raag
        self.ang = ang
        self.author = author
        self.authorGurmukhi = authorGurmukhi
        self.compact = compact
    }

    public var body: some View {
        VStack(spacing: tokens.spacing.xs) {
            Text("Raag \(raag) · Ang \(ang)\(authorSuffix)")
                .font(tokens.type.sansCaps)
                .tracking(1.0)
                .foregroundStyle(tokens.colors.ink3)
            if !compact, let authorGurmukhi {
                Text(authorGurmukhi)
                    .font(tokens.type.gurmukhi)
                    .foregroundStyle(tokens.colors.ink2)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private var authorSuffix: String {
        if let author { return " · \(author)" }
        return ""
    }
}

#Preview("ShabadMetaHeader · all themes") {
    VStack(spacing: 24) {
        ForEach(Theme.allCases) { theme in
            ShabadMetaHeader(
                raag: "Bilaaval",
                ang: 819,
                author: "Guru Arjan Dev Ji",
                authorGurmukhi: "ਗੁਰੂ ਅਰਜਨ ਦੇਵ ਜੀ"
            )
            .padding(20)
            .frame(maxWidth: .infinity)
            .background(theme.tokens.colors.bg)
            .environment(\.themeTokens, theme.tokens)
        }
    }
    .padding(20)
}
