//
//  PickRow.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  List row for the shabad picker, history, and any "pick one of several"
//  surfaces. Gurmukhi title + Latin meta + optional `live` chip + tap action.

import SwiftUI

public struct PickRow: View {

    @Environment(\.themeTokens) private var tokens

    public let titleGurmukhi: String
    public let meta: String
    public let isLive: Bool
    public let isActive: Bool
    public let trailing: AnyView?
    public let action: () -> Void

    public init(
        titleGurmukhi: String,
        meta: String,
        isLive: Bool = false,
        isActive: Bool = false,
        trailing: AnyView? = nil,
        action: @escaping () -> Void = {}
    ) {
        self.titleGurmukhi = titleGurmukhi
        self.meta = meta
        self.isLive = isLive
        self.isActive = isActive
        self.trailing = trailing
        self.action = action
    }

    public var body: some View {
        Button(action: action) {
            HStack(alignment: .center, spacing: tokens.spacing.md) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(titleGurmukhi)
                        .font(tokens.type.gurmukhi)
                        .foregroundStyle(tokens.colors.ink)
                        .lineLimit(1)
                    Text(meta)
                        .font(tokens.type.sansSmall)
                        .foregroundStyle(tokens.colors.ink3)
                        .lineLimit(1)
                }
                Spacer(minLength: tokens.spacing.sm)
                if isLive {
                    Text("Live".uppercased())
                        .font(tokens.type.sansCaps)
                        .tracking(0.7)
                        .foregroundStyle(tokens.colors.accent)
                }
                if let trailing {
                    trailing
                }
            }
            .padding(.horizontal, tokens.spacing.edge)
            .padding(.vertical, tokens.spacing.md)
            .background(
                isActive ? tokens.colors.accentSoft : Color.clear
            )
        }
        .buttonStyle(.plain)
        .overlay(
            Rectangle()
                .fill(tokens.colors.ruleSoft)
                .frame(height: 0.5),
            alignment: .bottom
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel(Text("\(titleGurmukhi), \(meta)"))
        .accessibilityAddTraits(.isButton)
    }
}

#Preview("PickRow · all themes") {
    VStack(spacing: 32) {
        ForEach(Theme.allCases) { theme in
            VStack(spacing: 0) {
                PickRow(titleGurmukhi: "ਤਾਤੀ ਵਾਉ ਨ ਲਗਈ",
                        meta: "Guru Arjan · Ang 819",
                        isLive: true, isActive: true) {}
                PickRow(titleGurmukhi: "ਸੋ ਦਰੁ ਕੇਹਾ ਸੋ ਘਰੁ ਕੇਹਾ",
                        meta: "Guru Nanak · Ang 8") {}
            }
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity)
            .background(theme.tokens.colors.bg)
            .environment(\.themeTokens, theme.tokens)
        }
    }
}
