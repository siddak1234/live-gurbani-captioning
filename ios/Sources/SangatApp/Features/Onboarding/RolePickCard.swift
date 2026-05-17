//
//  RolePickCard.swift
//  GurbaniCaptioningApp · SangatApp · Features/Onboarding
//
//  Card 4 of 4 — design canvas section 03, `V1Onb04_RolePick`
//  (`assets/v1-paper.jsx` lines 280-345).
//
//  Visual: meta header; "How will you use it today?" title + small
//  reassurance subtitle; two stacked cards (Sangat default-selected
//  with checkmark badge, Sevadar with "Requires unlock" caption).
//  Bottom: full-width "Continue as <selected>" pill (ink bg).
//
//  Tapping a card flips `selectedRole` on the view model. The CTA
//  label tracks the selection so users see exactly which mode they're
//  committing to.

import SwiftUI

public struct RolePickCard: View {

    @Environment(\.themeTokens) private var tokens

    @Binding public var selectedRole: AppMode
    public let onBack: () -> Void
    public let onContinue: () -> Void

    public init(
        selectedRole: Binding<AppMode>,
        onBack: @escaping () -> Void,
        onContinue: @escaping () -> Void
    ) {
        self._selectedRole = selectedRole
        self.onBack = onBack
        self.onContinue = onContinue
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            OnboardingMetaHeader(mode: selectedRole)

            VStack(alignment: .leading, spacing: tokens.spacing.sm) {
                Text("How will you use it today?")
                    .font(tokens.type.serifTitle)
                    .foregroundStyle(tokens.colors.ink)
                    .accessibilityAddTraits(.isHeader)

                Text("You can change this anytime from settings.")
                    .font(tokens.type.sans)
                    .foregroundStyle(tokens.colors.ink3)
            }
            .padding(.horizontal, tokens.spacing.edge + tokens.spacing.md)
            .padding(.top, tokens.spacing.xxl)

            VStack(spacing: tokens.spacing.md) {
                roleCard(
                    role: .sangat,
                    titleGurmukhi: "ਸੰਗਤ",
                    titleLatin: "Sangat",
                    body: "Read along during kirtan. Quiet interface, big type, almost no controls.",
                    showsRequiresUnlock: false
                )

                roleCard(
                    role: .sevadar,
                    titleGurmukhi: "ਸੇਵਾਦਾਰ",
                    titleLatin: "Sevadar",
                    body: "Operator mode. Manually pick shabads, nudge lines, cast to the projector, see engine confidence.",
                    showsRequiresUnlock: true
                )
            }
            .padding(.horizontal, tokens.spacing.edge)
            .padding(.top, tokens.spacing.xl)

            Spacer()

            buttonRow
                .padding(.horizontal, tokens.spacing.edge)
                .padding(.bottom, tokens.spacing.md)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private func roleCard(
        role: AppMode,
        titleGurmukhi: String,
        titleLatin: String,
        body: String,
        showsRequiresUnlock: Bool
    ) -> some View {
        Button {
            selectedRole = role
        } label: {
            HStack(alignment: .top, spacing: 0) {
                VStack(alignment: .leading, spacing: tokens.spacing.xs) {
                    HStack(spacing: tokens.spacing.sm) {
                        Text(titleGurmukhi)
                            .font(tokens.type.gurmukhiLarge.weight(.medium))
                            .foregroundStyle(tokens.colors.ink)
                        Text("·")
                            .foregroundStyle(tokens.colors.ink3)
                        Text(titleLatin)
                            .font(tokens.type.gurmukhiLarge.weight(.medium))
                            .foregroundStyle(tokens.colors.ink)
                    }

                    Text(body)
                        .font(tokens.type.sans)
                        .foregroundStyle(tokens.colors.ink2)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)

                    if showsRequiresUnlock {
                        Text("Requires unlock")
                            .font(tokens.type.sansCaps)
                            .tracking(0.6)
                            .foregroundStyle(tokens.colors.ink3)
                            .padding(.top, tokens.spacing.xs)
                    }
                }

                Spacer(minLength: tokens.spacing.md)

                if selectedRole == role {
                    ZStack {
                        Circle()
                            .fill(tokens.colors.ink)
                            .frame(width: 22, height: 22)
                        Image(systemName: "checkmark")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundStyle(tokens.colors.bg)
                    }
                    .accessibilityHidden(true)
                }
            }
            .padding(tokens.spacing.md + 4)
            .background(
                selectedRole == role ? tokens.colors.bgSoft : Color.clear,
                in: RoundedRectangle(cornerRadius: tokens.radii.lg, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: tokens.radii.lg, style: .continuous)
                    .strokeBorder(
                        selectedRole == role ? tokens.colors.ink : tokens.colors.rule,
                        lineWidth: selectedRole == role ? 1.5 : 1
                    )
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(titleLatin). \(body)")
        .accessibilityAddTraits(selectedRole == role ? [.isButton, .isSelected] : .isButton)
        .accessibilityIdentifier("onboarding.role.\(role.rawValue)")
    }

    private var buttonRow: some View {
        HStack(spacing: tokens.spacing.sm) {
            Button(action: onBack) {
                Text("Back")
                    .font(tokens.type.sans.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, tokens.spacing.md + 2)
                    .foregroundStyle(tokens.colors.ink2)
                    .overlay(
                        Capsule().stroke(tokens.colors.rule, lineWidth: 1)
                    )
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("onboarding.role.back")

            Button(action: onContinue) {
                Text(continueLabel)
                    .font(tokens.type.sans.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, tokens.spacing.md + 2)
                    .background(tokens.colors.ink, in: Capsule())
                    .foregroundStyle(tokens.colors.bg)
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("onboarding.role.continue")
            .layoutPriority(1)
        }
    }

    private var continueLabel: String {
        switch selectedRole {
        case .sangat:  return "Continue as Sangat"
        case .sevadar: return "Continue as Sevadar"
        }
    }
}

#Preview("RolePickCard · paper sangat") {
    StatefulPreviewWrapper(AppMode.sangat) { binding in
        RolePickCard(selectedRole: binding, onBack: {}, onContinue: {})
    }
    .previewTheme(.paper)
}

#Preview("RolePickCard · darbar sevadar") {
    StatefulPreviewWrapper(AppMode.sevadar) { binding in
        RolePickCard(selectedRole: binding, onBack: {}, onContinue: {})
    }
    .previewTheme(.darbar)
}

#Preview("RolePickCard · mool sangat") {
    StatefulPreviewWrapper(AppMode.sangat) { binding in
        RolePickCard(selectedRole: binding, onBack: {}, onContinue: {})
    }
    .previewTheme(.mool)
}

/// Tiny preview helper so `#Preview` can host a `@Binding`.
private struct StatefulPreviewWrapper<Value, Content: View>: View {
    @State private var value: Value
    let content: (Binding<Value>) -> Content

    init(_ initial: Value, @ViewBuilder content: @escaping (Binding<Value>) -> Content) {
        self._value = State(initialValue: initial)
        self.content = content
    }

    var body: some View {
        content($value)
    }
}
