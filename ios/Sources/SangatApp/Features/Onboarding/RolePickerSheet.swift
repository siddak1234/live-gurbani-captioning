//
//  RolePickerSheet.swift
//  GurbaniCaptioningApp · SangatApp · Features/Onboarding
//
//  Bottom-sheet role picker, reached from the role pill on
//  `SessionStartView`. Selection-is-commit: tapping a role card writes
//  to the binding (which the parent uses to update env.mode) and
//  dismisses the sheet — no separate Continue button, no Back.
//
//  Cards reuse the visual treatment of `RolePickCard` but at a smaller
//  scale appropriate for a bottom sheet (less title, no "Requires
//  unlock" caption — that affordance is for first-time onboarding).

import SwiftUI

public struct RolePickerSheet: View {

    @Environment(\.themeTokens) private var tokens

    @Binding public var selection: AppMode

    public init(selection: Binding<AppMode>) {
        self._selection = selection
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: tokens.spacing.lg) {
            Text("Choose role")
                .font(tokens.type.gurmukhiLarge.weight(.semibold))
                .foregroundStyle(tokens.colors.ink)
                .accessibilityAddTraits(.isHeader)

            VStack(spacing: tokens.spacing.sm) {
                roleCard(
                    role: .sangat,
                    titleGurmukhi: "ਸੰਗਤ",
                    titleLatin: "Sangat",
                    body: "Read along during kirtan. Quiet interface, big type."
                )
                roleCard(
                    role: .sevadar,
                    titleGurmukhi: "ਸੇਵਾਦਾਰ",
                    titleLatin: "Sevadar",
                    body: "Operator mode. Picker, confidence, cast."
                )
            }
        }
        .padding(.horizontal, tokens.spacing.edge)
        .padding(.top, tokens.spacing.lg)
        .padding(.bottom, tokens.spacing.lg)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(tokens.colors.bg)
        .accessibilityIdentifier("rolePicker.sheet")
    }

    @ViewBuilder
    private func roleCard(
        role: AppMode,
        titleGurmukhi: String,
        titleLatin: String,
        body: String
    ) -> some View {
        Button {
            selection = role
        } label: {
            HStack(alignment: .top, spacing: 0) {
                VStack(alignment: .leading, spacing: tokens.spacing.xs) {
                    HStack(spacing: tokens.spacing.sm) {
                        Text(titleGurmukhi)
                            .font(tokens.type.gurmukhi.weight(.medium))
                            .foregroundStyle(tokens.colors.ink)
                        Text("·")
                            .foregroundStyle(tokens.colors.ink3)
                        Text(titleLatin)
                            .font(tokens.type.serif.weight(.semibold))
                            .foregroundStyle(tokens.colors.ink)
                    }

                    Text(body)
                        .font(tokens.type.sansSmall)
                        .foregroundStyle(tokens.colors.ink2)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Spacer(minLength: tokens.spacing.md)

                if selection == role {
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
            .padding(tokens.spacing.md)
            .background(
                selection == role ? tokens.colors.bgSoft : Color.clear,
                in: RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: tokens.radii.md, style: .continuous)
                    .strokeBorder(
                        selection == role ? tokens.colors.ink : tokens.colors.rule,
                        lineWidth: selection == role ? 1.5 : 1
                    )
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(titleLatin). \(body)")
        .accessibilityAddTraits(selection == role ? [.isButton, .isSelected] : .isButton)
        .accessibilityIdentifier("rolePicker.\(role.rawValue)")
    }
}

#Preview("RolePickerSheet · paper") {
    StatefulPreviewWrapper(AppMode.sangat) { binding in
        RolePickerSheet(selection: binding)
    }
    .previewTheme(.paper)
}

#Preview("RolePickerSheet · darbar selected sevadar") {
    StatefulPreviewWrapper(AppMode.sevadar) { binding in
        RolePickerSheet(selection: binding)
    }
    .previewTheme(.darbar)
}

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
