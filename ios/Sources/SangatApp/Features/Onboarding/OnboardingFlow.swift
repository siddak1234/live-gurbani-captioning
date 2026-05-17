//
//  OnboardingFlow.swift
//  GurbaniCaptioningApp · SangatApp · Features/Onboarding
//
//  Top-level container for the four-card onboarding sequence
//  (design canvas section 03; `assets/v1-paper.jsx` 150-345). Owns the
//  `OnboardingViewModel`, dispatches to one of `WelcomeCard`,
//  `HowItWorksCard`, `MicPermissionCard`, `RolePickCard` based on
//  `vm.step`, and commits the user's choices to `AppEnvironment` when
//  the user lands on the Role-pick Continue button.
//
//  Completion writes three things in this order:
//    1. `env.mode = vm.selectedRole`         — drives Sangat vs Sevadar
//    2. `env.preferences.micPermissionAcknowledged = vm.micPermissionAcknowledged`
//    3. `env.hasCompletedOnboarding = true`  — flips Root to IdleView
//
//  Step 3 must run last so the env tree re-renders into IdleView with
//  the mode + mic flag already set.

import SwiftUI

public struct OnboardingFlow: View {

    @Environment(AppEnvironment.self) private var env
    @Environment(\.themeTokens) private var tokens

    @State private var viewModel: OnboardingViewModel

    public init(initial: OnboardingStep = .welcome) {
        _viewModel = State(initialValue: OnboardingViewModel(initial: initial))
    }

    public var body: some View {
        @Bindable var vm = viewModel

        ZStack {
            tokens.colors.bg.ignoresSafeArea()

            currentCard(vm: vm)
                .id(vm.step)
                .transition(.asymmetric(
                    insertion: .opacity.combined(with: .move(edge: .trailing)),
                    removal: .opacity.combined(with: .move(edge: .leading))
                ))
        }
        .animation(.easeInOut(duration: 0.25), value: vm.step)
    }

    @ViewBuilder
    private func currentCard(vm: OnboardingViewModel) -> some View {
        switch vm.step {
        case .welcome:
            WelcomeCard {
                env.haptics.play(.selection)
                vm.advance()
            }

        case .howItWorks:
            HowItWorksCard(
                onBack: {
                    env.haptics.play(.selection)
                    vm.back()
                },
                onContinue: {
                    env.haptics.play(.selection)
                    vm.advance()
                }
            )

        case .mic:
            MicPermissionCard(
                isRequestInFlight: vm.micRequestInFlight,
                onAllow: {
                    env.haptics.play(.selection)
                    Task { await vm.requestMicPermission() }
                },
                onSkip: {
                    env.haptics.play(.selection)
                    vm.skipMic()
                }
            )

        case .role:
            RolePickCard(
                selectedRole: Binding(
                    get: { vm.selectedRole },
                    set: { vm.selectedRole = $0 }
                ),
                onBack: {
                    env.haptics.play(.selection)
                    vm.back()
                },
                onContinue: {
                    complete()
                }
            )
        }
    }

    /// Apply selections + flip `hasCompletedOnboarding`. Order matters
    /// — mode + mic ack first, then onboarding flag last so the env
    /// re-render lands into IdleView with the right values already
    /// committed.
    private func complete() {
        env.haptics.play(.success)
        env.mode = viewModel.selectedRole
        env.preferences.micPermissionAcknowledged = viewModel.micPermissionAcknowledged
        env.hasCompletedOnboarding = true

        AppLogger.app.info("Onboarding complete — mode=\(self.env.mode.rawValue, privacy: .public) micAck=\(self.viewModel.micPermissionAcknowledged, privacy: .public)")
    }
}

#Preview("OnboardingFlow · paper") {
    OnboardingFlow()
        .environment(AppEnvironment.preview(hasCompletedOnboarding: false))
        .previewTheme(.paper)
}

#Preview("OnboardingFlow · darbar") {
    OnboardingFlow()
        .environment(AppEnvironment.preview(theme: .darbar, hasCompletedOnboarding: false))
        .previewTheme(.darbar)
}

#Preview("OnboardingFlow · mool, starts at role pick") {
    OnboardingFlow(initial: .role)
        .environment(AppEnvironment.preview(theme: .mool, hasCompletedOnboarding: false))
        .previewTheme(.mool)
}
