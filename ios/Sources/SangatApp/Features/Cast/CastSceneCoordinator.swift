//
//  CastSceneCoordinator.swift
//  GurbaniCaptioningApp · SangatApp · Features/Cast
//
//  Bridges between UIKit's external-display lifecycle and our SwiftUI
//  `CastReadingView`. Owns the second `UIWindow` attached to a
//  connected `UIScreen` and hosts the cast SwiftUI tree inside it via
//  `UIHostingController`. The hosted tree gets the SAME
//  `AppEnvironment` reference the phone uses, so `currentGuess` /
//  theme / mode changes drive both surfaces from one observable
//  source — no manual mirroring.
//
//  iOS 17 / 18 deprecation note
//  ----------------------------
//  `UIScreen.didConnectNotification` / `didDisconnectNotification` are
//  marked deprecated in favor of `UISceneSession`-based external
//  scene activation. The replacement requires a
//  `UIApplicationSceneManifest` entry in Info.plist plus a
//  `UIWindowSceneDelegate`, which is awkward to wire inside an SPM
//  executable target driven by `@main GurbaniCaptioningApp` (SwiftUI
//  App). The deprecated notifications still fire on iOS 17 and 18, so
//  M5.5 uses them and accepts the warning; M5.7 (when the live engine
//  rewrite touches App-startup wiring anyway) can migrate to the
//  scene API. Documented here so the migration is intentional.

import SwiftUI
import Observation

#if canImport(UIKit) && os(iOS)
import UIKit
#endif

@MainActor
@Observable
public final class CastSceneCoordinator {

    /// Pure policy: is the dedicated cast projector window allowed
    /// for the given mode? Cast is a **Sevadar-only** surface — see
    /// `RolePickCard` / `RolePickerSheet` onboarding copy ("cast to
    /// the projector") and the fact that `SevadarDock` is the only
    /// in-app surface with an `onCast` affordance. A Sangat-mode
    /// user connecting an external display falls back to whatever
    /// iOS shows by default (typically a black screen with the app
    /// icon) — we do not host a projector window for them.
    ///
    /// Used by `attach(to:)` to early-return and by
    /// `modeDidChange(to:)` to reactively detach when a user flips
    /// from Sevadar to Sangat mid-session.
    public static func shouldHostCastSurface(for mode: AppMode) -> Bool {
        switch mode {
        case .sevadar:  return true
        case .sangat:   return false
        }
    }

    /// True while an external display is connected and the cast window
    /// is mounted. Observable so the phone's chrome row can show a
    /// small "Casting" indicator without polling.
    public private(set) var isExternalScreenConnected: Bool = false

    /// Optional friendly name for the connected display. Always
    /// generic ("External display") in M5.5 — system APIs don't
    /// reliably expose the AirPlay device name.
    public private(set) var externalScreenName: String?

    // MARK: - Internals

    @ObservationIgnored
    private weak var env: AppEnvironment?

    #if canImport(UIKit) && os(iOS)
    @ObservationIgnored
    private var externalWindow: UIWindow?

    @ObservationIgnored
    private var notificationObservers: [NSObjectProtocol] = []
    #endif

    // MARK: - Init

    public init(env: AppEnvironment) {
        self.env = env

        #if canImport(UIKit) && os(iOS)
        startObservingScreens()
        attachToExistingExternalScreen()
        #endif

        AppLogger.app.info("CastSceneCoordinator initialized")
    }

    deinit {
        #if canImport(UIKit) && os(iOS)
        // deinit on a @MainActor class is @MainActor-isolated; we can
        // touch main-actor properties safely. Just remove observers
        // and drop the window reference.
        for token in notificationObservers {
            NotificationCenter.default.removeObserver(token)
        }
        externalWindow?.isHidden = true
        #endif
    }

    // MARK: - Public test seam

    /// Test-only: forcibly clear state for `CastSceneCoordinatorTests`
    /// running on macOS where UIKit is unavailable. Production callers
    /// should rely on UIScreen notifications instead.
    @_spi(Testing)
    public func _testReset() {
        isExternalScreenConnected = false
        externalScreenName = nil
        #if canImport(UIKit) && os(iOS)
        externalWindow?.isHidden = true
        externalWindow = nil
        #endif
    }

    // MARK: - Reactive role gate

    /// Called from `RootView` on `env.mode` changes. If the user
    /// switches to Sangat while a projector window is mounted, we
    /// tear it down. If they switch to Sevadar while an external
    /// screen is physically connected (and we were skipping the
    /// attach for role reasons), we mount it now. Idempotent.
    ///
    /// Lives **outside** the `#if canImport(UIKit)` block so the
    /// macOS test host can call it (no projector window can exist
    /// there, but the observable state still transitions correctly).
    public func modeDidChange(to newMode: AppMode) {
        let allowed = Self.shouldHostCastSurface(for: newMode)
        switch (allowed, isExternalScreenConnected) {
        case (false, true):
            AppLogger.cast.info("modeDidChange → \(newMode.rawValue, privacy: .public); tearing down cast window")
            #if canImport(UIKit) && os(iOS)
            detach()
            #else
            // No projector window can exist without UIKit; just clear
            // the observable flags so callers see a consistent state.
            isExternalScreenConnected = false
            externalScreenName = nil
            #endif
        case (true, false):
            #if canImport(UIKit) && os(iOS)
            // Re-scan: if an external screen is currently attached at
            // the OS level, attempt the projector mount now that role
            // permits it.
            if let screen = UIScreen.screens.first(where: { $0 !== UIScreen.main }) {
                AppLogger.cast.info("modeDidChange → sevadar; external screen present, attempting attach")
                attach(to: screen)
            }
            #endif
        case (true, true), (false, false):
            break  // already in the right state
        }
    }

    #if canImport(UIKit) && os(iOS)

    // MARK: - Screen observation

    private func startObservingScreens() {
        let center = NotificationCenter.default
        let connectToken = center.addObserver(
            forName: UIScreen.didConnectNotification,
            object: nil,
            queue: .main
        ) { [weak self] note in
            Task { @MainActor in
                guard let self, let screen = note.object as? UIScreen else { return }
                self.attach(to: screen)
            }
        }
        let disconnectToken = center.addObserver(
            forName: UIScreen.didDisconnectNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.detach()
            }
        }
        notificationObservers = [connectToken, disconnectToken]
    }

    private func attachToExistingExternalScreen() {
        // Cover the case where the external display was already
        // connected before the coordinator was instantiated (e.g.,
        // user picked an external display in the simulator before
        // launching the app).
        let external = UIScreen.screens.first { $0 !== UIScreen.main }
        if let external {
            attach(to: external)
        }
    }

    private func attach(to screen: UIScreen) {
        guard externalWindow == nil else {
            AppLogger.cast.debug("attach ignored — window already mounted")
            return
        }
        guard let env else {
            AppLogger.cast.error("attach failed — env was deallocated")
            return
        }
        // Role gate (carried-forward M5.5 finding, fixed in M5.4.1
        // touch): cast is a Sevadar-only surface. A Sangat-mode user
        // connecting an external display must not auto-mount the
        // projector window. If they switch to Sevadar mid-session,
        // `modeDidChange(to:)` will attempt the attach.
        guard Self.shouldHostCastSurface(for: env.mode) else {
            AppLogger.cast.info("attach skipped — current mode is \(env.mode.rawValue, privacy: .public); cast is Sevadar-only")
            return
        }

        // Prefer the scene-based path on iOS 17+. iOS spawns a
        // UIWindowScene for each external display when the app has a
        // matching scene configuration; even without an Info.plist
        // scene declaration, we can sometimes find one in
        // `connectedScenes`. If we find one, use it — that's the
        // canonical path. Otherwise, fall back to the deprecated
        // UIWindow(frame:) + screen path.
        let window: UIWindow
        if let scene = matchingExternalWindowScene(for: screen) {
            window = UIWindow(windowScene: scene)
            AppLogger.cast.info("attach: using UIWindowScene path")
        } else {
            window = UIWindow(frame: screen.bounds)
            window.screen = screen
            AppLogger.cast.info("attach: using deprecated UIWindow+UIScreen path")
        }
        window.frame = screen.bounds

        // Wrap the cast root in `CastRootContainer` so the environment
        // modifiers (theme, color scheme) are RE-APPLIED reactively
        // against the live AppEnvironment. UIHostingController snapshots
        // its rootView once; without this wrapper, switching theme on
        // the phone would never propagate to the projector. See the
        // wrapper's doc-comment for the Observation mechanics.
        let host = UIHostingController(rootView: CastRootContainer(env: env))
        host.view.frame = window.bounds
        host.view.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        window.rootViewController = host

        // `makeKeyAndVisible()` is what actually triggers the view
        // controller lifecycle (loadView → viewDidLoad → layout). On a
        // non-main UIScreen, this does NOT steal key-window status from
        // the phone scene — iOS scopes key per scene/screen. Without
        // this call the hosting controller's SwiftUI tree never lays
        // out and the external window renders as a pure-black UIWindow
        // background. The earlier `isHidden = false` approach silently
        // produced exactly that bug.
        window.makeKeyAndVisible()

        self.externalWindow = window
        self.isExternalScreenConnected = true
        self.externalScreenName = "External display"

        AppLogger.cast.info("attached to external screen (\(Int(screen.bounds.width), privacy: .public)x\(Int(screen.bounds.height), privacy: .public))")
    }

    private func detach() {
        externalWindow?.isHidden = true
        externalWindow?.rootViewController = nil
        externalWindow = nil
        isExternalScreenConnected = false
        externalScreenName = nil
        AppLogger.cast.info("detached from external screen")
    }

    /// Locate a `UIWindowScene` already associated with the given
    /// screen, if iOS spawned one. Returns nil when the app's
    /// Info.plist doesn't declare an external scene configuration —
    /// in that case the caller falls back to the legacy UIWindow path.
    private func matchingExternalWindowScene(for screen: UIScreen) -> UIWindowScene? {
        for scene in UIApplication.shared.connectedScenes {
            guard let windowScene = scene as? UIWindowScene else { continue }
            if windowScene.screen === screen, windowScene !== UIApplication.shared.firstWindowSceneOnMainScreen() {
                return windowScene
            }
        }
        return nil
    }

    #endif
}

#if canImport(UIKit) && os(iOS)
private extension UIApplication {
    /// Helper to exclude the main scene when scanning for an external
    /// `UIWindowScene`. Returns the first scene whose `screen === UIScreen.main`.
    func firstWindowSceneOnMainScreen() -> UIWindowScene? {
        for scene in connectedScenes {
            if let ws = scene as? UIWindowScene, ws.screen === UIScreen.main {
                return ws
            }
        }
        return nil
    }
}
#endif

/// Reactive root for the projector window.
///
/// `UIHostingController(rootView:)` snapshots its `rootView` at
/// construction time and re-renders that snapshot whenever SwiftUI
/// invalidates it. The trick is making SwiftUI invalidate the
/// snapshot when the user changes theme on the phone.
///
/// `AppEnvironment` is `@Observable`. Reading any of its tracked
/// properties inside a view body registers a dependency, so when
/// `env.theme` (which has a `didSet` writing to Preferences) flips,
/// SwiftUI invalidates this container's body. The body then re-
/// evaluates and re-applies the `.environment(\.themeTokens, ...)`
/// and `.preferredColorScheme(...)` modifiers with the *current*
/// theme value, and the child `CastReadingView` re-renders with the
/// new tokens.
///
/// Why a wrapper instead of moving the modifier application to the
/// child: putting the env modifiers directly in `CastSceneCoordinator.
/// attach()` snapshots them at attach-time (the bug the user hit on
/// the projector). The wrapper makes the snapshot-vs-reactive
/// boundary explicit and keeps `CastReadingView` itself unchanged.
struct CastRootContainer: View {

    let env: AppEnvironment

    var body: some View {
        // Read `env.theme` inside the body so SwiftUI registers it
        // as an Observation dependency. Without this read the body
        // wouldn't invalidate on theme changes.
        let theme = env.theme

        return CastReadingView()
            .environment(env)
            .environment(\.theme, theme)
            .environment(\.themeTokens, theme.tokens)
            .preferredColorScheme(theme.isDark ? .dark : .light)
    }
}
