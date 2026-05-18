//
//  AudioPermissions.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Thin wrapper over `AVAudioApplication.requestRecordPermission` (the
//  modern, iOS-17 API replacing the deprecated `AVAudioSession` variant).
//  Used by the onboarding mic-permission card and any code that needs to
//  check current status.
//
//  This is the *only* place in the app that calls the AVFoundation
//  permission APIs directly — all UI goes through this seam.

import Foundation

#if canImport(AVFoundation)
import AVFoundation
#endif

#if canImport(UIKit) && os(iOS)
import UIKit
#endif

/// Current state of microphone permission.
public enum MicPermissionStatus: Sendable {
    /// User has not been asked yet.
    case notDetermined
    /// User granted.
    case granted
    /// User denied. Must send them to Settings.app to re-enable.
    case denied
    /// Restricted (e.g. parental controls).
    case restricted
    /// `AVFoundation` is unavailable on this build (mac unit tests etc).
    case unavailable
}

@MainActor
public enum AudioPermissions {

    /// Current mic permission status.
    public static var status: MicPermissionStatus {
        #if canImport(AVFoundation) && (os(iOS) || os(macOS))
        if #available(iOS 17, macOS 14, *) {
            switch AVAudioApplication.shared.recordPermission {
            case .granted: return .granted
            case .denied: return .denied
            case .undetermined: return .notDetermined
            @unknown default: return .notDetermined
            }
        } else {
            // Fallback for older OS — not expected since deployment target is 17/14.
            return .unavailable
        }
        #else
        return .unavailable
        #endif
    }

    /// Prompt the user for mic permission. Returns the resulting status.
    /// Safe to call when already granted — the system call returns
    /// immediately in that case.
    @discardableResult
    public static func request() async -> MicPermissionStatus {
        #if canImport(AVFoundation) && (os(iOS) || os(macOS))
        if #available(iOS 17, macOS 14, *) {
            let granted = await AVAudioApplication.requestRecordPermission()
            AppLogger.app.info("Mic permission request — granted: \(granted, privacy: .public)")
            return granted ? .granted : .denied
        } else {
            return .unavailable
        }
        #else
        return .unavailable
        #endif
    }

    /// Open the Settings.app pane for this app. Used when mic
    /// permission has been denied via the system dialog — iOS will
    /// not re-prompt in that case, so the only path to grant is the
    /// OS Settings UI. Returns true if the open call was issued,
    /// false on platforms where it isn't available (macOS test host).
    ///
    /// ## Depth this can reach
    ///
    /// `UIApplication.openSettingsURLString` is the **deepest** link
    /// Apple permits via public API. On a real device it lands at
    /// `Settings → Sangat`, where the Microphone toggle is rendered
    /// inline (one tap from there). There is no public API to deep-
    /// link directly to `Settings → Sangat → Microphone`, or to
    /// `Settings → Privacy → Microphone → Sangat`. Apple ships
    /// per-permission deep-links only for notifications
    /// (`openNotificationSettingsURLString`, iOS 15.4+); microphone
    /// has no equivalent through iOS 26.
    ///
    /// ## Why we don't use `prefs:root=Privacy&path=MICROPHONE`
    ///
    /// That URL works at runtime but is a **private** scheme. Apple
    /// has rejected apps from the App Store for shipping it (the
    /// rejection notice cites non-public APIs and warns of developer
    /// account termination on repeat use). Off-limits.
    ///
    /// ## Simulator caveat
    ///
    /// On the iOS Simulator this URL frequently opens Settings at
    /// the root instead of the app pane — the simulator's
    /// Settings.app doesn't always rebuild its app list after a TCC
    /// decision. The behaviour is real-device-correct; the simulator
    /// bug is well-documented in Apple's forums. Verify on hardware.
    @discardableResult
    public static func openAppSettings() -> Bool {
        #if canImport(UIKit) && os(iOS)
        guard let url = URL(string: UIApplication.openSettingsURLString) else { return false }
        UIApplication.shared.open(url)
        AppLogger.app.info("AudioPermissions: opened Settings.app for the app's pane")
        return true
        #else
        return false
        #endif
    }
}
