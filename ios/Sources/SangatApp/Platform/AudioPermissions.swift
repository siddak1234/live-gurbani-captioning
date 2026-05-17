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
}
