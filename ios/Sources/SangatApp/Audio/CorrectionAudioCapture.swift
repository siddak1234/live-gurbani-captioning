//
//  CorrectionAudioCapture.swift
//  GurbaniCaptioningApp · SangatApp · Audio
//
//  Created for the Sangat iOS app, corrections feedback loop Phase 2b
//  (correction-emission audio wiring). See docs/corrections_feedback_loop_plan.md.
//
//  The decision seam between a correction and its audio clip. Kept as a tiny
//  pure helper (rather than inline in RootView) so the consent + capability
//  gating is unit-testable without a SwiftUI host or a live WhisperKit engine:
//  the actual capture is delegated to whatever `AudioClipCapturing` is passed in.

import Foundation

/// Captures a correction's audio clip iff the user opted in *and* the active
/// source can provide audio. Returns the local clip path, or nil.
public enum CorrectionAudioCapture {

    /// Seconds of audio preceding the correction to snapshot. Whisper-scale
    /// context window; bounded in practice by WhisperKit's buffer history.
    public static let windowSeconds: Double = 30

    /// - Parameters:
    ///   - id: correction id; names the clip so it joins to its record.
    ///   - optedIn: `Preferences.audioCaptureOptIn`.
    ///   - capturer: the active source if it supports audio (`env.captionSource
    ///     as? AudioClipCapturing`); nil for the demo source / opted-out engine.
    ///   - writer: the app's clip writer (`env.audioClipWriter`), or nil if the
    ///     store couldn't be created.
    /// - Returns: local clip path, or nil if any gate fails.
    @MainActor
    public static func capture(
        id: UUID,
        optedIn: Bool,
        capturer: (any AudioClipCapturing)?,
        writer: AudioClipWriter?
    ) -> String? {
        guard optedIn, let capturer, let writer else { return nil }
        return capturer.captureCorrectionClip(id: id, seconds: windowSeconds, writer: writer)
    }
}
