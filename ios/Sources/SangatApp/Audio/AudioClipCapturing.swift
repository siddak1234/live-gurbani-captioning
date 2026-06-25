//
//  AudioClipCapturing.swift
//  GurbaniCaptioningApp · SangatApp · Audio
//
//  Created for the Sangat iOS app, corrections feedback loop Phase 2
//  (Audio capture). See docs/corrections_feedback_loop_plan.md.
//
//  Capability a `CaptionSource` advertises when it can hand back a clip of the
//  audio just before a correction. Only the live engine has microphone audio,
//  so `LiveCaptionSource` conforms and `DemoCaptionSource` does not — the
//  Phase 2b correction-emission wiring discovers it with `as? AudioClipCapturing`
//  rather than widening the `CaptionSource` protocol.

import Foundation

/// A caption source that can snapshot recent mic audio and persist it as a clip.
@MainActor
public protocol AudioClipCapturing: AnyObject {

    /// Snapshot the last `seconds` of captured audio and write it via `writer`
    /// under the given correction `id`. Returns the local file path, or nil if
    /// no audio is available (not running, opted out upstream, or empty buffer).
    ///
    /// The caller is responsible for the `audioCaptureOptIn` consent gate; this
    /// method assumes consent has already been checked.
    func captureCorrectionClip(id: UUID, seconds: Double, writer: AudioClipWriter) -> String?
}
