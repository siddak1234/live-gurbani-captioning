//
//  CaptionSourceEvent.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Value type carried on the `CaptionSource.events` stream. Sendable so it
//  crosses isolation boundaries cleanly.

import Foundation
import GurbaniCaptioning

/// Single event published by a `CaptionSource`.
///
/// Cases mirror the state machine's vocabulary so a downstream observer can
/// implement a one-to-one handler without losing information.
public enum CaptionSourceEvent: Sendable {

    /// Engine state changed (e.g. `listening → tentative → committed`).
    case stateChanged(ShabadState)

    /// A new `LineGuess` is available. Will fire on every ASR chunk while
    /// running. `nil` means "no guess this chunk" (e.g. very early listening).
    case guessUpdated(LineGuess?)

    /// Runner-up candidate list updated. Fires after the same chunk that
    /// produced a `guessUpdated`. Ordered high-to-low score.
    case runnerUpsUpdated([LineGuess])

    /// Source-level error — model load failure, audio session denial.
    /// Encoded as `String` because we don't want to leak the concrete error
    /// type through the public surface.
    case error(String)

    /// Source started running (after `start()`).
    case started

    /// Source stopped running (after `stop()` or fatal `.error`).
    case stopped
}
