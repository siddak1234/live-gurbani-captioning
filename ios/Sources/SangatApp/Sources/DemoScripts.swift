//
//  DemoScripts.swift
//  GurbaniCaptioningApp
//
//  Created for the Sangat iOS app, M5.1 (Foundations).
//
//  Pre-canned timelines for `DemoCaptionSource`. Each script encodes the
//  engine's progression from `.listening → .tentative → .committed` with
//  realistic delays and runner-up populations.
//
//  Adding a new script
//  -------------------
//  1. Define the shabad's lines as a `[ShabadLine]` literal (or reference one
//     from `PreviewData`).
//  2. List the timeline as `DemoStep`s.
//  3. Add a static accessor at the bottom of `DemoScript`.
//
//  Demo data is the *only* place we hardcode shabad text. Production reads
//  from `ShabadCorpus` (loaded from the bundled `shabads.json`).

import Foundation
import GurbaniCaptioning

// MARK: - Script types

/// One discrete event in a demo timeline.
public struct DemoStep {
    /// Seconds to wait *before* applying this step.
    public let delay: TimeInterval
    /// New engine state, or `nil` to leave unchanged.
    public let state: ShabadState?
    /// New current-line guess, or `nil` to leave unchanged.
    public let guess: LineGuess?
    /// If true, clear the current guess regardless of `guess`.
    public let clearsGuess: Bool
    /// New runner-up list, or `nil` to leave unchanged.
    public let runnerUps: [LineGuess]?

    public init(
        delay: TimeInterval,
        state: ShabadState? = nil,
        guess: LineGuess? = nil,
        clearsGuess: Bool = false,
        runnerUps: [LineGuess]? = nil
    ) {
        self.delay = delay
        self.state = state
        self.guess = guess
        self.clearsGuess = clearsGuess
        self.runnerUps = runnerUps
    }
}

/// A named timeline of `DemoStep`s.
public struct DemoScript {
    public let name: String
    public let shabadId: Int
    public let steps: [DemoStep]

    public init(name: String, shabadId: Int, steps: [DemoStep]) {
        self.name = name
        self.shabadId = shabadId
        self.steps = steps
    }
}

// MARK: - Script factory helpers

/// Convenience to build a `LineGuess` for demo steps. Real `AsrChunk`s and
/// confidences are stand-ins; we just need the shape to match.
private func makeGuess(
    shabadId: Int,
    lineIdx: Int,
    confidence: Double,
    isCommitted: Bool,
    chunkStart: TimeInterval = 0,
    chunkEnd: TimeInterval = 5,
    chunkText: String = ""
) -> LineGuess {
    LineGuess(
        chunk: AsrChunk(start: chunkStart, end: chunkEnd, text: chunkText),
        shabadId: shabadId,
        lineIdx: lineIdx,
        confidence: confidence,
        isCommitted: isCommitted
    )
}

// MARK: - Canonical script: Tati Vao Na Lagai

extension DemoScript {

    /// The default demo: Tati Vao Na Lagai (Guru Arjan, Bilaaval, Ang 819).
    /// Mirrors what a real session would look like:
    ///   - 0.0s  listening
    ///   - 5.0s  tentative (first guess, low confidence)
    ///   - 8.0s  tentative (3-of-5 chunks agree)
    ///   - 12.0s committed
    ///   - 18.0s line 1
    ///   - 24.0s line 2
    ///   - 30.0s line 3
    ///   - 36.0s line 4
    ///   - 42.0s line 5
    public static let tatiVaoNaLagai = DemoScript(
        name: "Tati Vao Na Lagai",
        shabadId: 1789,
        steps: [
            // initial listening — empty
            DemoStep(delay: 0.0, state: .listening, clearsGuess: true, runnerUps: []),
            // first faint candidates appear
            DemoStep(
                delay: 5.0,
                state: .tentative(shabadId: 1789),
                guess: makeGuess(shabadId: 1789, lineIdx: 0, confidence: 62.0, isCommitted: false),
                runnerUps: [
                    makeGuess(shabadId: 4501, lineIdx: 0, confidence: 48.0, isCommitted: false),
                    makeGuess(shabadId: 941,  lineIdx: 0, confidence: 41.0, isCommitted: false),
                ]
            ),
            // confidence building
            DemoStep(
                delay: 3.0,
                state: .tentative(shabadId: 1789),
                guess: makeGuess(shabadId: 1789, lineIdx: 0, confidence: 74.0, isCommitted: false)
            ),
            // committed
            DemoStep(
                delay: 4.0,
                state: .committed(shabadId: 1789),
                guess: makeGuess(shabadId: 1789, lineIdx: 0, confidence: 87.2, isCommitted: true),
                runnerUps: []
            ),
            // advance through lines
            DemoStep(delay: 6.0, guess: makeGuess(shabadId: 1789, lineIdx: 1, confidence: 88.4, isCommitted: true)),
            DemoStep(delay: 6.0, guess: makeGuess(shabadId: 1789, lineIdx: 2, confidence: 89.1, isCommitted: true)),
            DemoStep(delay: 6.0, guess: makeGuess(shabadId: 1789, lineIdx: 3, confidence: 86.7, isCommitted: true)),
            DemoStep(delay: 6.0, guess: makeGuess(shabadId: 1789, lineIdx: 4, confidence: 91.0, isCommitted: true)),
            DemoStep(delay: 6.0, guess: makeGuess(shabadId: 1789, lineIdx: 5, confidence: 88.3, isCommitted: true)),
        ]
    )

    /// Short script that lands on `.committed` quickly — useful for screenshot
    /// runs and tests that just want a committed shabad without waiting.
    public static let quickCommit = DemoScript(
        name: "Quick commit",
        shabadId: 1789,
        steps: [
            DemoStep(delay: 0.0, state: .listening, clearsGuess: true),
            DemoStep(
                delay: 0.5,
                state: .committed(shabadId: 1789),
                guess: makeGuess(shabadId: 1789, lineIdx: 1, confidence: 88.4, isCommitted: true)
            ),
        ]
    )
}
