//  ShabadStateMachine.swift
//
//  Live blind-mode shabad identification + commit/reset state machine.
//
//  Swift port of:
//    - src/shabad_id.py     (per-chunk global match + chunk-vote aggregation)
//    - src/smoother.py      (stay-bias logic for the current line)
//    - src/engine.py        (top-level commit logic)
//
//  Lifecycle:
//    1. Sewadar opens the app and starts streaming audio. ``state = .listening``.
//    2. As ASR chunks arrive (from CaptionEngine), each chunk is scored against
//       every shabad in the corpus; the top-1 (shabad_id, line_idx) is tallied.
//    3. Once N consecutive chunks vote for the same shabad with enough
//       confidence margin, commit: ``state = .committed(shabadId)``. Future
//       chunks are only scored against that one shabad's lines.
//    4. Sewadar can hit "Reset" → back to ``.listening``.
//
//  This mirrors what bani.karanbirsingh.com does conceptually but with our
//  scorer and our corpus.

import Foundation

public enum ShabadState: Equatable {
    case listening                    // collecting chunks, no shabad committed yet
    case tentative(shabadId: Int)     // gathering evidence for this candidate
    case committed(shabadId: Int)     // confident; only matching this shabad's lines now
}

public struct ShabadCommitConfig {
    /// How many consecutive same-shabad votes before committing.
    public let consecutiveVotesToCommit: Int
    /// Top-1 vs runner-up score gap required for a vote to count.
    public let minVoteMargin: Double
    /// Minimum top-1 score for a vote to count at all.
    public let minVoteScore: Double
    /// Max chunks of audio to keep in the rolling buffer for identification.
    public let bufferChunks: Int

    public init(consecutiveVotesToCommit: Int = 3,
                minVoteMargin: Double = 5.0,
                minVoteScore: Double = 40.0,
                bufferChunks: Int = 12) {
        self.consecutiveVotesToCommit = consecutiveVotesToCommit
        self.minVoteMargin = minVoteMargin
        self.minVoteScore = minVoteScore
        self.bufferChunks = bufferChunks
    }

    public static let `default` = ShabadCommitConfig()
}

/// Single "current line" guess produced by the state machine after each chunk.
public struct LineGuess: Equatable {
    public let chunk: AsrChunk
    public let shabadId: Int
    public let lineIdx: Int
    public let confidence: Double
    public let isCommitted: Bool

    public init(chunk: AsrChunk,
                shabadId: Int,
                lineIdx: Int,
                confidence: Double,
                isCommitted: Bool) {
        self.chunk = chunk
        self.shabadId = shabadId
        self.lineIdx = lineIdx
        self.confidence = confidence
        self.isCommitted = isCommitted
    }
}

public final class ShabadStateMachine {

    public private(set) var state: ShabadState
    private let corpus: ShabadCorpus
    private let config: ShabadCommitConfig

    private var voteBuffer: [Int] = []         // recent top-shabad votes
    private var lastLineGuess: LineGuess?      // smoother stay-bias context

    public init(corpus: ShabadCorpus, config: ShabadCommitConfig = .default) {
        self.corpus = corpus
        self.config = config
        self.state = .listening
    }

    // MARK: - Session control

    /// Sewadar pressed Reset, or session is starting fresh.
    public func reset() {
        state = .listening
        voteBuffer.removeAll(keepingCapacity: true)
        lastLineGuess = nil
    }

    /// Override blind detection — Sewadar selected a shabad manually.
    public func commit(shabadId: Int) {
        state = .committed(shabadId: shabadId)
        voteBuffer.removeAll(keepingCapacity: true)
        lastLineGuess = nil
    }

    // MARK: - Chunk ingestion

    /// Process one ASR chunk. Returns the current best line guess (may be nil
    /// during listening if no candidate scored high enough).
    public func processChunk(_ chunk: AsrChunk) -> LineGuess? {
        switch state {
        case .committed(let sid):
            return matchAgainstCommittedShabad(sid, chunk: chunk)

        case .listening, .tentative:
            return voteForShabad(chunk: chunk)
        }
    }

    // MARK: - Blind mode (listening / tentative)

    private func voteForShabad(chunk: AsrChunk) -> LineGuess? {
        // Score this chunk against every shabad's lines; keep the top per-shabad
        // line score. The shabad with the best top-line wins this vote.
        var bestPerShabad: [(shabadId: Int, lineIdx: Int, score: Double)] = []
        for shabadId in corpus.allShabadIds {
            let lines = corpus.lines(for: shabadId)
            guard let match = FuzzyMatcher.matchChunk(chunk.text, against: lines,
                                                     scoreThreshold: 0.0,
                                                     marginThreshold: 0.0) else { continue }
            bestPerShabad.append((shabadId, match.lineIdx, match.score))
        }
        bestPerShabad.sort { $0.score > $1.score }
        guard let top = bestPerShabad.first else { return nil }

        let runnerUp = bestPerShabad.dropFirst().first?.score ?? 0.0
        let margin = top.score - runnerUp

        // Only count this as a vote if confidence is sufficient.
        if top.score >= config.minVoteScore && margin >= config.minVoteMargin {
            voteBuffer.append(top.shabadId)
            if voteBuffer.count > config.bufferChunks {
                voteBuffer.removeFirst(voteBuffer.count - config.bufferChunks)
            }
            // Check for commit condition.
            let tail = voteBuffer.suffix(config.consecutiveVotesToCommit)
            if tail.count >= config.consecutiveVotesToCommit,
               Set(tail).count == 1,
               let sid = tail.first {
                state = .committed(shabadId: sid)
                return LineGuess(chunk: chunk, shabadId: sid,
                                 lineIdx: top.lineIdx, confidence: top.score,
                                 isCommitted: true)
            }
            state = .tentative(shabadId: top.shabadId)
        }

        return LineGuess(chunk: chunk, shabadId: top.shabadId,
                         lineIdx: top.lineIdx, confidence: top.score,
                         isCommitted: false)
    }

    // MARK: - Committed mode (oracle on the chosen shabad)

    private func matchAgainstCommittedShabad(_ shabadId: Int, chunk: AsrChunk) -> LineGuess? {
        let lines = corpus.lines(for: shabadId)
        guard !lines.isEmpty else { return nil }

        // Stay-bias: if the previous chunk landed on line N and this chunk's
        // top-1 is within the bias margin of line N's score, prefer staying.
        // Mirrors src/smoother.py:smooth_with_stay_bias logic.
        let scores = FuzzyMatcher.scoreChunk(chunk.text, against: lines)
        guard let topIdx = scores.indices.max(by: { scores[$0] < scores[$1] }) else { return nil }
        var chosenIdx = topIdx
        let stayBias = 6.0

        if let prev = lastLineGuess, prev.shabadId == shabadId {
            let prevLineIdx = prev.lineIdx
            if prevLineIdx < scores.count {
                let prevScore = scores[prevLineIdx]
                let topScore = scores[topIdx]
                if (topScore - prevScore) < stayBias {
                    chosenIdx = prevLineIdx
                }
            }
        }

        let guess = LineGuess(chunk: chunk, shabadId: shabadId,
                              lineIdx: chosenIdx, confidence: scores[chosenIdx],
                              isCommitted: true)
        lastLineGuess = guess
        return guess
    }
}
