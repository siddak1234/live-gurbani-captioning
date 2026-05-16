//  FuzzyMatcher.swift
//
//  Swift port of the Python matcher.py blend: token-sort-ratio + WRatio.
//  Operates on transliterated Gurmukhi (rapidfuzz operates on the same).
//  No external dependencies.
//
//  Algorithm (mirrors rapidfuzz.fuzz.token_sort_ratio + WRatio):
//    - normalize: lowercase, strip non-alphanumeric, fold to ASCII
//    - token_sort_ratio: split → sort tokens alphabetically → join → Jaro-Winkler
//    - WRatio: weighted blend of multiple similarity ratios
//    - blend: weighted sum, default 0.5 * token_sort + 0.5 * WRatio (matches v3.2)
//
//  Parity gate (M5.3 audit): scoring a fixture chunk against a fixture line
//  list in Swift must produce the same top-1 index as the Python matcher
//  on ≥90% of fixtures. Fixtures live in Tests/GurbaniCaptioningTests/Fixtures/.

import Foundation

public struct LineMatch: Equatable {
    public let lineIdx: Int
    public let score: Double
    public let runnerUpScore: Double

    public init(lineIdx: Int, score: Double, runnerUpScore: Double) {
        self.lineIdx = lineIdx
        self.score = score
        self.runnerUpScore = runnerUpScore
    }
}

public enum FuzzyMatcher {

    // MARK: - Public API

    /// Score one chunk of ASR text against every candidate line. Returns score per line (0-100).
    public static func scoreChunk(
        _ chunkText: String,
        against lines: [ShabadLine],
        ratio: ScoringRatio = .blend,
        blend: BlendWeights = .canonical
    ) -> [Double] {
        let normalizedChunk = normalize(chunkText)
        if normalizedChunk.isEmpty {
            return Array(repeating: 0.0, count: lines.count)
        }
        return lines.map { line in
            score(normalizedChunk, against: normalize(line.transliterationEnglish ?? line.banidbGurmukhi),
                  ratio: ratio, blend: blend)
        }
    }

    /// Pick the best line for a chunk.
    public static func matchChunk(
        _ chunkText: String,
        against lines: [ShabadLine],
        scoreThreshold: Double = 0.0,
        marginThreshold: Double = 0.0,
        ratio: ScoringRatio = .blend,
        blend: BlendWeights = .canonical
    ) -> LineMatch? {
        let scores = scoreChunk(chunkText, against: lines, ratio: ratio, blend: blend)
        guard let topIdx = scores.indices.max(by: { scores[$0] < scores[$1] }) else { return nil }
        let topScore = scores[topIdx]
        if topScore < scoreThreshold { return nil }

        // Runner-up for margin gating.
        let runnerUp = scores.enumerated()
            .filter { $0.offset != topIdx }
            .max(by: { $0.element < $1.element })?
            .element ?? 0.0
        if (topScore - runnerUp) < marginThreshold { return nil }

        return LineMatch(lineIdx: topIdx, score: topScore, runnerUpScore: runnerUp)
    }

    // MARK: - Scoring primitives

    public enum ScoringRatio {
        case ratio            // Levenshtein normalized similarity
        case tokenSortRatio   // sort tokens then compare
        case wRatio           // rapidfuzz-style weighted ratio
        case blend            // weighted blend per BlendWeights
    }

    public struct BlendWeights {
        public let tokenSortRatio: Double
        public let wRatio: Double

        public init(tokenSortRatio: Double, wRatio: Double) {
            self.tokenSortRatio = tokenSortRatio
            self.wRatio = wRatio
        }

        /// Path A v3.2 canonical: 0.5 token_sort_ratio + 0.5 WRatio.
        public static let canonical = BlendWeights(tokenSortRatio: 0.5, wRatio: 0.5)
    }

    /// Score in [0, 100]. Higher = more similar.
    static func score(_ a: String, against b: String,
                      ratio: ScoringRatio, blend: BlendWeights) -> Double {
        if a.isEmpty || b.isEmpty { return 0.0 }
        switch ratio {
        case .ratio:
            return basicRatio(a, b) * 100.0
        case .tokenSortRatio:
            return tokenSortRatio(a, b) * 100.0
        case .wRatio:
            return wRatio(a, b)
        case .blend:
            let ts = tokenSortRatio(a, b) * 100.0
            let wr = wRatio(a, b)
            return blend.tokenSortRatio * ts + blend.wRatio * wr
        }
    }

    /// Normalize: lowercase, fold to ASCII (best-effort), strip punctuation.
    public static func normalize(_ s: String) -> String {
        // ASCII fold via NFKD decomposition + diacritic strip (best-effort).
        let lower = s.lowercased()
        let folded = lower.folding(options: [.diacriticInsensitive, .widthInsensitive], locale: nil)
        var out = ""
        out.reserveCapacity(folded.count)
        for c in folded {
            if c.isLetter || c.isNumber || c == " " {
                out.append(c)
            } else {
                out.append(" ")
            }
        }
        // Collapse multi-spaces and trim.
        return out.split(separator: " ").joined(separator: " ")
    }

    // MARK: - Ratios

    /// Jaro-Winkler distance, returns similarity in [0.0, 1.0].
    static func jaroWinkler(_ a: String, _ b: String, p: Double = 0.1, maxL: Int = 4) -> Double {
        let aChars = Array(a)
        let bChars = Array(b)
        let la = aChars.count, lb = bChars.count
        if la == 0 && lb == 0 { return 1.0 }
        if la == 0 || lb == 0 { return 0.0 }

        let matchDistance = max(la, lb) / 2 - 1
        var aMatches = Array(repeating: false, count: la)
        var bMatches = Array(repeating: false, count: lb)
        var matches = 0
        var transpositions = 0

        for i in 0..<la {
            let start = max(0, i - matchDistance)
            let end = min(i + matchDistance + 1, lb)
            for j in start..<end {
                if bMatches[j] || aChars[i] != bChars[j] { continue }
                aMatches[i] = true
                bMatches[j] = true
                matches += 1
                break
            }
        }
        if matches == 0 { return 0.0 }

        var k = 0
        for i in 0..<la where aMatches[i] {
            while !bMatches[k] { k += 1 }
            if aChars[i] != bChars[k] { transpositions += 1 }
            k += 1
        }
        let m = Double(matches)
        let jaro = (m / Double(la) + m / Double(lb) + (m - Double(transpositions) / 2.0) / m) / 3.0

        // Winkler boost for common prefix up to maxL.
        var prefix = 0
        for i in 0..<min(min(la, lb), maxL) {
            if aChars[i] == bChars[i] { prefix += 1 } else { break }
        }
        return jaro + Double(prefix) * p * (1.0 - jaro)
    }

    /// Basic Levenshtein-based ratio (rapidfuzz's `ratio`).
    static func basicRatio(_ a: String, _ b: String) -> Double {
        let aChars = Array(a)
        let bChars = Array(b)
        let la = aChars.count, lb = bChars.count
        if la == 0 && lb == 0 { return 1.0 }
        let dist = levenshtein(aChars, bChars)
        return 1.0 - Double(dist) / Double(la + lb)
    }

    static func levenshtein(_ a: [Character], _ b: [Character]) -> Int {
        let la = a.count, lb = b.count
        if la == 0 { return lb }
        if lb == 0 { return la }
        var prev = Array(0...lb)
        var curr = Array(repeating: 0, count: lb + 1)
        for i in 1...la {
            curr[0] = i
            for j in 1...lb {
                let cost = a[i - 1] == b[j - 1] ? 0 : 1
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            }
            swap(&prev, &curr)
        }
        return prev[lb]
    }

    /// rapidfuzz's `token_sort_ratio`: split → sort → join → Jaro-Winkler.
    static func tokenSortRatio(_ a: String, _ b: String) -> Double {
        let aTokens = a.split(separator: " ").map(String.init).sorted()
        let bTokens = b.split(separator: " ").map(String.init).sorted()
        let aSorted = aTokens.joined(separator: " ")
        let bSorted = bTokens.joined(separator: " ")
        return jaroWinkler(aSorted, bSorted)
    }

    /// rapidfuzz's `WRatio`: weighted blend of several substring/Jaro-Winkler comparisons.
    /// This is an approximation; rapidfuzz's exact WRatio uses partial_ratio + token_set
    /// with length-dependent weights. We use a stable simpler blend: 60% jaro-winkler on
    /// normalized + 40% token-set-ratio. Parity-tested against rapidfuzz on our fixtures.
    static func wRatio(_ a: String, _ b: String) -> Double {
        let jw = jaroWinkler(a, b)
        let ts = tokenSetRatio(a, b)
        return (0.6 * jw + 0.4 * ts) * 100.0
    }

    /// rapidfuzz's `token_set_ratio`: split → set → align via intersection.
    static func tokenSetRatio(_ a: String, _ b: String) -> Double {
        let aSet = Set(a.split(separator: " ").map(String.init))
        let bSet = Set(b.split(separator: " ").map(String.init))
        let intersection = aSet.intersection(bSet)
        let diffAB = aSet.subtracting(bSet)
        let diffBA = bSet.subtracting(aSet)

        let inter = intersection.sorted().joined(separator: " ")
        let t1 = (inter + " " + diffAB.sorted().joined(separator: " ")).trimmingCharacters(in: .whitespaces)
        let t2 = (inter + " " + diffBA.sorted().joined(separator: " ")).trimmingCharacters(in: .whitespaces)

        let r1 = jaroWinkler(inter, t1)
        let r2 = jaroWinkler(inter, t2)
        let r3 = jaroWinkler(t1, t2)
        return max(r1, r2, r3)
    }
}
