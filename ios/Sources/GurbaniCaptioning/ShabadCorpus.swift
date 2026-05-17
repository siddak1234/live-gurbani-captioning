//  ShabadCorpus.swift
//
//  Loads the SGGS corpus from the bundled JSON resource. Generated on the
//  desktop by scripts/build_ios_corpus.py and placed in
//  Sources/GurbaniCaptioning/Resources/shabads.json before building.
//
//  The format mirrors corpus_cache/*.json from the Python side: each shabad
//  is one JSON object with shabad_id + lines[]. The bundled resource is a
//  single file holding *all* shabads as a top-level array, lazily indexed.

import Foundation

public enum ShabadCorpusError: Error, LocalizedError {
    case resourceMissing(String)
    case decodingFailed(Error)
    case unknownShabad(Int)

    public var errorDescription: String? {
        switch self {
        case .resourceMissing(let name):
            return "ShabadCorpus: missing bundle resource '\(name)'. Run scripts/build_ios_corpus.py before building."
        case .decodingFailed(let e):
            return "ShabadCorpus: JSON decode failed: \(e.localizedDescription)"
        case .unknownShabad(let id):
            return "ShabadCorpus: shabad_id \(id) not in corpus"
        }
    }
}

public final class ShabadCorpus {

    /// Map shabad_id → Shabad. Backed by the bundled JSON file.
    private let byId: [Int: Shabad]

    public init(byId: [Int: Shabad]) {
        self.byId = byId
    }

    /// Load the default corpus from the package bundle.
    public static func loadFromBundle(resource: String = "shabads",
                                      ext: String = "json",
                                      bundle: Bundle? = nil) throws -> ShabadCorpus {
        let activeBundle = bundle ?? .module
        guard let url = activeBundle.url(forResource: resource, withExtension: ext) else {
            throw ShabadCorpusError.resourceMissing("\(resource).\(ext)")
        }
        let data = try Data(contentsOf: url)
        do {
            let shabads = try JSONDecoder().decode([Shabad].self, from: data)
            var byId: [Int: Shabad] = [:]
            for shabad in shabads {
                byId[shabad.shabadId] = shabad
            }
            return ShabadCorpus(byId: byId)
        } catch {
            throw ShabadCorpusError.decodingFailed(error)
        }
    }

    // MARK: - Lookup

    public var allShabadIds: [Int] { Array(byId.keys) }
    public var count: Int { byId.count }

    public func shabad(_ id: Int) -> Shabad? { byId[id] }

    public func lines(for shabadId: Int) -> [ShabadLine] {
        byId[shabadId]?.lines ?? []
    }

    public func line(shabadId: Int, lineIdx: Int) -> ShabadLine? {
        byId[shabadId]?.lines.first { $0.lineIdx == lineIdx }
    }
}
