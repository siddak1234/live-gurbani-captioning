//  ContentView.swift
//
//  Minimal UI for the Sewadar:
//    - Big "Listen" / "Stop" button (mic toggle)
//    - Status banner: listening / tentative shabad / committed shabad
//    - Current line in big Gurmukhi
//    - Confidence indicator
//    - Reset button (drop committed shabad, listen again)
//
//  Production polish (raag info, history scroll, manual shabad picker) lives
//  in M6+. This is the first-light version that proves the wire works.

import SwiftUI
import GurbaniCaptioning

struct ContentView: View {
    @StateObject private var vm = CaptionViewModel()

    var body: some View {
        VStack(spacing: 24) {
            statusBanner
            Spacer()
            lineDisplay
            Spacer()
            controls
        }
        .padding(32)
        .task {
            await vm.prepare()
        }
    }

    private var statusBanner: some View {
        HStack {
            Circle()
                .fill(vm.stateColor)
                .frame(width: 14, height: 14)
            Text(vm.stateLabel)
                .font(.headline)
            Spacer()
            if let conf = vm.currentConfidence {
                Text(String(format: "%.0f", conf))
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private var lineDisplay: some View {
        VStack(spacing: 12) {
            if let line = vm.currentLineGurmukhi {
                Text(line)
                    .font(.system(size: 36, weight: .medium, design: .serif))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            } else {
                Text("ਸ੍ਰਵਣ ਕਰੋ")
                    .font(.system(size: 28, weight: .light, design: .serif))
                    .foregroundStyle(.secondary)
            }
            if let verseId = vm.currentVerseId {
                Text(verseId)
                    .font(.caption.monospaced())
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity, minHeight: 200)
    }

    private var controls: some View {
        HStack(spacing: 20) {
            Button {
                Task { await vm.toggleListening() }
            } label: {
                Label(vm.isRunning ? "Stop" : "Listen",
                      systemImage: vm.isRunning ? "stop.circle.fill" : "mic.circle.fill")
                    .font(.title2.weight(.semibold))
                    .frame(minWidth: 160)
                    .padding(.vertical, 12)
            }
            .buttonStyle(.borderedProminent)

            Button {
                vm.reset()
            } label: {
                Label("Reset", systemImage: "arrow.counterclockwise")
                    .frame(minWidth: 100)
                    .padding(.vertical, 12)
            }
            .buttonStyle(.bordered)
            .disabled(!vm.isCommitted)
        }
    }
}

// MARK: - View model

@MainActor
final class CaptionViewModel: ObservableObject {

    @Published var isRunning = false
    @Published var stateLabel = "Idle"
    @Published var stateColor: Color = .secondary
    @Published var currentLineGurmukhi: String?
    @Published var currentVerseId: String?
    @Published var currentConfidence: Double?
    @Published var isCommitted = false
    @Published var lastError: String?

    private var engine: CaptionEngine?
    private var corpus: ShabadCorpus?

    func prepare() async {
        do {
            let corpus = try ShabadCorpus.loadFromBundle()
            self.corpus = corpus

            let cfg = CaptionEngine.Config(
                modelPath: "surt-small-v3-kirtan",   // will be looked up in bundle / WhisperKit cache
                language: "punjabi",
                chunkSeconds: 5.0
            )
            let eng = CaptionEngine(config: cfg, corpus: corpus)
            eng.delegate = CaptionEngineBridge(viewModel: self)
            try await eng.prepare()
            self.engine = eng
        } catch {
            self.lastError = error.localizedDescription
        }
    }

    func toggleListening() async {
        guard let engine else { return }
        if isRunning {
            engine.stop()
            isRunning = false
        } else {
            do {
                try await engine.start()
                isRunning = true
            } catch {
                lastError = error.localizedDescription
                isRunning = false
            }
        }
    }

    func reset() {
        engine?.resetShabad()
        currentLineGurmukhi = nil
        currentVerseId = nil
        currentConfidence = nil
        isCommitted = false
    }
}

/// Delegate adapter — keeps the view model out of the public framework surface.
final class CaptionEngineBridge: CaptionEngineDelegate {
    weak var viewModel: CaptionViewModel?

    init(viewModel: CaptionViewModel) {
        self.viewModel = viewModel
    }

    func captionEngine(_ engine: CaptionEngine, didUpdate guess: LineGuess?) {
        Task { @MainActor in
            guard let guess, let corpus = viewModel?.engine.flatMap({ _ in viewModel?.corpus }) else {
                return
            }
            let line = corpus.line(shabadId: guess.shabadId, lineIdx: guess.lineIdx)
            viewModel?.currentLineGurmukhi = line?.banidbGurmukhi
            viewModel?.currentVerseId = line?.verseId
            viewModel?.currentConfidence = guess.confidence
        }
    }

    func captionEngine(_ engine: CaptionEngine, didChangeState state: ShabadState) {
        Task { @MainActor in
            switch state {
            case .listening:
                viewModel?.stateLabel = "Listening…"
                viewModel?.stateColor = .gray
                viewModel?.isCommitted = false
            case .tentative(let sid):
                viewModel?.stateLabel = "Tentative #\(sid)"
                viewModel?.stateColor = .orange
                viewModel?.isCommitted = false
            case .committed(let sid):
                viewModel?.stateLabel = "Committed #\(sid)"
                viewModel?.stateColor = .green
                viewModel?.isCommitted = true
            }
        }
    }

    func captionEngine(_ engine: CaptionEngine, didEncounterError error: Error) {
        Task { @MainActor in
            viewModel?.lastError = error.localizedDescription
        }
    }
}

#Preview {
    ContentView()
}
