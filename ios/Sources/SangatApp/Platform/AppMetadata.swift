//
//  AppMetadata.swift
//  GurbaniCaptioningApp · SangatApp · Platform
//
//  Created for the Sangat iOS app, corrections feedback loop Phase 0
//  (Contracts & consent scaffolding). See
//  docs/corrections_feedback_loop_plan.md.
//
//  Version stamping for the corrections feedback loop. Every uploaded
//  correction carries the app + model + schema versions so the training side
//  knows which model made the mistake and which wire schema to parse.
//
//  Phase 0 status: defined and tested, consumed by nobody yet — the upload
//  envelope (`CorrectionEnvelope`) defaults its metadata fields from here, and
//  the sync engine (Phase 4) sends them. No behavior change today.

import Foundation

/// Static, read-only metadata describing this build and the bundled model.
public enum AppMetadata {

    /// User-facing app version (`CFBundleShortVersionString`, e.g. "0.1.0").
    public static var appVersion: String {
        (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "0.0.0"
    }

    /// Build number (`CFBundleVersion`).
    public static var buildNumber: String {
        (Bundle.main.infoDictionary?["CFBundleVersion"] as? String) ?? "0"
    }

    /// Identifier for the bundled Core ML model + quantization. **Bump this
    /// whenever the bundled `.mlmodelc` set changes** so corrections are
    /// attributable to the exact model that produced them. Today's bundle is
    /// the 6-bit uniform export of surt-small-v3 (see CLAUDE.md M5.7d).
    public static let modelVersion: String = "surt-small-v3-kirtan@6bit"

    /// Wire-schema version for `CorrectionEnvelope`. Increment on any
    /// breaking change to the uploaded shape so the server can route versions.
    public static let schemaVersion: Int = 1
}
