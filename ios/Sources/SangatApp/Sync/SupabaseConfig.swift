//
//  SupabaseConfig.swift
//  GurbaniCaptioningApp · SangatApp · Sync
//
//  Created for the Sangat iOS app, corrections feedback loop Phase 4
//  (Sync engine). See docs/corrections_feedback_loop_plan.md.
//
//  Connection details for the dedicated `gurbani-captioning` Supabase project
//  (separate from Autom8x). The **publishable** key is safe to embed in the
//  client — it ships in every app binary by design, and Row-Level Security is
//  the actual protection (anon can INSERT only; reads are service_role-only).
//  The service_role/secret key MUST NEVER appear in the app.

import Foundation

public enum SupabaseConfig {
    /// REST base URL for the dedicated project.
    public static let url = "https://mlovvqoiuuihmymkypfm.supabase.co"

    /// Publishable (anon) API key — safe to embed; RLS is the protection.
    public static let publishableKey = "sb_publishable_t0Y2HZh1Kko6-uPovQItdA_fVVUzUyU"

    /// REST endpoint for the corrections table.
    public static var correctionsEndpoint: String { "\(url)/rest/v1/corrections" }

    /// Edge Function endpoint for the right-to-delete path.
    public static var deleteFunctionEndpoint: String { "\(url)/functions/v1/delete-my-data" }
}
