# Corrections feedback loop — design & phased plan

**Status:** planning. No app source edited yet. This doc is the source of truth
for the feature; update it as phases land.

**Goal:** when a user corrects the engine (most importantly the Sevadar/Sangat
"Pick shabad" action — the engine got the shabad wrong), durably capture that
correction on-device, and sync it to our Supabase store when the user is
online, so the correction can feed the next `surt-small-v3` fine-tune. Audio is
the trainable asset; text alone only helps the matcher, not the acoustic model.

Mirrors the repo's phase discipline: each phase is **gated** — don't start N+1
until N's success criterion is met or a deliberate pivot is documented. No phase
ships data off-device without explicit, separate consent.

---

## 1. Locked decisions (answered 2026-05-21)

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | On-device store engine | **SwiftData** | Apple-native; app already targets iOS 17. No new dependency. |
| 2 | Supabase access | **`supabase-swift` SDK** | Faster to build than a hand-rolled REST client. |
| 3 | Upload path | **Edge Function gateway** | Server-side validation, dedupe, rate-limiting; client never holds `service_role`. |
| 4 | Audio codec | **AAC** (Opus intended) | Phase 2 finding: `AVAudioFile`'s Opus path ignores `AVEncoderBitRateKey` (~320 KB/30s), while AAC honors it (~120 KB/30s) and meets the budget. Default = AAC (m4a); `.opus` still selectable; transcode server-side if Opus is wanted. Revisit low-bitrate Opus via AVAssetWriter later. |
| 5 | Device identity | **Keychain-persistent** | Stable anonymous `device_id`; survives reinstall; enables dedupe + right-to-delete. |

---

## 2. What we have today (verified)

- **`CorrectionEvent`** ([ios/Sources/SangatApp/Corrections/CorrectionEvent.swift](../ios/Sources/SangatApp/Corrections/CorrectionEvent.swift))
  — rich `Codable` value type: `id`, `timestamp`, `sessionId`, `kind`,
  `predicted` (shabadId, lineIdx, confidence, runnerUps), `groundTruth`
  (shabadId, lineIdx), `engineStateRaw`, `recentChunks` (ASR text+timing),
  `audioBufferPath?`, `audioStart/End`, `notes`.
- **`CorrectionKind`** — `hardNegPos` (committed A, user picked B — our target
  signal), `softPos`, `runnerUpEndorsed`, `lineNudge`, `retroactive`.
- **`CorrectionLog`** protocol (`record` / `recent` / `clear` / `approximateCount`),
  `Sendable`. Swappable in one line at
  [AppEnvironment.swift:152](../ios/Sources/SangatApp/App/AppEnvironment.swift#L152).
  - `NoopCorrectionLog` (discards), `InMemoryCorrectionLog` (NSLock array,
    **evaporates on app kill**). No durable impl exists.
- **Emission** funnels through `recordIfOptedIn` (gated on
  `preferences.correctionsOptIn`, default OFF) at
  [RootView.swift:395](../ios/Sources/SangatApp/App/RootView.swift#L395). Live sites:
  Sevadar picker mismatch → `hardNegPos`
  ([RootView.swift:406](../ios/Sources/SangatApp/App/RootView.swift#L406)),
  Sangat long-press → `hardNegPos`, line nudge → `lineNudge`.
- **`Preferences`** ([Preferences.swift](../ios/Sources/SangatApp/Persistence/Preferences.swift))
  — `UserDefaults` facade; only `correctionsOptIn` exists for this feature.
- **Privacy today:** `audioBufferPath` is always `nil` (no capture). No network
  layer anywhere (no `URLSession`/Supabase/Keychain in `ios/Sources/`). Settings
  copy currently says "we do not upload anything in this build."

### The four gaps between "schema" and "feedback loop"
1. No durable store — corrections lost on app kill.
2. No audio capture — `audioBufferPath` never set; only the engine's already-wrong
   ASR text is retained. **Retraining the acoustic model needs the audio.**
3. No network layer — no backend, queue, reachability, or device identity.
4. No upload consent — sync is "a separate decision behind a separate toggle."

---

## 3. Target architecture

```
┌─────────────────────────── iOS app (SangatApp) ───────────────────────────┐
│  CaptionEngine ──(rolling PCM ring buffer ~30s)──┐                          │
│       │                                          ▼                          │
│  user taps "Pick shabad"                AudioClipWriter ──> <id>.opus (disk) │
│       │                                          │                          │
│       ▼                                          ▼                          │
│  CorrectionEventBuilder ──> DurableCorrectionLog (SwiftData) status=pending │
│                                                  │                          │
│                                          SyncCoordinator                    │
│                                 (NWPathMonitor + background URLSession)      │
└──────────────────────────────────────│─────────────────────────────────────┘
                          anon JWT      ▼   invoke Edge Function (gateway)
┌──────────────────────────────── Supabase ──────────────────────────────────┐
│  Auth (anonymous)                                                            │
│  Edge Function `submit-correction`: validate → dedupe → signed upload URL    │
│  Storage bucket: correction-audio/<device_id>/<correction_id>.opus           │
│  Postgres: devices, corrections, model_versions   (+ RLS)                    │
└──────────────────────────────────────│─────────────────────────────────────┘
                       service_role     ▼  (server only — never on device)
┌──────────────────────── Training side (Python, Mac) ───────────────────────┐
│  scripts/pull_corrections.py → download audio+labels → human QA gate →       │
│  training_data/<batch>/manifest.json → next surt LoRA fine-tune              │
└─────────────────────────────────────────────────────────────────────────────┘
```

Three consent gates, all default OFF: **record** (`correctionsOptIn`, exists),
**audio capture** (`audioCaptureOptIn`, new), **upload** (`uploadOptIn`, new).

### 3a. Frontend module map (new Swift)

| Module (path under `ios/Sources/`) | Responsibility | Phase |
|---|---|---|
| `SangatApp/Corrections/DurableCorrectionLog.swift` | `CorrectionLog` impl over SwiftData; per-record `syncStatus` (pending/uploading/uploaded/failed), `attemptCount`, `lastError` | 1 |
| `GurbaniCaptioning/Audio/AudioRingBuffer.swift` | Thread-safe rolling ~30s PCM window fed by `CaptionEngine` | 2 |
| `SangatApp/Audio/AudioClipWriter.swift` | Encode window → Opus (`AVAudioConverter`), write to app-support, return path | 2 |
| `SangatApp/Sync/SupabaseService.swift` | `supabase-swift` wrapper: anon auth, invoke Edge Function, Storage upload | 4 |
| `SangatApp/Sync/SyncCoordinator.swift` | Outbox drain, `NWPathMonitor`, background `URLSession`, backoff, idempotency | 4 |
| `SangatApp/Sync/Reachability.swift` | Wi-Fi vs cellular gating | 4 |
| `SangatApp/Platform/DeviceIdentity.swift` | Keychain anonymous `device_id` (UUID) | 0 |
| `SangatApp/Platform/AppMetadata.swift` | `appVersion`, `buildNumber`, `modelVersion`, `schemaVersion` | 0 |
| `SangatApp/Persistence/Preferences.swift` (extend) | `uploadOptIn`, `audioCaptureOptIn`, `wifiOnlyUpload` | 0 |
| `SangatApp/Features/Corrections/CorrectionsSettingsView.swift` (extend) | Upload/audio consent toggles; fix "we do not upload" copy; queue status; "Delete my data" | 5 |

### 3b. Backend objects (Supabase) — defined here, applied in Phase 3

Migrations live in `supabase/migrations/` (created in Phase 3).

- **`devices`** — `device_id uuid PK`, `first_seen timestamptz`, `app_version text`, `os_version text`.
- **`corrections`** — mirror of `CorrectionEvent` plus lineage:
  `id uuid PK (client-generated)`, `device_id uuid FK`, `session_id uuid`,
  `kind text`, `predicted_shabad_id int`, `predicted_line_idx int`,
  `confidence double precision`, `runner_ups jsonb`,
  `ground_truth_shabad_id int`, `ground_truth_line_idx int`,
  `engine_state text`, `recent_chunks jsonb`, `audio_path text`,
  `audio_start double precision`, `audio_end double precision`,
  `model_version text`, `schema_version int`, `created_at timestamptz`,
  `export_status text default 'new'`, `review_status text default 'unreviewed'`.
- **`model_versions`** — registry so each correction is attributable to the
  model that erred.
- **Storage bucket** `correction-audio` (private), object key
  `<device_id>/<correction_id>.<ext>` (`.m4a` AAC default, `.caf` if Opus —
  per the Phase 2 codec finding).
- **RLS:** a device may `insert`/`select` only its own rows; no client reads
  others'; only `service_role` reads all. App ships **anon key only**.

### 3c. Communication protocol (user ↔ Supabase)

1. **First launch:** anonymous sign-in → JWT in Keychain (auto-refresh); upsert `devices`.
2. **Correction (offline-safe):** write SwiftData row `pending`; if audio consent,
   flush ring buffer → Opus file. Never blocks UI.
3. **Drain (foreground + connectivity, Wi-Fi by default):** per pending record,
   call Edge Function → (a) it returns a signed upload URL; (b) client PUTs audio
   to Storage; (c) Edge Function inserts the `corrections` row, **idempotent on
   client UUID** (`onConflict=id`); (d) mark local `uploaded`; purge local audio
   per retention.
4. **Resilience:** background `URLSession` (`isDiscretionary`,
   `waitsForConnectivity`) survives suspension/kill; exponential backoff;
   `attemptCount`/`lastError` tracked; poison records parked after N tries.
5. **Deletion:** "Delete my data" → Edge Function deletes device's rows + objects → local purge.

Audio-first-then-row ordering avoids dangling metadata; client-UUID idempotency
makes an offline batch of 10 drain exactly-once.

---

## 4. Phased plan (gated)

| Phase | Name | Role | Gate (success criterion) |
|---|---|---|---|
| **0** | Contracts & consent scaffolding | Mobile Architect | Toggles + identity + version stamping + wire-schema doc exist; **app behaves identically**; tests green. |
| 1 | Durable local store | Mobile Engineer | Corrections survive app kill; count reflects disk; unit tests. |
| 2 | Audio capture (local only) | Audio/Edge ML Engineer | ~30s Opus clip written, plays back, ≤150 KB; zero audio when opted out. |
| 3 | Supabase backend | Backend/Platform Engineer | Anon JWT can insert + upload under RLS; `service_role` reads all; cross-device read denied. |
| 4 | Sync engine | Backend + Mobile | Offline queues; reconnect drains; mid-upload kill resumes; retries produce no duplicates; cellular suppressed when Wi-Fi-only. |
| 5 | Consent, privacy, identity, lifecycle | Privacy + Mobile | Nothing leaves device without explicit opt-in; deletion verified end-to-end; privacy copy accurate. |
| 6 | Training ingestion (close the loop) | ML/Speech Data Engineer | A real correction round-trips into a training manifest; benchmark-shabad holdout enforced; review queue exists. |
| 7 | Observability & rollout | MLOps/Platform | Dashboards + alerts live; rate limits/quotas; staged feature-flagged rollout; integration tests on a Supabase branch. |

### Cross-cutting guarantees (every phase)
- Consent is structural: no record without `correctionsOptIn`; no clip without
  `audioCaptureOptIn`; no byte off-device without `uploadOptIn`.
- App ships **anon key only**; `service_role` never on device.
- Every correction stamped with `model_version` + `schema_version`.
- Idempotent uploads (client UUID PK); exactly-once server-side.
- Data minimization + right-to-delete; no PII; anonymous device id.
- Schema changes via Supabase migrations in git.
- All client work scoped to `ios/` + permitted docs (architectural invariants A1–A10).

---

## 5. Phase 0 — Contracts & consent scaffolding (THE FIRST STEP)

**Principle:** additive only, default OFF, **no behavior change**. Nothing in
Phase 0 is read by any emission/sync site yet — it lays the contracts the later
phases depend on. No SwiftData, no audio, no network, no SPM dependency, no UI.

### 0.1 Preferences — three new toggles
File: [Preferences.swift](../ios/Sources/SangatApp/Persistence/Preferences.swift).
Add to the `Key` enum + computed `Bool` accessors, matching the existing pattern:
- `audioCaptureOptIn` — default **false**.
- `uploadOptIn` — default **false**.
- `wifiOnlyUpload` — default **true**.
Test: extend `PreferencesTests` to assert defaults + round-trip via in-memory suite.

### 0.2 DeviceIdentity — Keychain anonymous id
New: `ios/Sources/SangatApp/Platform/DeviceIdentity.swift`.
- `deviceId: UUID` — read from Keychain (`kSecClassGenericPassword`,
  `kSecAttrAccessibleAfterFirstUnlock`); create + store on first access.
- Persistent across launches and (per decision 5) reinstalls.
- A `reset()` for the Phase 5 "delete my data" path (stub-callable now).
Test: same id across two reads; survives a fresh `DeviceIdentity` instance.

### 0.3 AppMetadata — version stamping
New: `ios/Sources/SangatApp/Platform/AppMetadata.swift`.
- `appVersion` ← `Bundle.main` `CFBundleShortVersionString` (Info.plist line 19).
- `buildNumber` ← `CFBundleVersion`.
- `modelVersion` ← single constant string for the bundled model, e.g.
  `"surt-small-v3-kirtan@6bit"`. Source of truth lives next to the model config
  (today `modelPath` is set at [AppEnvironment.swift:212](../ios/Sources/SangatApp/App/AppEnvironment.swift#L212));
  bump this string whenever the bundled `.mlmodelc` changes.
- `schemaVersion` ← `Int` constant, start at **1**.

### 0.4 Wire schema contract (Swift ↔ Postgres)
Document the field mapping (this doc, §3b is the column list). Decide JSON
encoding for `runner_ups` (`{"<shabadId>": <score>}`) and `recent_chunks`
(`[{start,end,text}]`). Add `schemaVersion` to the envelope so server can route
versions. No new Swift type required if `CorrectionEvent` + a thin
`CorrectionEnvelope { deviceId, appVersion, modelVersion, schemaVersion, event }`
suffices — confirm at build time.

### 0.5 Phase 0 gate (must all hold before Phase 1)
- New Preferences keys exist, default off/off/on, covered by tests.
- `DeviceIdentity.deviceId` stable across launches (Keychain) — tested.
- `AppMetadata` returns appVersion + modelVersion + schemaVersion.
- Wire-schema mapping documented here.
- `swift build` + `swift test --filter SangatAppTests` green.
- **No behavior change:** existing flows untouched; toggles read by nobody yet.
- Invariants A1–A10 hold (changes confined to `ios/` + this doc).

### Phase 0 touch budget (planned)
- Modify: `Preferences.swift`, `PreferencesTests.swift`.
- Add: `Platform/DeviceIdentity.swift` (+ test), `Platform/AppMetadata.swift` (+ test).
- Docs: this file.
- **Not touched:** `AppEnvironment` injection (Phase 1), `CaptionEngine` (Phase 2),
  `CorrectionsSettingsView` UI (Phase 5), `Package.swift` deps (Phase 4).

---

## 6. Open items / running status
- [x] **Phase 0 implemented (2026-05-21)** — gate met. Additive, default-OFF,
  behavior-neutral; `swift build` clean + `swift test --filter SangatAppTests`
  green (192 tests). Landed:
  - `Preferences`: `audioCaptureOptIn`=false, `uploadOptIn`=false,
    `wifiOnlyUpload`=true (+ tests).
  - `Platform/DeviceIdentity.swift` — Keychain anon UUID behind an injectable
    `IdentityStore` seam (+ tests via in-memory store).
  - `Platform/AppMetadata.swift` — `appVersion`/`buildNumber`/`modelVersion`
    =`surt-small-v3-kirtan@6bit`/`schemaVersion`=1 (+ tests).
  - `Corrections/CorrectionEnvelope.swift` — chosen over reusing `CorrectionEvent`;
    wraps event + lineage, `id` mirrors `event.id` (idempotency key), Codable (+ tests).
  - `Platform/Logger.swift` — added `AppLogger.sync` category.
  - Nothing reads these yet — no emission/sync site wired.
- [x] **Phase 1 implemented (2026-05-21)** — gate met. Durable SwiftData
  `CorrectionLog` (`CorrectionRecord` entity + `CorrectionSyncStatus` +
  `DurableCorrectionLog`), injection swapped in `production()` with in-memory
  fallback; per-record `syncStatus` + outbox helpers (`pending`/`markStatus`/
  `status`, unused until Phase 4). `swift build` clean + `swift test --filter
  SangatAppTests` green (200 tests). On-device data model documented in §7.
  Behavior-neutral by default (emission still gated on `correctionsOptIn`).
- [x] **Phase 2 (capture primitives) implemented (2026-05-21)** — gate met for
  what's host-verifiable. `AudioClipWriter` (window trim + AAC/Opus encode +
  total-bytes retention prune), `AudioClipCapturing` capability, engine
  `snapshotRecentAudio(seconds:)` reading WhisperKit's `audioProcessor.audioSamples`,
  and `LiveCaptionSource.captureCorrectionClip(...)`. `swift test` green (209,
  2 parity skips). Two findings documented: Opus→AAC default (decision #4 row);
  prune never deletes the newest clip.
  - **Retention cap decision (was open): total-bytes budget, 50 MB default**
    (`AudioClipWriter.defaultStorageBudgetBytes`), oldest pruned first, newest
    always kept. Resolved.
- [x] **Phase 2b (emission wiring) implemented (2026-05-21)** — both `hardNegPos`
  sites (Sevadar picker + Sangat wrong-shabad) now capture audio via
  `CorrectionAudioCapture` (gated: `correctionsOptIn` → `audioCaptureOptIn` →
  source is `AudioClipCapturing` → writer present) and thread `audioBufferPath`
  through `makeHardNegPos`. `AppEnvironment` owns the `AudioClipWriter`. Capture
  happens only inside `recordIfOptedIn` (no orphan clips). Gating logic
  host-tested with a stub capturer; `swift test` green (214, 2 parity skips).
  - **Device verification owed:** live mic snapshot → clip → playback on a real
    iPhone (no device yet — see beta blockers). The seam + gating are tested; the
    live-engine `captureCorrectionClip` path runs only on device.
  - Encode is synchronous on the tap (rare, user-initiated); move off-main if it
    ever hitches.
- [ ] `supabase-swift` SPM pin — added in Phase 4.
- [ ] Supabase project provisioning — Phase 3 (can use MCP).

---

## 7. Current on-device data model (Phase 1 — documented for future states)

This is the source of truth for what we persist on the device today, so future
SwiftData migrations, the Supabase schema (§3b), and the training-ingestion
parser (Phase 6) all agree on shape. **If you change this, bump the relevant
version and update §3b + this section.**

### Storage
- **Engine:** SwiftData (`@Model`), default on-disk store at SwiftData's
  standard Application Support location (one SQLite store + `-wal`/`-shm`).
- **Implementation:** `DurableCorrectionLog` ([ios/.../Corrections/DurableCorrectionLog.swift](../ios/Sources/SangatApp/Corrections/DurableCorrectionLog.swift)),
  injected in `production()` at
  [AppEnvironment.swift](../ios/Sources/SangatApp/App/AppEnvironment.swift) via
  `makeCorrectionLog()`; falls back to `InMemoryCorrectionLog` if the store
  can't open. Previews/tests still use in-memory.
- **Concurrency:** all `ModelContext` access confined to one private serial
  `DispatchQueue`; `record` is async (non-blocking), reads are `sync`.

### `CorrectionRecord` (SwiftData entity)
[ios/.../Corrections/CorrectionRecord.swift](../ios/Sources/SangatApp/Corrections/CorrectionRecord.swift)

| Column | Type | Notes |
|---|---|---|
| `id` | `UUID` | `@Attribute(.unique)` → idempotent upsert; mirrors `CorrectionEvent.id`; upload idempotency key |
| `createdAt` | `Date` | mirrors `CorrectionEvent.timestamp`; sort key for "newest N" |
| `syncStatusRaw` | `String` | `CorrectionSyncStatus` raw value; outbox query key |
| `attemptCount` | `Int` | upload attempts; incremented on each `.uploading` transition |
| `lastError` | `String?` | last upload error, diagnostic only |
| `audioPath` | `String?` | mirror of `CorrectionEvent.audioBufferPath` for sync/cleanup (set in Phase 2) |
| `payload` | `Data` | **JSON-encoded `CorrectionEvent`** — the full rich record |

**Design choice:** the rich correction fields (predicted/groundTruth/runnerUps/
recentChunks/…) live only inside the `payload` blob. The store never queries
them on-device, so this keeps the SwiftData schema small + stable and means a new
`CorrectionEvent` field is a payload change, **not** a SwiftData migration. The
rich shape's source of truth remains `CorrectionEvent` + the wire
`CorrectionEnvelope`.

### `CorrectionSyncStatus` lifecycle
[ios/.../Corrections/CorrectionSyncStatus.swift](../ios/Sources/SangatApp/Corrections/CorrectionSyncStatus.swift)

```
pending ──(sync starts)──> uploading ──(server confirms)──> uploaded   [terminal]
   ▲                           │
   └──────(retryable)──────────┘
                               └──(non-retryable)──────────> failed     [terminal]
```
Only `pending` records are eligible for upload (`DurableCorrectionLog.pending(limit:)`).
Status transitions via `markStatus(_:forId:error:)`. These outbox helpers are
concrete (not on the `CorrectionLog` protocol) and are **unused by app code
until Phase 4**.

### Captured audio clips (Phase 2)
[ios/.../Audio/AudioClipWriter.swift](../ios/Sources/SangatApp/Audio/AudioClipWriter.swift)

- **Location:** `Application Support/CorrectionAudio/`.
- **File:** `<correctionId>.<ext>` — `.m4a` (AAC, default) or `.caf` (Opus).
  The filename's UUID equals `CorrectionEvent.id` / `CorrectionRecord.id`, so a
  clip joins to its record (and `CorrectionRecord.audioPath` mirrors the full path).
- **Format:** 16 kHz mono. AAC ~32 kbps (~120 KB / 30 s); the snapshot window is
  the last N seconds of `CaptionEngine.snapshotRecentAudio(seconds:)`.
- **Source:** read on demand from WhisperKit's `audioProcessor.audioSamples`;
  available history is bounded by WhisperKit's purge policy (may be < N seconds).
- **Retention:** total-bytes budget (default 50 MB); oldest pruned after each
  write, newest always kept. `deleteClip(atPath:)` after confirmed upload;
  `deleteAll()` for Phase 5 "Delete my data".
- **Consent:** written only when `Preferences.audioCaptureOptIn` is on (caller-
  enforced); device-local until Phase 4 upload + the separate `uploadOptIn`.
- **Trigger (Phase 2b):** both `hardNegPos` emission sites in `RootView` call
  `CorrectionAudioCapture.capture` → `LiveCaptionSource.captureCorrectionClip`,
  threading the path into `CorrectionEvent.audioBufferPath` (and thus
  `CorrectionRecord.audioPath`). Consent chain: `correctionsOptIn` →
  `audioCaptureOptIn` → source supports audio → writer exists. Gating is tested;
  live capture is device-verified.

### Versioning & migration
- Wire/contract version = `AppMetadata.schemaVersion` (currently **1**), carried
  on every upload, stored server-side per row.
- SwiftData schema changes: additive column changes are lightweight-migratable;
  anything else needs a `VersionedSchema` + `SchemaMigrationPlan` (not yet
  required — single schema today).

### Maps to
- Wire: `CorrectionEnvelope` (§3b adds device + version lineage around the event).
- Server: the Postgres `corrections` table (§3b) — same fields, `payload`'s
  inner fields expanded into columns server-side for queryability.

---

## 8. Phase 3 — Supabase backend (NEXT, plan)

**Role:** Backend / Platform Engineer (lead) + Data Engineer.

**Goal:** a reproducible Supabase backend (schema + storage + security +
gateway) so a device can submit a correction (metadata row + audio object), and
the training side (`service_role`) can read everything — **developed and tested
locally first**; applying it to the existing shared project is a separate,
approval-gated step.

**Principle:** first phase that adds files **outside `ios/`** (a new `supabase/`
dir). No `ios/` changes in Phase 3 — the `supabase-swift` SDK + client wiring is
Phase 4. Backend is the unit of work here.

### Shared-project coexistence (existing Prisma project) — decided 2026-05-21
We **reuse the existing Supabase project** (the one the other app uses via its
Prisma `DATABASE_URL`), not a new project. Isolation rules:
- **Dedicated schema** (proposed `gurbani_captioning`) for all our tables —
  never `public` (Prisma owns `public`). Keep our schema out of Prisma's managed
  schema list so neither tool's migrations touch the other's objects.
- **Additive migrations only on the shared remote. NEVER `supabase db reset`
  against it** — that wipes the other app's data. `db reset` is for the *local*
  stack only; apply to remote as a single additive migration (Supabase MCP
  `apply_migration`).
- **Auth is project-global** — enabling anonymous sign-ins would affect the
  other app. **Recommended: gateway-only writes** (Edge Function runs as
  `service_role`); RLS denies direct client access. No project-wide auth change.
- **Storage bucket + Edge Function are additive/isolated** — a new
  `correction-audio` bucket and `submit-correction` function don't affect the
  other app.

### CONFIRMED target + strict-additive minimal plan (2026-05-21)
Target project = **Autom8x** (`nwmfpmfsqfexxvkhoise`, org `phpmuidvaycwbelxsgkv`,
us-west-2, PG 17). Its `public` schema is a **live Prisma app with real data**
(users, workspaces, projects, GL-code tables w/ 10k+ rows, `_prisma_migrations`).

**User constraint (hard):** *do not touch or edit anything existing; only ADD a
new data table; existing data + auth must not be tweaked.* So Phase 3 is reduced
to a strict-additive minimum:
- **One new schema `gurbani_captioning` + one table `corrections`** (all fields
  denormalized as columns — no separate `devices`/`model_versions` tables). New
  schema = physically isolated from `public`/Prisma; nothing existing is read or
  altered. Reversible by `DROP SCHEMA gurbani_captioning` with zero blast radius.
- **No auth change.** No anonymous sign-in toggle, no provider/role/policy edits
  on existing objects. **No user sign-in in the app** (open → use; picking a
  shabad is an automatic, disclosed correction — consent via the in-app toggle,
  not an account).
- **No Edge Function, no storage bucket** in this step (those exceed "a table").
  Audio is **metadata-only for now**; a bucket is a separate, explicitly-approved
  follow-up.
- **Write path:** app inserts via Autom8x's existing **anon/publishable key** +
  an RLS **INSERT** policy on *our* table only (with column CHECKs); **SELECT/ALL
  restricted to `service_role`** (training reads). Policies live on our table
  only — existing tables' RLS untouched. (Trade-off: the Gurbani app embeds
  Autom8x's public anon key, and anon insert is a mild abuse surface; an Edge
  Function gateway can harden this later if wanted.)
- **Apply path:** build + test the migration on a **throwaway local stack**
  first; show the exact SQL; apply to Autom8x only on approval via MCP
  `apply_migration` (single additive migration; **never** `db reset` remote).

This supersedes the fuller §3b/§8 design (devices/model_versions tables, Edge
Function, signed-URL bucket) where they conflict — those become optional later
additions, each needing explicit approval.

### Prereqs (DONE 2026-05-21)
- ✅ **Supabase CLI 2.101.0** installed (`brew install supabase/tap/supabase`).
- ✅ **Docker 29.2.1** running.
- Deno not needed (the CLI bundles a runtime for functions).

### Local build + validation (DONE 2026-05-21)
- `supabase init` → `supabase/` (config.toml, migrations). `project_id =
  "live-gurbani-captioning"` (local label only).
- Migration `supabase/migrations/20260525224845_gurbani_captioning_corrections.sql`
  — schema `gurbani_captioning` + table `corrections` + grants + RLS (anon
  insert-only, service_role full). Metadata-only (audio_* columns reserved,
  nullable). **Touches nothing in `public`.**
- Local `[api].schemas` set to include `gurbani_captioning` (mirrors the one
  additive remote API setting needed).
- **Validated on the local stack (all pass):** anon INSERT `return=minimal` →
  201; anon SELECT → 401 (blocked); service_role SELECT → row; bad `kind` →
  rejected by RLS `with_check`; duplicate `id` → 409; RLS enabled+forced.

### Client contract (for Phase 4 — learned from local tests)
- Insert with **`Prefer: return=minimal`** — anon has **no SELECT** grant, so a
  representation/RETURNING insert 401s. (supabase-swift: `returning: .minimal`.)
- **Treat HTTP 409 as success** (row already uploaded) — idempotency via PK.
- Target the schema via `Content-Profile: gurbani_captioning` (supabase-swift
  `.schema("gurbani_captioning")`).
- Requires `gurbani_captioning` in the project's **exposed API schemas**.

### PIVOT → dedicated project, and Phase 3 DONE (2026-05-21)
Decision changed: instead of the shared Autom8x project we created a **separate
free Supabase project** — cleaner isolation, we own all settings, zero Autom8x
risk, and the app carries its own keys. **Autom8x was never written to**
(read-only inspection only). The Autom8x "strict-additive" subsections above are
now historical.

**Project:** `gurbani-captioning` — ref `mlovvqoiuuihmymkypfm`, org `siddak1234`
(free, $0/mo), region `us-west-1`. URL `https://mlovvqoiuuihmymkypfm.supabase.co`.

Because it's our own project, the table lives in **`public`** (no schema-exposure
toggle, no `Content-Profile` header needed). Applied via MCP `apply_migration`,
mirrored by the files in `supabase/migrations/`:
1. `20260525224845_corrections_table.sql` — `public.corrections` + indexes + RLS
   (anon insert-only, service_role full), metadata-only (audio_* reserved).
2. `20260525230725_tighten_anon_grants.sql` — revoke Supabase's default broad
   anon grants; anon = INSERT only.

**Live validation (all pass):** anon INSERT `return=minimal` → 201; anon SELECT →
401 (blocked); duplicate `id` → 409 (idempotent); bad `kind` → RLS-rejected;
`anon_grants` = INSERT only; **security advisor: 0 lints**. Test rows deleted.

**Free-tier notes:** project pauses after ~7 days inactivity (restore on demand);
limits 500 MB DB / 1 GB storage; org now at 2/2 free projects.

**Credentials for Phase 4** (publishable key is safe to embed; **service_role key
never ships in the app**): the URL above + the `sb_publishable_...` key from
`get_publishable_keys` / dashboard.

Phase 3 = **DONE.**

---

## 9. Phase 4 — iOS sync engine (DONE 2026-05-21)

**Role:** Backend + Mobile Engineer. **Goal:** drain pending durable corrections
to the `gurbani-captioning` project when opted-in + online, advancing each
record's sync status, idempotently.

### Engineering decision (deviation, flagged)
Used a **thin URLSession PostgREST client, not the `supabase-swift` SDK**
(decision #2 had picked the SDK). Rationale: the contract is a single
authenticated POST; the SDK adds a large dependency tree (+ xcodegen/Package
churn) for no benefit and doesn't help here, while a thin client gives exact
control over the `return=minimal`/409 contract. **Zero new dependencies.**
Reversible if the SDK is preferred later.

### Resilience model
Foreground/online-triggered drain (not a background `URLSession`): an
interrupted upload leaves the record `pending` (or stale-`uploading`, requeued
at next sync via `requeueInFlight()`), so the next trigger retries.
Exactly-once is guaranteed server-side by the primary key (409 → treated as
success). Sufficient for tiny, infrequent metadata; true background-session
upload can be added later.

### Files
- Added `ios/Sources/SangatApp/Sync/`: `SupabaseConfig` (URL + publishable key —
  safe to embed; service_role never ships), `CorrectionRowDTO` (envelope →
  snake_case row), `CorrectionsUploader` (`UploadOutcome` + status classifier +
  `SupabaseRESTUploader`), `Reachability` (`NWPathMonitor`), `SyncCoordinator`
  (outbox drain).
- Modified: `Corrections/DurableCorrectionLog.swift` (+`requeueInFlight()`),
  `App/AppEnvironment.swift` (build + expose `syncCoordinator`, nil when no
  durable store / in previews), `App/RootView.swift` (one-line `.task` trigger:
  `await env.syncCoordinator?.sync()`).
- Tests `ios/Tests/SangatAppTests/Sync/`: `UploadOutcomeTests`,
  `CorrectionRowDTOTests`, `SyncCoordinatorTests`.

### Audit
- **Touch budget:** 3 modified + 5 added source + 3 added test files; all under
  `ios/`. **Scope clean** — nothing outside `ios/` + `supabase/` + this doc.
- **Invariants:** A1 (ios-only) ✅; consent structural — `sync()` early-returns
  unless `uploadOptIn` ✅; service_role key never in app (publishable only) ✅;
  idempotent (PK + 409-as-success) ✅; no new SPM dependency ✅.
- **Tests:** `swift build` clean; full suite **230 tests, 0 failures** (2 parity
  skips); 16 new Phase-4 tests cover the classifier, DTO column mapping, and the
  drain across all gates/outcomes (opt-in, offline, Wi-Fi-only, success,
  retryable, permanent, already-uploaded, stale-requeue).
- **Live verification:** the exact full-column DTO JSON inserted to the real
  project → 201; stored row verified (jsonb `runner_ups`/`recent_chunks`,
  Gurmukhi preserved, client `created_at` vs server `inserted_at`,
  `export_status='new'`); test rows deleted (table empty). Security advisor: 0
  lints (Phase 3).

### Deferred
- Audio upload (needs a Storage bucket) — local clips are kept, not deleted on
  metadata upload. Separate approved step.
- Background `URLSession` for upload-during-suspension (current model retries on
  next foreground).
- Consent UI (Settings upload/audio toggles, "Delete my data") → Phase 5.

Phase 4 = **DONE.**

---

## 10. Phase 5 — consent / privacy UI (DONE 2026-05-21)

**Role:** Privacy + Mobile. **Goal:** make consent user-controllable and the
copy honest before any beta use (until now nothing could be opted into and the
copy falsely said "we do not upload").

### Changes (`CorrectionsSettingsView`)
- New toggles, nested under the master "Help improve detection" opt-in (shown
  only when it's on): **"Upload my corrections"** (`uploadOptIn`) → **"Wi-Fi
  only"** (`wifiOnlyUpload`, shown when upload on); **"Include a short audio
  clip"** (`audioCaptureOptIn`). Each seeds from / persists to `Preferences`.
- Fixed the inaccurate *"We do not upload anything in this build"* copy →
  "Private by default … only sent if you turn on uploading … no account, no
  personal info"; explainer now mentions the optional audio clip.
- "Clear corrections" → **"Delete on-device data"**: also purges audio clips
  (`audioClipWriter.deleteAll()`). Storage subtitle corrected.

### Audit
- **Touch budget:** 1 modified source (`CorrectionsSettingsView.swift`) + 1
  modified test (`CorrectionsViewSmokeTests.swift`). ios-only.
- **Invariants:** consent structural — toggles drive the gates already enforced
  by `recordIfOptedIn` / `captureCorrectionClip` / `SyncCoordinator` ✅; privacy
  copy now accurate ✅.
- **Tests:** build clean; full suite **231 pass** (2 parity skips); smokes
  construct the view incl. upload+audio on.

### Owed
- **Visual verification on the Simulator** — the upload/audio section renders
  behind `if optedIn` / `if uploadOptIn` (seeded in `onAppear`), which unit
  smokes don't drive. Matches the standing simulator-verification debt.
- ✅ **Server-side delete — DONE (2026-05-21).** `delete-my-data` Edge Function
  deployed (`verify_jwt=false`; self-authorizes via the secret device UUID; runs
  as `service_role` to delete rows for that `device_id`). Client: the "Delete my
  data" button purges local data + calls `SyncCoordinator.deleteMyData()`.
  Live-verified (insert 2 → delete → `*/2`; bad id → 400; table empty).
  `supabase/functions/delete-my-data/index.ts`.

Phase 5 = **DONE** (UI + server-delete); Simulator visual check still owed.

---

## 11. Owed items → phase assignments (tracker)

Every deferred item now lives in a phase, so nothing floats:

| Item | Phase | Status |
|---|---|---|
| Server-side delete (`delete-my-data` Edge Function + client call) | **5** (privacy) | ✅ DONE 2026-05-21 |
| Audio upload — Storage bucket + client upload of the clip | **6a** | ✅ DONE 2026-05-21 |
| Training ingestion — `scripts/pull_corrections.py` → manifest + QA gate + holdout | **6b** | owed (best after 6a so audio is present) |
| Background `URLSession` upload (survive suspension mid-upload) | **7** (hardening) | optional |
| Observability — client metrics, rate limits, dashboards | **7** | owed |
| Simulator/device E2E verification of capture→upload | **8** (beta readiness) | owed (needs device/sim) |

### Phase 6a — audio upload (sketch)
Add a private `correction-audio` Storage bucket + storage RLS (a device writes
only its own `<device_id>/` prefix). In `SyncCoordinator`, after the metadata
row uploads, upload the local clip to `correction-audio/<device_id>/<id>.<ext>`,
set the row's `audio_path` to that key, then delete the local clip. Idempotent
(overwrite by key). Honors `audioCaptureOptIn` + `uploadOptIn` + Wi-Fi.

### Phase 6b — training ingestion (sketch)
`scripts/pull_corrections.py`: read `public.corrections` via **service_role**
(key from env, never committed), download referenced audio, enforce the
benchmark-shabad holdout (`configs/datasets.yaml`), emit a reviewable manifest
under `training_data/<batch>/`, and flip `export_status` to `exported` (or
`discarded`) behind a human QA gate.

---

## 12. Phase 6a — audio upload (DONE 2026-05-21)

**Role:** Backend + Mobile. Audio clips now upload to a private Storage bucket
alongside the metadata, completing the trainable-signal pipeline. Consent-gated:
clip exists only with `audioCaptureOptIn`; upload only with `uploadOptIn` +
online (+ Wi-Fi unless disabled).

### Backend (`supabase/migrations/`, applied via MCP)
- `20260525234000_correction_audio_bucket.sql` — private `correction-audio`
  bucket (5 MB limit, audio mime types) + storage RLS: **anon INSERT-only**,
  service_role full. Object key `<device_id>/<correction_id>.<ext>`.
- `delete-my-data` Edge Function **v2** now also purges the device's Storage
  objects (lists `<device_id>/`, deletes), so right-to-delete covers audio.

### Client (`ios/Sources/SangatApp/Sync/`)
- `SupabaseStorageUploader` (`CorrectionAudioUploading`) — POSTs the clip; **no
  `x-upsert`** (anon is insert-only); detects Storage's "409 Duplicate" body →
  `alreadyUploaded` for idempotent retries.
- `CorrectionRowDTO.audio_path` now holds the **Storage key** (never the local path).
- `CorrectionsUploading.upload` gains `storageAudioPath`.
- `SyncCoordinator`: uploads audio **first** (so the row carries the key — anon
  can't UPDATE post-insert), then metadata; deletes the local clip on success;
  audio-retryable re-queues the whole record.
- `AppEnvironment` wires the storage uploader + clip writer into the coordinator.

### Audit
- **Touch budget:** +1 migration, +1 function redeploy, +1 new client source, 4
  modified client source, 2 modified tests; scope = `ios/` + `supabase/`.
- **Invariants:** consent structural (audio only with both opt-ins + online) ✅;
  anon insert-only on the bucket (no read/list/update) ✅; idempotent
  (409→already) ✅; `audio_path` = storage key, not local path ✅;
  **security advisor: 0 lints** (re-run after storage policies).
- **Tests:** build clean; full suite **233 pass** (2 parity skips); +2 coordinator
  audio tests (audio→key→metadata→clip deleted; audio-retryable keeps pending+clip).
- **Live verification:** anon audio upload → 200; metadata row with key → 201;
  duplicate → 409-in-body; `delete-my-data` purged rows + objects; bucket/table
  empty after.

Phase 6a = **DONE.**

### Approach / deliverables (all under a new `supabase/`)
1. **Local stack:** `supabase init` → `config.toml`; `supabase start` (Docker:
   Postgres + Auth + Storage + Edge runtime). Develop/test entirely locally.
2. **Migrations** `supabase/migrations/NNNN_*.sql`:
   - `devices`, `corrections`, `model_versions` (per §3b); FKs; `CHECK`
     constraints for `kind` / `export_status` / `review_status`.
   - Indexes: `corrections(device_id)`, `corrections(export_status, created_at)`.
   - All tables in the **dedicated `gurbani_captioning` schema**, never `public`.
   - **RLS enabled + deny direct client access**; only `service_role` (the Edge
     Function) reads/writes — gateway-only (see coexistence).
3. **Storage:** private bucket `correction-audio` + storage RLS so a device can
   write/read only its own `<device_id>/` prefix.
4. **Auth:** gateway-only — **no project-wide anonymous auth** (it's global and
   would affect the other app). The Edge Function authenticates the request and
   writes as `service_role`. (See coexistence + decision 4.)
5. **Edge Function** `supabase/functions/submit-correction/` (Deno/TS), runs as
   `service_role`: authenticate (device_id + app key) → validate envelope
   (`schema_version`, required fields) → **upsert `corrections` idempotent on
   `id`** → return a **signed upload URL** for the audio object → per-device
   rate-limit. (A `delete-my-data` function stub may land here for Phase 5.)
6. **Tests** `supabase/tests/` (pgTAP or a script): device A submits row +
   uploads object; device B is **denied** A's row (RLS); `service_role` reads
   all; Edge Function dedupes (same `id` twice → one row).
7. `.gitignore`: local Supabase artifacts + `.env` (never commit `service_role`).

### Identity / RLS (shared-project-safe)
Gateway-only: the `submit-correction` Edge Function (running as `service_role`)
is the sole writer; RLS on our tables **denies direct anon/authenticated access**
and allows only `service_role`. The Keychain `device_id` (Phase 0) is the device
identity, passed to the function and stored as the `device_id` column. No
project-wide anonymous auth required. Abuse control = per-device rate-limit in
the function + an app key. (Alternative, if the other app's owner agrees:
project-wide anon auth with `auth.uid()`-owned rows.)

### Success criteria / gate
- `supabase db reset` applies all migrations cleanly on the local stack.
- Local integration test: anon device submits (row + signed-URL upload) under
  RLS; cross-device read denied; `service_role` reads all; Edge Function dedupes
  on `id`.
- Fully reproducible from `supabase/` in git; **no secrets committed**.
- **Remote provisioning is NOT required to pass the gate** — it's a separate
  approval-gated step before Phase 4.

### Touch budget
- Add: `supabase/` (config, migrations, functions, tests), `.gitignore` entries,
  optional `Makefile` `supabase-*` targets.
- **Not touched:** `ios/` (Phase 4), the training-side `scripts/pull_corrections.py`
  (Phase 6).

### Deferred out of Phase 3
- iOS `supabase-swift` SDK + `SyncCoordinator` + app→server calls → **Phase 4**.
- Applying the additive migration to the **shared remote** project (via Supabase
  MCP `apply_migration`, **with approval**) → gated step before Phase 4.

### Decisions / status
1. **Local-first: confirmed.** Develop/test on the local stack; apply an additive
   migration to the shared remote later (approval-gated, never `db reset`).
2. **Prereqs (checked): CLI missing + Docker daemon down.** Install
   `supabase/tap/supabase` + start Docker before the local stack. (Awaiting OK.)
3. **Target project + schema:** which existing Supabase project? (Point me to the
   project ref, or I can `list_projects` via MCP once you OK touching that
   account.) Schema name proposed: `gurbani_captioning`.
4. **Auth approach:** gateway-only writes (recommended, no global auth change)
   vs enable project-wide anonymous auth.
5. **Upload pattern:** signed upload URL (recommended) vs base64-in-function.
