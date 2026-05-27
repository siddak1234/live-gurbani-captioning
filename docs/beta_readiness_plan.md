# Beta Readiness Plan

**Created:** 2026-05-27. **Owner:** siddak. **Companion to:** [`docs/corrections_feedback_loop_plan.md`](corrections_feedback_loop_plan.md).

Five groups of work to get from "the app runs locally and the feedback loop
works in tests" to "a signed TestFlight build in beta testers' hands collecting
real-world corrections." Each task has an explicit owner (**YOU** = siddak,
**CLAUDE** = code/config) and an audit step that proves it's done.

## Status legend
- ⏳ pending
- 🔄 in progress
- ✅ done

---

## Group 1 — Apple Developer account & signing (gates everything)

Without this group, no signed `.ipa`, no TestFlight, no Keychain on device.

| # | Task | Owner | Audit | Status |
|---|---|---|---|---|
| 1.1 | Enroll in Apple Developer Program ($99/yr) at developer.apple.com/programs | YOU | Enrollment email received; team ID visible in Membership page | ⏳ |
| 1.2 | Register bundle ID `com.sangat.app` (or fallback `com.siddak.sangat` if taken) in Certificates, Identifiers & Profiles → Identifiers | YOU | Bundle ID listed in App Store Connect | ⏳ |
| 1.3 | Create iOS Distribution certificate via Xcode → Settings → Accounts → Manage Certificates (Xcode handles cert + automatic provisioning profile) | YOU (in Xcode) | `security find-identity -v -p codesigning` lists "Apple Distribution: <Your Name>" | ⏳ |
| 1.4 | Send Claude: (a) **team ID** (10-char string from Membership), (b) **final bundle ID** chosen, (c) **your developer-program-registered email** | YOU | Claude has all three | ⏳ |
| 1.5 | Update `ios/project.yml`: set `DEVELOPMENT_TEAM = <team-id>`, flip `CODE_SIGNING_ALLOWED = YES`, set `CODE_SIGN_STYLE = Automatic`, regenerate xcodeproj via `xcodegen` | CLAUDE | `xcodebuild -showBuildSettings -scheme GurbaniCaptioningApp \| grep DEVELOPMENT_TEAM` returns the team ID | ⏳ |
| 1.6 | First signed archive: `xcodebuild archive -scheme GurbaniCaptioningApp -archivePath build/Sangat.xcarchive -destination 'generic/platform=iOS'` | CLAUDE | `.xcarchive` exists; `codesign -dvv` on the embedded `.app` shows the user's cert + team | ⏳ |

---

## Group 2 — App Store Connect setup

| # | Task | Owner | Audit | Status |
|---|---|---|---|---|
| 2.1 | App Store Connect → My Apps → "+" → New App. Platform iOS, name "Sangat", primary lang English, bundle ID from 1.2, SKU `sangat-001`, user access Full | YOU | App appears in Apps list | ⏳ |
| 2.2 | **Privacy Nutrition Labels** — Claude drafts the exact answers based on the 3 consent gates + audio upload + Keychain UUID; user clicks through the questionnaire in App Store Connect | CLAUDE drafts, YOU submits | Privacy section shows "Complete" green check | ⏳ |
| 2.3 | Upload first build via Xcode Organizer → Distribute App → TestFlight | YOU | Build appears in App Store Connect → TestFlight tab, status "Processing" (clears in ~5-10 min) | ⏳ |
| 2.4 | Add internal testers (just YOU and any close collaborators) by email; accept invite on device | YOU | TestFlight app on iPhone shows Sangat available to install | ⏳ |

---

## Group 3 — Real-device verification (Phase 8 owed)

**Cannot be done on the simulator.** The simulator misses Keychain persistence,
ANE acceleration, real mic capture, and end-to-end Supabase round trip from a
real device id.

| # | Task | Owner | Audit | Status |
|---|---|---|---|---|
| 3.1 | Install signed build on iPhone via TestFlight (from 2.4) | YOU | App launches; onboarding flow renders | ⏳ |
| 3.2 | Verify `DeviceIdentity` persists across launches (Keychain works on signed builds, fails on unsigned sim) | YOU + CLAUDE | Two launches on the same device produce correction rows in Supabase with the **same `device_id` UUID** | ⏳ |
| 3.3 | Measure ANE perf: time from tap-Listen to first caption with real kirtan playing | YOU + CLAUDE | Wall-clock measurement recorded in `docs/phase8_device_perf.md`. Target: < 5s. Stretch: real-time factor < 1.0 | ⏳ |
| 3.4 | End-to-end correction flow: identify shabad (or pick manually) → tap "wrong shabad" → select correct → row arrives in Supabase with audio clip | YOU + CLAUDE | Row visible in `public.corrections` with `kind = 'hardNegPos'`; matching `.<ext>` audio file in `correction-audio` bucket at `<device_id>/<correction_id>.<ext>` | ⏳ |
| 3.5 | Test the 45s manual-pick timeout (`42619ce`) — let it expire on silent audio, pick manually, confirm caption flows | YOU | Picker sheet auto-appears at ~45s; manual pick locks engine; caption lines render in the chosen layout | ⏳ |
| 3.6 | Tap-through 5+ minute kirtan session to surface any UI hangs, memory growth, or thermal issues | YOU | App stays responsive; memory in Xcode Debug Navigator stays under ~600 MB; no thermal warnings | ⏳ |

---

## Group 4 — Feedback infrastructure decision

| # | Task | Owner | Audit | Status |
|---|---|---|---|---|
| 4.1 | **Decide:** TestFlight built-in feedback only **OR** add custom Settings → "Send feedback" → `public.feedback` Supabase table | YOU | Decision noted here | ⏳ |
| 4.2 | If custom: Claude adds Supabase migration for `public.feedback` (RLS anon-INSERT-only, mirrors `corrections`), Settings UI button, uploader reusing existing `SupabaseRESTUploader` infra (~1h work) | CLAUDE | Tester taps Send feedback → row appears in Supabase `feedback` table | ⏳ |

**Claude's recommendation:** TestFlight built-in for v1. Crash reports auto-collect
in App Store Connect → TestFlight → Crashes. Testers can screenshot-annotate via
the TestFlight app. Saves a Supabase table that may turn out to be unused. Tradeoff:
feedback lives in App Store Connect (not Supabase), so it's not in the same place
as the corrections data — but for v1 of beta that's fine.

---

## Group 5 — Plan-doc paperwork

| # | Task | Owner | Audit | Status |
|---|---|---|---|---|
| 5.1 | Update `docs/corrections_feedback_loop_plan.md` §15 with the realtime-loop fix + 45s timeout + "tests passing ≠ feature works" lesson | CLAUDE | Diff committed | ✅ |
| 5.2 | Update memory `project_corrections_feedback_loop.md` with the start() blocking gotcha | CLAUDE | Memory file updated | ✅ |
| 5.3 | Write this doc | CLAUDE | This file exists in the repo | ✅ |

---

## Execution order

1. **Now** (CLAUDE, no creds): Group 5 ✅
2. **Now** (YOU): Group 1 items 1.1-1.4 in parallel (enrollment takes 12-48h)
3. **As soon as user has team ID + bundle ID:** Claude does 1.5, 1.6, 2.2 draft
4. **YOU:** 2.1, 2.3, 2.4 (each ~5 min after the previous), then 3.1
5. **Together:** Group 3 verification (~30 min, app on real iPhone, kirtan playing nearby)
6. **YOU decide:** Group 4
7. **Open beta:** add external testers (up to 10k via public TestFlight link)

**Estimated wall-clock to first TestFlight build:** 1-2 days, almost entirely
gated by Apple Developer enrollment processing time.

---

## Out of scope for v1 beta

To avoid scope creep:

- App Store full release (TestFlight is the goal; App Store is post-beta).
- iPad-specific layout tuning beyond what Auto Layout gives for free.
- Background `URLSession` for correction upload (foreground + scenePhase trigger is sufficient for tiny payloads; can revisit if testers report stranded uploads).
- Observability UI in Settings (data is wired via `SyncCoordinator.stats`; visibility is owed but not blocking).
- Server-side training pipeline (`pull_corrections.py` end-to-end with real service-role key — meaningless until rows exist).
- ANE compute-unit override (`MLComputeUnits.cpuAndNeuralEngine` is WhisperKit's default for `.mlmodelc` on A12+, so no explicit config needed unless we discover otherwise during 3.3).
