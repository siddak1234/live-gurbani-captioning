# Privacy Nutrition Labels — App Store Connect draft (task 2.2)

Pre-filled answers for the App Store Connect → App Privacy questionnaire. Derived
from the three consent gates (`correctionsOptIn`, `audioCaptureOptIn`,
`uploadOptIn`, all default **OFF**), the Supabase upload path, and the
Keychain-persistent anonymous device UUID. Submit these in App Store Connect →
your app → App Privacy → "Get Started" / "Edit".

**Ground truth this is built from:**
- Nothing leaves the device unless the user turns on `uploadOptIn` (default off).
- Audio is only captured if `audioCaptureOptIn` is also on (default off).
- No account, no sign-in, no name/email. Identity is an anonymous UUID stored in
  the Keychain, used only to dedup uploads and honor "Delete my data".
- Backend is our own Supabase project (`mlovvqoiuuihmymkypfm`), not an ad/analytics
  SDK. No third-party tracking.

---

## Q1: Do you or your third-party partners collect data from this app?

**Yes** — but only when the user opts in. (App Store requires "Yes" if *any* code
path collects data, even opt-in. The labels below all get marked as optional /
not-collected-by-default where the questionnaire allows, but the gate itself is
the opt-in toggle.)

## Data types collected

### 1. Audio Data  → category: **User Content → Audio Data**
- **Collected:** Yes (only when `audioCaptureOptIn` + `uploadOptIn` are on).
- **Linked to the user's identity?** Yes — linked to the anonymous device UUID.
  (Apple counts a persistent device identifier as "linked" even without a name.)
- **Used for tracking?** No.
- **Purposes:**
  - **App Functionality** (the correction the user submitted includes the short
    clip that was misidentified).
  - **Product Personalization / App Functionality → model improvement** — the clip
    is used to fine-tune the captioning model. Under Apple's purpose list this is
    **"App Functionality"** and/or **"Analytics"** is *not* right; use
    **"Product Personalization"** is also not right. Closest correct purpose:
    **"App Functionality"** + **"Other Purposes" → improving the ML model**. In the
    free-text "purpose" field write: *"Audio clips of misidentified kirtan are used
    to improve the on-device captioning model."*

### 2. Other User Content (the correction itself) → **User Content → Other User Content**
- **Collected:** Yes (only when `correctionsOptIn` + `uploadOptIn` are on).
- **What it is:** which shabad/line the engine got wrong and what the user picked
  instead (text identifiers + timestamps, no free-text from the user).
- **Linked to identity?** Yes — to the anonymous device UUID.
- **Used for tracking?** No.
- **Purpose:** App Functionality + ML model improvement (same free-text note).

### 3. Device ID  → category: **Identifiers → Device ID**
- **Collected:** Yes (only when uploads happen).
- **What it is:** an anonymous UUID we generate and store in the Keychain. Not the
  IDFA, not the vendor ID. Used to dedup uploads (idempotent PK) and to power the
  "Delete my data" button.
- **Linked to identity?** Yes (it *is* the pseudo-identity).
- **Used for tracking?** No.
- **Purpose:** App Functionality.

## Data types NOT collected (answer "No" / leave unchecked)

- Contact Info (name, email, phone, address) — none. No account.
- Health & Fitness — none.
- Financial Info — none.
- Location (precise or coarse) — none.
- Browsing/Search History — none.
- Contacts — none.
- Diagnostics / Crash data — **handled by TestFlight/App Store Connect, not by our
  code.** If you ship custom crash reporting later, revisit. For v1: not collected
  by the app itself.
- Purchases, Usage Data (product interaction analytics) — none. We do not log
  screen views or analytics events.

## Q: Used for tracking (App Tracking Transparency)?

**No.** Data is never shared with third parties for advertising or cross-app
tracking. Our Supabase backend is first-party storage. → You do **not** need an
ATT prompt (`NSUserTrackingUsageDescription` not required).

---

## Sensitive-data note (read before submitting)

The audio is recordings of **religious worship (kirtan)**, and the corrections
encode which **scripture line** was sung. Apple's "Sensitive Info" data type
explicitly includes *religious beliefs*. This app's whole purpose is captioning
Sikh scripture, so the religious nature is inherent and disclosed in the app
description — but be deliberate here:

- Apple's questionnaire does not force "Sensitive Info" for audio of religious
  singing; the data types above (Audio Data, User Content, Device ID) are the
  accurate primary classifications.
- The **honest, defensible position**: we disclose in onboarding and Settings
  exactly what is captured, it is strictly opt-in (triple-gated, default off),
  anonymous, and deletable on demand. The privacy *policy* text (required URL
  for App Store Connect) should state the religious-content nature plainly.

**Owed before submit:** a hosted **privacy policy URL** (App Store Connect
requires one even for TestFlight external testing). Draft lives at
`docs/privacy_policy.md` (TODO — write next), host on GitHub Pages or the
project site.

---

## Quick checklist to mark "Complete" in App Store Connect

- [ ] Audio Data → collected, linked, not tracking, App Functionality + ML note
- [ ] Other User Content → collected, linked, not tracking, App Functionality
- [ ] Device ID → collected, linked, not tracking, App Functionality
- [ ] Everything else → not collected
- [ ] Tracking → No
- [ ] Privacy policy URL → hosted and pasted
