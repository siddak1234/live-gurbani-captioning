# Privacy Policy — Sangat

**Effective date:** 2026-06-24
**Contact:** siddak1234@gmail.com

This is the privacy policy for **Sangat**, an iOS app that listens to live
kirtan (Sikh devotional singing) and displays the canonical Gurbani lines being
sung. This document is written to be hosted at a public URL (e.g. GitHub Pages)
and linked from the app's App Store Connect listing, as Apple requires.

---

## The short version

- **No account, no sign-in, no name or email is required to use Sangat.**
- **Nothing leaves your device unless you explicitly turn it on.** Every data-
  sharing feature is off by default.
- We never sell your data, never show ads, and never use third-party tracking.
- You can delete everything we hold about your device at any time, from inside
  the app.

---

## What the app does on your device

To caption kirtan, Sangat uses your device's **microphone** to listen to audio
and runs a speech-recognition model **entirely on your device**. This audio
processing is local: the raw microphone audio is **not transmitted anywhere** as
part of normal captioning.

## What we collect — and only if you opt in

Sangat has three independent privacy switches, **all OFF by default**, found in
Settings:

1. **Save corrections** — when you correct the app (for example, telling it the
   shabad it identified was wrong), the correction is stored on your device.
2. **Upload my corrections** — sends your stored corrections to our server so we
   can improve the captioning model. Includes an optional "Wi-Fi only" setting.
3. **Include a short audio clip** — attaches a brief audio snippet of the
   misidentified segment to an uploaded correction, so the model can learn from
   the actual sound.

If, and only if, you enable uploading, we collect:

| Data | What it is | Why |
|---|---|---|
| **Correction metadata** | Which shabad/line the app got wrong and what the correct one was, plus timestamps | To improve captioning accuracy |
| **Short audio clip** | A brief recording of the misidentified kirtan (only if you also enable "Include a short audio clip") | To improve the acoustic model |
| **Anonymous device ID** | A random identifier generated on your device | To avoid duplicate uploads and to let you delete your data |

The **anonymous device ID** is a random value created on your device and stored
in the iOS Keychain. It is **not** your name, email, phone number, Apple ID,
advertising identifier, or any other personal identifier. We cannot use it to
contact you or identify you personally; it exists only to group your own
uploads together so the "Delete my data" feature can find and remove them.

## What we do NOT collect

We do not collect your name, email, phone number, address, contacts, location,
browsing history, health data, financial data, or advertising identifiers. We
do not track you across apps or websites, and we do not use the App Tracking
Transparency framework because we do no tracking.

## Sensitive / religious content

Because Sangat captions Sikh scripture, any audio clips and corrections you
choose to upload are inherently related to **religious practice**. We treat this
data accordingly: it is strictly opt-in, anonymized, transmitted over encrypted
connections, used only to improve the app's captioning, and deletable on demand.
We never share it for advertising or any unrelated purpose.

## How your data is stored and shared

When you opt in to uploading, data is sent over an encrypted (HTTPS) connection
to our backend hosted on **Supabase** (a third-party cloud database provider
acting as our data processor). Audio clips are stored in a private storage
bucket; correction metadata is stored in a private database table. This data is
accessible only to the app's maintainers for the purpose of improving the model.
We do not sell, rent, or share this data with any other third party.

## Your choices and your right to delete

- You can turn any of the three switches off at any time in Settings.
- **"Delete on-device data"** removes all corrections and audio clips stored
  locally on your device.
- **"Delete my data"** sends a request to our server to delete all corrections
  and audio clips associated with your anonymous device ID. This is a true
  server-side deletion.

## Children's privacy

Sangat is a general-audience app and does not knowingly collect personal
information from children. Because we collect no personal identifiers at all,
the app does not build profiles of any user.

## Changes to this policy

If we change what we collect or how we use it, we will update this policy and
revise the effective date above. Material changes will be reflected in the app's
release notes.

## Contact

Questions about this policy or your data: siddak1234@gmail.com

---

### Hosting checklist (internal — remove before publishing if you like)

- [ ] Fill the effective date and contact email above.
- [ ] Publish at a stable public URL (GitHub Pages on this repo is simplest:
      enable Pages, or drop the rendered HTML in a `docs/` Pages site).
- [ ] Paste that URL into App Store Connect → app → App Information → Privacy
      Policy URL (required before TestFlight external testing).
- [ ] Keep the wording consistent with the nutrition-label answers in
      `docs/privacy_nutrition_labels.md`.
