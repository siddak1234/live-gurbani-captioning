# OOS v1 — curation playbook

Companion to [`eval_data/oos_v1/README.md`](../eval_data/oos_v1/README.md). The README explains *what* the OOS pack looks like (directory layout, JSON schema, eval workflow). This doc explains *how to pick what goes into it* so the resulting accuracy number is interpretable, not a random sample.

A 5-case OOS pack with arbitrary picks is barely better than no OOS pack — variance per shabad is high, one outlier skews the mean. The picks have to be deliberate.

## Selection rule: 3 representative + 2 stress

The five v1 cases split into two roles:

**3 representative cases (slots 1–3)** — match the median expected production input. The model should handle these cleanly; failure here means the model is broken, not the data set is hard.

Selection criteria:
- Common raga (ਆਸਾ, ਗਉੜੀ, ਸੋਰਠਿ, ਧਨਾਸਰੀ — most-frequent in SGGS by line count)
- Mid-tempo (60–100 BPM); rhythmic but not rushed
- Mainstream ragi (Bhai Harjinder Singh Srinagar, Bhai Niranjan Singh, Bhai Anantvir Singh, etc.) — clean studio or well-mic'd darbar recording
- 60–180 s clip length
- Shabad has at least 6 distinct pangtis (lines) so segment-level scoring has signal

**2 stress cases (slots 4–5)** — surface specific failure modes. Slot 4 stresses *acoustic robustness*, slot 5 stresses *line-tracking* (the matcher/smoother side).

- **Slot 4: acoustic stressor.** One of: fast tempo (≥ 140 BPM), heavy harmonium dominating vocal, live recording with audience noise, or a vocal style with high vibrato that confounds the ASR (e.g., gurmat sangeet recordings with classical alaap before the shabad).
- **Slot 5: matcher stressor.** One of: refrain-heavy shabad where the rahao repeats many times (loop-aware HMM should help here, Path A's smoother historically gets confused); a shabad whose first pangti shares vocabulary with a different shabad in BaniDB (collision-prone); or a shabad with very short lines where the per-chunk scoring window straddles boundaries.

The two stress cases are NOT meant to make the model look bad — they're meant to surface *failure modes* that the representative cases won't catch. If the model scores 95% on slots 1–3 and 40% on slot 4, that's a useful, actionable diagnostic. A mean of 84% would hide it.

## Hard constraints

Every candidate must satisfy:

1. **Shabad ID NOT in `{4377, 1821, 1341, 3712}`** — the paired benchmark's holdout. Verified by [`configs/datasets.yaml`](../configs/datasets.yaml) `holdout.shabad_ids`.
2. **Recording NOT in the training pulls.** Add the recording's source identifier (YouTube video_id, Sikhnet Radio asset id, archive.org item) to `holdout.video_ids` BEFORE running any training pull on a dataset that might include it.
3. **Audio license compatible with internal eval use.** We do not redistribute the recordings; we link to the source via the GT JSON's `source_url`. Sikhnet Radio archive (CC-licensed), archive.org Gurbani Kirtan collection (mixed), and creator-uploaded YouTube (with attribution) are all acceptable for *internal* OOS measurement.
4. **At least one verified BaniDB entry per pangti in the segments.** No "best guess" labels — every `line_idx` must trace to a real BaniDB line via `verse_id`.

## Candidate pool (starter — to be confirmed per case)

These are *suggestions*, not picks. Pick the final 5 from this pool (or substitute equivalents) after listening to the audio and verifying BaniDB alignment.

**Representative candidates (pool for slots 1–3):**

| Pool ID | Suggested shabad | Raga | Why representative |
|---|---|---|---|
| R1 | ਜਪੁ — opening (Mool Mantar) | Asa-adjacent (Salok) | Universally familiar; clean baseline |
| R2 | ਆਸਾ ਕੀ ਵਾਰ — any non-benchmark pauri | Asa | Mid-tempo, popular morning recordings |
| R3 | ਸੋ ਦਰੁ ਰਾਗੁ ਆਸਾ ਮਹਲਾ ੧ | Asa | Mainstream evening rehras shabad |
| R4 | ਸੁਖਮਨੀ ਸਾਹਿਬ — any non-benchmark ashtpadi | Gauri | Slow, clear; many recordings |
| R5 | ਅਨੰਦੁ ਸਾਹਿਬ — pauri 1 | Ramkali | Distinct opening line; reusable |

**Acoustic stressor candidates (pool for slot 4):**

| Pool ID | Suggested shabad | Stress dimension | Why it's a stressor |
|---|---|---|---|
| A1 | Asa di Vaar with classical alaap intro (e.g., Bhai Baldeep Singh recordings) | Long non-shabad intro | Tests whether the engine waits for real lyrics vs hallucinates an early shabad ID |
| A2 | Any fast rhythm (≥ 140 BPM) — popular Bhai Anantvir or Bhai Maninder Singh selections | Tempo | Whisper-small's chunk granularity vs fast line transitions |
| A3 | Live darbar recording with sangat singing along | Multi-speaker | ASR confidence drops; matcher should still snap to canonical |
| A4 | Harmonium-heavy recording (any with prominent baja) | Acoustic confound | Tests whether vocal isolation matters (Path A's Demucs probe was inconclusive) |

**Matcher stressor candidates (pool for slot 5):**

| Pool ID | Suggested shabad | Stress dimension | Why it's a stressor |
|---|---|---|---|
| M1 | A shabad with 4+ rahao repetitions (refrain-heavy) | Loop-aware HMM | Path A's smoother historically wobbles on rahao loops; Path B claims to fix this |
| M2 | A short-line shabad (e.g., ਪਉੜੀਆਂ with avg line ≤ 3s) | Sub-chunk lines | Per-chunk voting window straddles multiple lines |
| M3 | A shabad whose first pangti uses common vocabulary (e.g., starts with "ਸਤਿਗੁਰ ਸਾਚੇ" or similar high-IDF token) | Initial-line ambiguity | Shabad ID is hardest at the start; this stresses the blind buffer |

## Labor budget

Per case, from "I have a candidate" to "GT JSON committed":

| Step | Time | Blocking deps |
|---|---|---|
| Source audio (download + verify license) | 5–10 min | yt-dlp / ffmpeg |
| Convert to 16 kHz mono WAV | 1 min | ffmpeg |
| Bootstrap labels via Path A v3.2 oracle mode | 2–5 min compute | torch, transformers, peft, **brew Python 3.12** |
| Hand-correct line boundaries against audio (the actual labor) | 15–25 min | audio editor + BaniDB API access |
| Verify `verse_id` and `banidb_gurmukhi` fields per segment | 5–10 min | BaniDB lookups |
| Commit GT JSON | 1 min | none |

Total per case: **~30–45 minutes**. Five cases: **~3 hours of focused labor.**

On the current dev Mac, Python 3.12, ffmpeg, yt-dlp, torch, transformers, PEFT, and accelerate are installed and validated by Phase 2. The remaining blockers are curation labor and a small helper extension for arbitrary OOS audio URLs.

## Execution order — what to do first

1. **Now:** confirm the 5 picks from the pool above. Listen to candidate recordings on Sikhnet Radio or YouTube. Note the audio URLs.
2. **Next code PR:** extend or add an audio-fetch helper that accepts arbitrary source URLs (the existing benchmark fetch path is hardcoded to the 4 paired-benchmark videos).
3. **After audio fetch:** bootstrap GT JSONs via `eval_oos.py --oracle` for each case.
4. **Manual review:** open each bootstrap JSON in your editor, listen along, correct line boundaries. Save under `eval_data/oos_v1/test/`.
5. **Lock in:** add each recording's `video_id` (or equivalent source identifier) to `configs/datasets.yaml` → `holdout.video_ids` so it's never accidentally pulled into training.
6. **Baseline:** run `eval_oos.py --engine-config configs/inference/v3_2.yaml` against the curated pack. The v3.2 score is the v0 OOS number. Phase 2 fine-tunes must beat this on average AND not regress catastrophically on any single case.

## What `oos_v1` is NOT

- **It is not the production accuracy metric.** 5 cases is too few. The phased program calls for `oos_v2` at Phase 5 with ≥ 20 shabads + ≥ 5 ragis. `oos_v1` is the fast-iteration eval that catches major regressions while Phase 2–4 train.
- **It is not statistically rigorous.** Per-case variance is high. Treat per-case scores as diagnostic, not averageable. A useful rule of thumb: if `oos_v1` mean moves > 5 pts between adapters, that's signal. < 5 pts is noise.
- **It is not a substitute for the paired benchmark.** The benchmark is the apples-to-apples cross-engine comparison. The OOS set is the honesty check.

The right way to think about `oos_v1`: it's the first independent observation you'd refuse to ship a model without. The first observation you should never trust completely.
