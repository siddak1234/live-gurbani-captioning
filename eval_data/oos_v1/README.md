# OOS evaluation set — v1

This directory holds **out-of-set** (OOS) audio + ground truth for honest accuracy measurement. "Out-of-set" means **shabads not in the paired benchmark's 4 test cases** and **recordings not used in any training run**.

A model that scores 95% on the paired benchmark but 60% here is benchmark-overfit. We need both numbers to claim a real result.

## Directory layout

```
eval_data/oos_v1/
├── README.md          # this file (committed)
├── cases.yaml         # OOS source manifest (committed)
├── audio/             # 16 kHz mono WAVs (GITIGNORED — copyright, size)
│   ├── case_001_16k.wav
│   ├── case_002_16k.wav
│   └── ...
├── drafts/            # machine-generated bootstrap labels (NOT scored)
│   ├── case_001.json
│   └── ...
└── test/              # ground truth JSONs (committed once curated)
    ├── case_001.json
    ├── case_002.json
    └── ...
```

The audio is **deliberately gitignored**. We do not redistribute kirtan recordings — fetch them from their original source (Sikhnet Radio archive, YouTube link in the GT JSON, etc.). The GT JSONs are small text files and ARE committed once human-curated.

## GT JSON shape (identical to the paired benchmark)

```json
{
  "video_id": "case_001",
  "shabad_id": 5621,
  "uem": {"start": 0.0, "end": 240.0},
  "source_url": "https://www.sikhnet.com/...",
  "segments": [
    {
      "start": 12.0,
      "end": 28.5,
      "line_idx": 1,
      "verse_id": "ABC1",
      "banidb_gurmukhi": "ਪਹਿਲੀ ਪੰਗਤੀ ਦਾ ਟੈਕਸਟ"
    },
    {
      "start": 28.5,
      "end": 42.0,
      "line_idx": 2,
      "verse_id": "DEF2",
      "banidb_gurmukhi": "ਦੂਜੀ ਪੰਗਤੀ ਦਾ ਟੈਕਸਟ"
    }
  ]
}
```

Required fields: `video_id`, `shabad_id`, `segments[*].{start, end, line_idx}`. Optional but strongly recommended: `verse_id`, `banidb_gurmukhi`, `uem.{start,end}`, `source_url`.

## How to add a new case

1. Pick a shabad NOT in `{4377, 1821, 1341, 3712}` and not present in any training pull.
2. Find a clean recording (Sikhnet Radio, archive.org/details/GurbaniKirtan, well-mic'd YouTube).
3. Cache the shabad corpus once the BaniDB ID is known:
   ```bash
   make corpus-oos OOS_SHABAD_ID=5621
   ```
4. Convert to 16 kHz mono WAV. Preferred path:
   ```bash
   make fetch-oos-audio \
     OOS_URL='case_001=https://youtube.com/watch?v=...' \
     OOS_CLIP='case_001=30-210'
   ```
   The `case_001` stem must match the eventual `test/case_001.json`.
   `OOS_CLIP` is optional but recommended for long source recordings; v1 cases
   should usually score a 60-180s curated window rather than an entire 10-minute
   YouTube upload.

   Manual fallback:
   ```bash
   ffmpeg -i input.mp4 -ss 30 -t 180 -ar 16000 -ac 1 -y eval_data/oos_v1/audio/case_NNN_16k.wav
   ```
5. Curate the GT JSON. Recommended workflow:
   - Generate draft labels in `drafts/`:
     ```bash
     make bootstrap-oos-gt
     ```
   - Build the local review page:
     ```bash
     make oos-review-pack
     ```
   - Open `eval_data/oos_v1/review/index.html` in a browser and the matching
     draft JSON in your editor.
   - Hand-correct the line boundaries against the audio. This takes ~10-15 minutes per recording.
   - Remove the draft marker by setting `curation_status` to `HUMAN_CORRECTED_V1`.
   - Make sure the file includes `total_duration`, `uem`, and every segment's
     `start`, `end`, `line_idx`, `verse_id`, and `banidb_gurmukhi`.
   - Save as `eval_data/oos_v1/test/case_NNN.json`.
6. Run `make validate-oos-gt`. It must pass before any OOS score is trusted.
7. Re-run `eval_oos.py` without `--oracle` to get the real blind+live score.

Never score or publish from `drafts/`. The evaluator only reads `test/`, and
`test/` is reserved for human-corrected ground truth.

## Running an evaluation

For the current best runtime architecture (`phase2_9_loop_align`), use the
ID-lock OOS target:

```bash
make eval-oos-loop-align
```

That command first runs `make validate-oos-gt`, then runs the same stack that
scored 91.2% on the paired benchmark:
faster-whisper word timestamps for pre-lock shabad ID, `v5b_mac_diverse` for
post-lock captions, retro-buffered finalization, and simran-aware null
alignment.

For older single-engine configs, use:

```bash
python scripts/eval_oos.py \
  --data-dir eval_data/oos_v1 \
  --pred-dir submissions/oos_v1_<engine_name> \
  --engine-config configs/inference/v3_2.yaml
```

The script invokes the paired benchmark's `eval.py` as a subprocess — same scorer, same metric (1s frame accuracy), no drift risk.

## Target size for v1

5-10 cases covering a mix of:
- Different ragis / vocal timbres
- Different recording conditions (studio, live darbar, phone recording)
- Different ragas / shabad lengths
- At least one rahao-heavy shabad (refrain repetition stress-tests the smoother)

Below 5 cases the numbers are too noisy. Above 10 the curation cost outweighs the marginal signal.

For the selection rule (3 representative + 2 stress), candidate pool, labor estimate, and audit-style failure modes, see the curation playbook: [`docs/oos_v1_curation.md`](../../docs/oos_v1_curation.md).
For the current five-case labeling queue, use the checkpoint worksheet:
[`docs/oos_v1_labeling_checkpoint.md`](../../docs/oos_v1_labeling_checkpoint.md).
For the parser-based triage of the current drafts, see:
[`docs/oos_v1_machine_audit.md`](../../docs/oos_v1_machine_audit.md).

## Holdout discipline

When any of these recordings is used in a training pull, that case must be **removed from this set** for honest evaluation. The training pipeline's holdout (`configs/datasets.yaml` → `holdout.video_ids`) should grow whenever this set does.
