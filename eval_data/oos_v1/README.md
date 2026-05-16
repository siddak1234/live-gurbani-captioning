# OOS evaluation set — v1

This directory holds **out-of-set** (OOS) audio + ground truth for honest accuracy measurement. "Out-of-set" means **shabads not in the paired benchmark's 4 test cases** and **recordings not used in any training run**.

A model that scores 95% on the paired benchmark but 60% here is benchmark-overfit. We need both numbers to claim a real result.

## Directory layout

```
eval_data/oos_v1/
├── README.md          # this file (committed)
├── audio/             # 16 kHz mono WAVs (GITIGNORED — copyright, size)
│   ├── case_001_16k.wav
│   ├── case_002_16k.wav
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
3. Convert to 16 kHz mono WAV:
   ```bash
   ffmpeg -i input.mp4 -ar 16000 -ac 1 -y eval_data/oos_v1/audio/case_NNN_16k.wav
   ```
4. Curate the GT JSON. Recommended workflow:
   - Run Path A v3.2 in oracle mode (shabad_id known) to bootstrap timings:
     ```bash
     python scripts/eval_oos.py \
       --data-dir eval_data/oos_v1 \
       --pred-dir /tmp/bootstrap_v3_2 \
       --engine-config configs/inference/v3_2.yaml \
       --oracle
     ```
   - Open the resulting JSON in your editor.
   - Hand-correct the line boundaries against the audio. This takes ~10-15 minutes per recording.
   - Save as `eval_data/oos_v1/test/case_NNN.json`.
5. Re-run `eval_oos.py` without `--oracle` to get the real blind+live score.

## Running an evaluation

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

## Holdout discipline

When any of these recordings is used in a training pull, that case must be **removed from this set** for honest evaluation. The training pipeline's holdout (`configs/datasets.yaml` → `holdout.video_ids`) should grow whenever this set does.
