#!/usr/bin/env python3
"""Build a tiny manifest for smoke-testing the fine-tune pipeline.

Extracts 4 short audio snippets (5s each) from one benchmark recording —
deliberately a recording whose shabad we already know — and pairs them
with the canonical line text at those time spans, derived from the GT.

THIS IS A PIPELINE-VALIDATION SMOKE MANIFEST ONLY. The resulting
LoRA adapter from training on it would be useless and must never be
evaluated against the benchmark (it'd be training on test data).

Real training data comes from elsewhere — see CLAUDE.md / Phase B plan.
"""

from __future__ import annotations

import json
import pathlib
import sys

import soundfile as sf


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    gt_path = repo_root.parent / "live-gurbani-captioning-benchmark-v1" / "test" / "IZOsmkdmmcg.json"
    audio_path = repo_root / "audio" / "IZOsmkdmmcg_16k.wav"
    out_dir = repo_root / "training_data" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = json.loads(gt_path.read_text())
    audio, sr = sf.read(str(audio_path), dtype="float32")
    assert sr == 16000

    # Take 4 short snippets each 5 seconds long, centered in different GT segments.
    snippets = []
    for i, seg in enumerate(gt["segments"][:4]):
        center = (seg["start"] + seg["end"]) / 2
        start_sample = max(0, int((center - 2.5) * sr))
        end_sample = min(len(audio), start_sample + 5 * sr)
        clip = audio[start_sample:end_sample]
        clip_path = out_dir / f"snippet_{i:02d}.wav"
        sf.write(str(clip_path), clip, sr)
        snippets.append({
            "audio": clip_path.name,  # relative to manifest
            "text": seg.get("banidb_gurmukhi", "").strip(),
        })

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(snippets, ensure_ascii=False, indent=2))
    print(f"Wrote smoke manifest with {len(snippets)} snippets to {manifest_path}")
    for s in snippets:
        print(f"  {s['audio']}: {s['text'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
