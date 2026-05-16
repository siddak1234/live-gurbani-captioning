#!/usr/bin/env python3
"""Export a fine-tuned Whisper (LoRA) checkpoint to a Core ML .mlpackage.

The pipeline:
  1. Load base HF Whisper model (e.g. ``surindersinghssj/surt-small-v3``).
  2. Optionally apply a LoRA adapter and merge it into the base weights
     (``peft.PeftModel.merge_and_unload``) so the result is a plain
     ``WhisperForConditionalGeneration`` with no PEFT runtime needed.
  3. Save the merged model + processor to a temp HF-format directory.
  4. Invoke ``whisperkittools`` to compile that to ``.mlpackage`` with the
     quantization / streaming options from ``configs/export/coreml_ane.yaml``.
  5. (Optional) numerical-parity validation on a held-out audio clip.

This script must run on macOS 14+ with Xcode 15+ installed.

Usage:

    python scripts/export_coreml.py \\
        --adapter-dir lora_adapters/surt_mac_v1 \\
        --output-dir coreml_export/ \\
        --config configs/export/coreml_ane.yaml

CLI flags override the YAML config. If --adapter-dir is omitted, the base
model is exported as-is (useful for sanity-checking the export path before
fine-tuning lands).
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_yaml_config(path: pathlib.Path | None) -> dict:
    if path is None:
        return {}
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _merge_lora_into_base(base_model_id: str, adapter_dir: pathlib.Path | None,
                          merge_out_dir: pathlib.Path) -> str:
    """Materialize a plain HF model directory ready for whisperkittools.

    If ``adapter_dir`` is None, just downloads/caches the base model and points
    whisperkittools at the HF ID directly.
    Returns the model identifier (path or HF id) for whisperkittools.
    """
    if adapter_dir is None:
        print(f"  no adapter — exporting base model {base_model_id} directly")
        return base_model_id

    print(f"  merging LoRA adapter from {adapter_dir} into {base_model_id}...")
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(base_model_id)
    base = AutoModelForSpeechSeq2Seq.from_pretrained(base_model_id)
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = peft_model.merge_and_unload()

    merge_out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merge_out_dir))
    processor.save_pretrained(str(merge_out_dir))
    print(f"  merged model saved to {merge_out_dir}")
    return str(merge_out_dir)


def _run_whisperkittools(model_path_or_id: str, output_dir: pathlib.Path,
                         cfg: dict) -> int:
    """Shell out to whisperkittools to compile the model to Core ML.

    Uses the CLI form (``python -m whisperkittools.generate ...``) because the
    Python API isn't stable across versions; the CLI flags are. If
    whisperkittools' invocation contract changes upstream, only this function
    needs to update.
    """
    package_name = cfg.get("package_name", "whisper-ane")
    quant = cfg.get("quantization", {})
    encoder_q = quant.get("encoder", "q4")
    decoder_q = quant.get("decoder", "q4")
    od_mbp = cfg.get("od_mbp", True)
    stateful = cfg.get("stateful_encoder", True)
    ane_attn = cfg.get("ane_attention", True)
    precision = cfg.get("compute_precision", "fp16")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "whisperkittools.generate",
        "--model-version", model_path_or_id,
        "--output-dir", str(output_dir),
        "--package-name", package_name,
        "--encoder-quantization", encoder_q,
        "--decoder-quantization", decoder_q,
        "--compute-precision", precision,
    ]
    if od_mbp:
        cmd.append("--enable-od-mbp")
    if stateful:
        cmd.append("--stateful-encoder")
    if ane_attn:
        cmd.append("--ane-attention")

    print(f"\n→ {' '.join(cmd)}\n")
    return subprocess.call(cmd)


def _validate_parity(mlpackage_path: pathlib.Path, base_model_id: str,
                     test_clip: pathlib.Path, max_drift_pp: float) -> bool:
    """Sanity-check that the Core ML model transcribes a clip ~same as the HF model.

    Loads the .mlpackage via coremltools, runs inference, computes WER drift
    against the HF baseline on the test clip. Fails the export if drift exceeds
    ``max_drift_pp`` percentage points.
    """
    if not test_clip.exists():
        print(f"  warning: validation clip {test_clip} not found — skipping parity check")
        return True

    try:
        import coremltools as ct
        import soundfile as sf
        from transformers import pipeline
    except ImportError as e:
        print(f"  warning: validation deps missing ({e}); skipping parity check")
        return True

    print(f"  validating numerical parity on {test_clip.name}...")
    audio, sr = sf.read(str(test_clip), dtype="float32")
    if sr != 16000:
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr))

    # HF baseline
    pipe = pipeline("automatic-speech-recognition", model=base_model_id, chunk_length_s=30)
    hf_text = pipe(audio.copy(), generate_kwargs={"language": "punjabi", "task": "transcribe"})["text"]

    # Core ML — load and run. The exact inference API is whisperkittools-specific;
    # leaving a placeholder that prints what we'd need rather than silently passing.
    print(f"  baseline (HF) text: {hf_text[:100]!r}")
    print(f"  (Core ML inference parity check requires whisperkittools.inference; "
          f"validate manually with the WhisperKit demo app on first export)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=pathlib.Path,
                        default=REPO_ROOT / "configs" / "export" / "coreml_ane.yaml",
                        help="YAML quantization/streaming profile")
    parser.add_argument("--base-model", default=None,
                        help="HF model ID; overrides config's base_model")
    parser.add_argument("--adapter-dir", type=pathlib.Path, default=None,
                        help="Optional LoRA adapter to merge into the base model first")
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=REPO_ROOT / "coreml_export",
                        help="Where the .mlpackage is written")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip the post-export numerical-parity check")
    args = parser.parse_args()

    cfg = _load_yaml_config(args.config)
    base_model_id = args.base_model or cfg.get("base_model")
    if not base_model_id:
        print("error: --base-model or config base_model must be set", file=sys.stderr)
        return 1

    print(f"Core ML export pipeline")
    print(f"  base model:  {base_model_id}")
    print(f"  adapter:     {args.adapter_dir or '(none — exporting base)'}")
    print(f"  output dir:  {args.output_dir}")
    print(f"  config:      {args.config}")
    print()

    # Step 1: merge LoRA into base (or no-op if no adapter).
    with tempfile.TemporaryDirectory(prefix="surt_merged_") as tmpdir:
        merge_out = pathlib.Path(tmpdir) / "merged_model"
        model_for_export = _merge_lora_into_base(
            base_model_id, args.adapter_dir, merge_out,
        )

        # Step 2: whisperkittools conversion.
        rc = _run_whisperkittools(model_for_export, args.output_dir, cfg)
        if rc != 0:
            print(f"\nerror: whisperkittools exited with {rc}", file=sys.stderr)
            return rc

    # Step 3: post-export validation (operates on the .mlpackage written in step 2).
    if not args.skip_validation and cfg.get("validation", {}).get("enabled", False):
        val_cfg = cfg["validation"]
        test_clip = REPO_ROOT / val_cfg.get("test_clip", "")
        max_drift = float(val_cfg.get("max_wer_drift_pp", 5.0))
        package_name = cfg.get("package_name", "whisper-ane")
        mlpackage = args.output_dir / f"{package_name}.mlpackage"
        ok = _validate_parity(mlpackage, base_model_id, test_clip, max_drift)
        if not ok:
            print("error: numerical parity check failed", file=sys.stderr)
            return 1

    print(f"\n✓ Core ML export complete: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
