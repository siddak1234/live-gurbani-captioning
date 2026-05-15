#!/usr/bin/env python3
"""LoRA fine-tuning of an ASR model for Path B (CTC) or Whisper-family acoustic models.

PEFT (parameter-efficient fine-tuning) trains a tiny adapter (~1% of params)
rather than the full model. With LoRA we only train low-rank decompositions
of attention matrices — small enough to fit alongside the frozen base on
modest GPUs.

Supports two model families:
  - CTC models (default): w2v-bert-punjabi, MMS, wav2vec2 variants. Uses
    AutoModelForCTC + Trainer + CTC loss. Default target_modules cover
    w2v-bert's linear_q/k/v/out attention projections.
  - Whisper / Seq2Seq models (surt-small-v3, openai/whisper-*): uses
    AutoModelForSpeechSeq2Seq + Seq2SeqTrainer with token-level CE loss.
    Default target_modules are Whisper's q_proj/k_proj/v_proj/out_proj.

Auto-detection: model_ids containing "whisper" or "surt" route to the
Seq2Seq path. Override with --model-type {ctc,whisper}.

Usage examples:

  # CTC fine-tune (existing behavior — w2v-bert default)
  python scripts/finetune_path_b.py \\
      --manifest training_data/manifest.json \\
      --output-dir lora_adapters/kirtan_v1 \\
      --epochs 3 --batch-size 2 --lr 5e-5

  # Whisper fine-tune (surt-small-v3)
  python scripts/finetune_path_b.py \\
      --model-id surindersinghssj/surt-small-v3 \\
      --manifest training_data/manifest.json \\
      --output-dir lora_adapters/surt_kirtan_v1 \\
      --epochs 3 --batch-size 4 --lr 1e-5 --bf16

Designed to run on Apple Silicon (MPS, CTC path validated) or CUDA GPU
(both paths). On CUDA, --bf16 is recommended (auto-enabled by default).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _is_whisper_model(model_id: str, override: str = "auto") -> bool:
    """Detect whether to route through the Whisper Seq2Seq path.

    Override values:
      - "ctc": force CTC path
      - "whisper": force Whisper path
      - "auto": detect from model_id
    """
    if override == "whisper":
        return True
    if override == "ctc":
        return False
    mid = model_id.lower()
    return "whisper" in mid or "/surt-" in mid or mid.endswith("/surt") or "surt-small" in mid or "surt-medium" in mid


def _default_target_modules(model_type: str, user_override: str | None) -> list[str]:
    """Resolve LoRA target_modules based on model family or user override."""
    if user_override:
        return [m.strip() for m in user_override.split(",") if m.strip()]
    if model_type == "whisper":
        return ["q_proj", "k_proj", "v_proj", "out_proj"]
    # CTC default — w2v-bert / wav2vec2-bert attention naming
    return ["linear_q", "linear_k", "linear_v", "linear_out"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True,
                        help="Path to training manifest JSON")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True,
                        help="Where to save LoRA adapter weights")
    parser.add_argument("--model-id", default="kdcyberdude/w2v-bert-punjabi",
                        help="HF model ID for the base model")
    parser.add_argument("--model-type", choices=["auto", "ctc", "whisper"], default="auto",
                        help="Force model family. 'auto' detects from --model-id (default)")
    parser.add_argument("--lora-target-modules", default=None,
                        help="Comma-separated LoRA target module names. Default depends on "
                             "model type: CTC=linear_q/k/v/out, Whisper=q_proj/k_proj/v_proj/out_proj")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch * grad_accum)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="If >0, override epochs and run this many steps (smoke test)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-manifest", type=pathlib.Path, default=None,
                        help="Optional held-out manifest for periodic evaluation")
    parser.add_argument("--bf16", action="store_true", default=None,
                        help="Enable bf16 mixed precision. Auto-on when CUDA available, off otherwise.")
    parser.add_argument("--language", default="punjabi",
                        help="Whisper generation language tag (Whisper path only). Default: punjabi.")
    args = parser.parse_args()

    import torch

    is_whisper = _is_whisper_model(args.model_id, args.model_type)
    model_type_resolved = "whisper" if is_whisper else "ctc"
    target_modules = _default_target_modules(model_type_resolved, args.lora_target_modules)

    # bf16 auto-detection: enable on CUDA, disable otherwise.
    if args.bf16 is None:
        args.bf16 = bool(torch.cuda.is_available())

    print(f"Model: {args.model_id}")
    print(f"Type: {model_type_resolved} (detected={args.model_type == 'auto'})")
    print(f"LoRA target_modules: {target_modules}")
    print(f"bf16: {args.bf16}")

    if is_whisper:
        return _run_whisper_train(args, target_modules)
    return _run_ctc_train(args, target_modules)


# -----------------------------------------------------------------------------
# CTC path — existing behavior, preserved byte-for-byte for w2v-bert and friends.
# -----------------------------------------------------------------------------

def _run_ctc_train(args, target_modules: list[str]) -> int:
    import torch
    from transformers import (
        AutoFeatureExtractor,
        AutoModelForCTC,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    from src.path_b.dataset import load_manifest, to_hf_dataset

    print(f"Loading {args.model_id} (CTC)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)
    model = AutoModelForCTC.from_pretrained(args.model_id)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",  # CTC isn't a HF preset; FEATURE_EXTRACTION leaves CTC head intact
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading training manifest: {args.manifest}")
    train_records = load_manifest(args.manifest)
    print(f"  {len(train_records)} training records")
    train_ds = to_hf_dataset(train_records, tokenizer, feature_extractor)

    eval_ds = None
    if args.eval_manifest:
        eval_records = load_manifest(args.eval_manifest)
        print(f"  {len(eval_records)} eval records")
        eval_ds = to_hf_dataset(eval_records, tokenizer, feature_extractor)

    audio_key = "input_features" if "input_features" in train_ds.column_names else "input_values"

    import numpy as np

    def data_collator(features: list[dict]) -> dict:
        max_label_len = max(len(f["labels"]) for f in features)
        feats_np = [np.asarray(f[audio_key], dtype=np.float32) for f in features]
        if audio_key == "input_features":
            max_time = max(a.shape[0] for a in feats_np)
            feat_dim = feats_np[0].shape[1]
            audio_batch = torch.zeros(len(features), max_time, feat_dim, dtype=torch.float32)
            attn_mask = torch.zeros(len(features), max_time, dtype=torch.long)
            for i, a in enumerate(feats_np):
                t = a.shape[0]
                audio_batch[i, :t] = torch.from_numpy(a)
                attn_mask[i, :t] = 1
        else:
            max_len = max(a.shape[0] for a in feats_np)
            audio_batch = torch.zeros(len(features), max_len, dtype=torch.float32)
            attn_mask = torch.zeros(len(features), max_len, dtype=torch.long)
            for i, a in enumerate(feats_np):
                n = a.shape[0]
                audio_batch[i, :n] = torch.from_numpy(a)
                attn_mask[i, :n] = 1

        labels = torch.stack([
            torch.nn.functional.pad(
                torch.tensor(f["labels"]),
                (0, max_label_len - len(f["labels"])),
                value=-100,
            ) for f in features
        ])
        return {audio_key: audio_batch, "attention_mask": attn_mask, "labels": labels}

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs if args.max_steps == 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.save_steps if eval_ds else None,
        report_to=[],
        gradient_checkpointing=False,
        fp16=False,
        bf16=args.bf16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nStarting CTC training (device={device})...")
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    feature_extractor.save_pretrained(str(args.output_dir))
    (args.output_dir / "base_model.txt").write_text(args.model_id + "\n")
    print(f"\nSaved LoRA adapter to {args.output_dir}")
    return 0


# -----------------------------------------------------------------------------
# Whisper path — new. Routes surt-small-v3 and openai/whisper-* fine-tunes
# through AutoProcessor + AutoModelForSpeechSeq2Seq + Seq2SeqTrainer.
# Untested on CTC test suite by design — Whisper-only.
# -----------------------------------------------------------------------------

def _run_whisper_train(args, target_modules: list[str]) -> int:
    import torch
    from transformers import (
        AutoProcessor,
        AutoModelForSpeechSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    from src.path_b.dataset import load_manifest, to_hf_dataset_whisper

    print(f"Loading {args.model_id} (Whisper Seq2Seq)...")
    processor = AutoProcessor.from_pretrained(args.model_id, language=args.language, task="transcribe")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id)

    # Pin generation language/task on the model so decoding follows the same setting.
    if hasattr(model, "generation_config"):
        model.generation_config.language = args.language
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading training manifest: {args.manifest}")
    train_records = load_manifest(args.manifest)
    print(f"  {len(train_records)} training records")
    train_ds = to_hf_dataset_whisper(train_records, processor)

    eval_ds = None
    if args.eval_manifest:
        eval_records = load_manifest(args.eval_manifest)
        print(f"  {len(eval_records)} eval records")
        eval_ds = to_hf_dataset_whisper(eval_records, processor)

    import numpy as np

    def data_collator(features: list[dict]) -> dict:
        # Whisper produces fixed-shape input_features (n_mels, n_frames) — pad to batch max.
        feats_np = [np.asarray(f["input_features"], dtype=np.float32) for f in features]
        max_t = max(a.shape[-1] for a in feats_np)
        n_mels = feats_np[0].shape[0]
        audio_batch = torch.zeros(len(features), n_mels, max_t, dtype=torch.float32)
        for i, a in enumerate(feats_np):
            audio_batch[i, :, :a.shape[-1]] = torch.from_numpy(a)
        max_label_len = max(len(f["labels"]) for f in features)
        labels = torch.stack([
            torch.nn.functional.pad(
                torch.tensor(f["labels"]),
                (0, max_label_len - len(f["labels"])),
                value=-100,
            ) for f in features
        ])
        return {"input_features": audio_batch, "labels": labels}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs if args.max_steps == 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.save_steps if eval_ds else None,
        report_to=[],
        gradient_checkpointing=False,
        fp16=False,
        bf16=args.bf16,
        predict_with_generate=False,  # speeds up training; switch on for WER eval at large scale
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nStarting Whisper Seq2Seq training (device={device})...")
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    (args.output_dir / "base_model.txt").write_text(args.model_id + "\n")
    print(f"\nSaved LoRA adapter to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
