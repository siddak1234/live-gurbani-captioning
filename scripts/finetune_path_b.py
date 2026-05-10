#!/usr/bin/env python3
"""LoRA fine-tuning of a CTC acoustic model (default w2v-bert-punjabi) for Path B.

PEFT (parameter-efficient fine-tuning) trains a tiny adapter (~1% of params)
rather than the full model. With LoRA we only train low-rank decompositions
of attention matrices — small enough to fit alongside the frozen 600M base on
Apple Silicon, and small enough that 10-20 hours of in-domain data avoids
overfitting.

The fine-tuned adapter is saved alongside the base model reference. Inference
(via Path B's encoder) loads the base model and applies the adapter on top.

Expected workflow:
  1. Build a training manifest JSON listing (audio, text) pairs.
  2. Run this script with --manifest pointing at it.
  3. Save adapter weights to --output-dir.
  4. At inference, pass --adapter-dir to run_path_b_hmm.py (TODO wiring).

Usage:

  python scripts/finetune_path_b.py \\
      --manifest training_data/manifest.json \\
      --output-dir lora_adapters/kirtan_v1 \\
      --epochs 3 --batch-size 2 --lr 5e-5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True,
                        help="Path to training manifest JSON")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True,
                        help="Where to save LoRA adapter weights")
    parser.add_argument("--model-id", default="kdcyberdude/w2v-bert-punjabi",
                        help="HF model ID for the base CTC model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device batch size (small for Mac MPS / unified memory)")
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
    args = parser.parse_args()

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

    # Load tokenizer + feature extractor.
    print(f"Loading {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)
    model = AutoModelForCTC.from_pretrained(args.model_id)

    # Apply LoRA. Target the attention projection matrices, which is where
    # most of wav2vec2-bert's representational variance lives. Tweak
    # target_modules for other architectures.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["linear_q", "linear_k", "linear_v", "linear_out"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",  # CTC isn't a HF preset; FEATURE_EXTRACTION leaves CTC head intact
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Build datasets.
    print(f"Loading training manifest: {args.manifest}")
    train_records = load_manifest(args.manifest)
    print(f"  {len(train_records)} training records")
    train_ds = to_hf_dataset(train_records, tokenizer, feature_extractor)

    eval_ds = None
    if args.eval_manifest:
        eval_records = load_manifest(args.eval_manifest)
        print(f"  {len(eval_records)} eval records")
        eval_ds = to_hf_dataset(eval_records, tokenizer, feature_extractor)

    # Data collator handles both input key conventions:
    # - "input_values" (raw audio): wav2vec2 / MMS family
    # - "input_features" (mel features, 2D): w2v-bert / SeamlessM4T family
    audio_key = "input_features" if "input_features" in train_ds.column_names else "input_values"

    import numpy as np

    def data_collator(features: list[dict]) -> dict:
        max_label_len = max(len(f["labels"]) for f in features)
        # HF datasets stores numpy arrays as nested lists by default. Re-cast.
        feats_np = [np.asarray(f[audio_key], dtype=np.float32) for f in features]
        if audio_key == "input_features":
            # 2D arrays: (time, feature_dim). Pad along time.
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
                value=-100,  # CTC ignore index
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
        gradient_checkpointing=False,  # not stable on MPS in all transformers versions
        fp16=False,
        bf16=False,
        # MPS auto-detected by recent transformers; no flag needed.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    print(f"\nStarting training (device={'mps' if torch.backends.mps.is_available() else 'cpu'})...")
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    feature_extractor.save_pretrained(str(args.output_dir))
    (args.output_dir / "base_model.txt").write_text(args.model_id + "\n")
    print(f"\nSaved LoRA adapter to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
