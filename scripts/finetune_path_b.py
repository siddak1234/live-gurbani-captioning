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
import os
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


def _load_yaml_defaults(config_path: pathlib.Path) -> dict:
    """Load training hyperparameters from a YAML config file.

    Keys must match argparse ``dest`` names (i.e. underscores, not hyphens).
    Values of ``None`` / ``null`` are ignored so YAML can declare a key while
    leaving its actual default to the script's own auto-detection.
    """
    import yaml
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return {k: v for k, v in raw.items() if v is not None}


def _resolve_report_to(arg: str | list | None) -> list[str]:
    """Resolve the ``--report-to`` argument to a concrete list HF Trainer accepts.

    "auto" — wandb if WANDB_API_KEY is set and WANDB_MODE != "disabled";
             else tensorboard if the package is importable;
             else [] (no tracking).
    "none" / "" — [] explicitly.
    Anything else — comma-split list passed through verbatim.

    A list already (e.g. set by YAML) — returned as-is.
    """
    if isinstance(arg, list):
        return arg
    if arg is None:
        arg = "auto"
    s = str(arg).strip().lower()
    if s == "auto":
        if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE", "").lower() != "disabled":
            return ["wandb"]
        try:
            import tensorboard  # noqa: F401
            return ["tensorboard"]
        except ImportError:
            return []
    if s in ("none", "off", ""):
        return []
    return [t.strip() for t in str(arg).split(",") if t.strip()]


def _resolve_warmup(args) -> tuple[float, int]:
    """Return (warmup_ratio, warmup_steps) with ratio winning when > 0.

    HF TrainingArguments accepts both; if both are non-zero, warmup_ratio takes
    precedence (HF behavior). We mirror that here and zero out the unused one
    so logs and run_card.json show the actual schedule.
    """
    ratio = float(getattr(args, "warmup_ratio", 0.0) or 0.0)
    if ratio > 0:
        return ratio, 0
    return 0.0, int(args.warmup_steps)


def _validate_eval_save_steps(args, parser: argparse.ArgumentParser) -> None:
    """HF Trainer requires save_steps to be a multiple of eval_steps when
    load_best_model_at_end is true. Surface this at argparse time with a clear
    message instead of letting Trainer fail mid-init.
    """
    if not getattr(args, "load_best_model_at_end", False):
        return
    if args.eval_strategy == "no":
        parser.error("load_best_model_at_end=true requires eval_strategy != 'no'")
    if args.save_steps % args.eval_steps != 0:
        parser.error(
            f"save_steps ({args.save_steps}) must be a multiple of eval_steps "
            f"({args.eval_steps}) when load_best_model_at_end=true"
        )


def main() -> int:
    # Two-pass parsing: first peek at --config so its keys can become defaults
    # for the real parser; then re-parse with CLI args overriding YAML values.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=pathlib.Path, default=None)
    pre_args, _ = pre.parse_known_args()
    yaml_defaults: dict = {}
    if pre_args.config:
        yaml_defaults = _load_yaml_defaults(pre_args.config)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=None,
                        help="YAML config providing defaults; CLI args still override.")
    parser.add_argument("--manifest", type=pathlib.Path, default=None,
                        help="Path to training manifest JSON (required unless set in --config)")
    parser.add_argument("--output-dir", type=pathlib.Path, default=None,
                        help="Where to save LoRA adapter weights (required unless set in --config)")
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
    parser.add_argument("--warmup-ratio", type=float, default=0.0,
                        help="Fraction of total steps used for warmup. When > 0, supersedes --warmup-steps.")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-manifest", type=pathlib.Path, default=None,
                        help="Optional held-out manifest for periodic evaluation")
    parser.add_argument("--bf16", action="store_true", default=None,
                        help="Enable bf16 mixed precision. Auto-on when CUDA available, off otherwise.")
    parser.add_argument("--fp16", action="store_true", default=None,
                        help="Enable fp16 mixed precision. Auto-on when MPS available (and bf16 off), off otherwise.")
    parser.add_argument("--language", default="punjabi",
                        help="Whisper generation language tag (Whisper path only). Default: punjabi.")

    # -- Reproducibility --
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for torch/numpy/random + HF dataloader. MPS isn't bit-deterministic.")

    # -- Regularization --
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="AdamW weight decay. HF default 0.0; PEFT-Whisper FT literature uses 0.01.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping threshold (HF default).")

    # -- LR schedule --
    parser.add_argument("--lr-scheduler-type", default="linear",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                                 "constant", "constant_with_warmup", "inverse_sqrt"],
                        help="LR decay shape. HF default is 'linear'; cosine is the Whisper FT norm.")

    # -- Memory --
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False,
                        help="Trade ~20%% throughput for ~30%% activation memory. Off by default.")

    # -- Eval + early stopping --
    parser.add_argument("--eval-strategy", default="no", choices=["no", "steps", "epoch"],
                        help="Trainer eval strategy. 'no' disables; 'steps' uses --eval-steps.")
    parser.add_argument("--eval-steps", type=int, default=500,
                        help="Eval cadence when --eval-strategy=steps.")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop training after N evals without improvement. 0 disables. "
                             "Requires --eval-strategy != 'no' and --load-best-model-at-end.")
    parser.add_argument("--load-best-model-at-end", action="store_true", default=False,
                        help="Restore the best-eval checkpoint at end of training. Required for early stopping.")

    # -- Tracking --
    parser.add_argument("--report-to", default="auto",
                        help="Experiment tracker(s). 'auto' picks wandb if WANDB_API_KEY set, "
                             "else tensorboard if importable, else none. 'none' disables. "
                             "Comma-list passes through to HF Trainer.")
    parser.add_argument("--wandb-project", default="kirtan-asr",
                        help="WANDB_PROJECT — used when report_to resolves to include wandb.")
    parser.add_argument("--run-name", default=None,
                        help="Trainer run_name + wandb run name. Null auto-generates from config hash + timestamp.")

    # Apply YAML defaults BEFORE parse_args so CLI args still win.
    if yaml_defaults:
        known = {a.dest for a in parser._actions}
        unknown = set(yaml_defaults) - known
        if unknown:
            print(f"Warning: ignoring unknown YAML keys: {sorted(unknown)}", file=sys.stderr)
        parser.set_defaults(**{k: v for k, v in yaml_defaults.items() if k in known})

    args = parser.parse_args()

    # Convert YAML-supplied paths from str to pathlib.Path (set_defaults bypasses type=).
    if args.manifest is not None and not isinstance(args.manifest, pathlib.Path):
        args.manifest = pathlib.Path(args.manifest)
    if args.output_dir is not None and not isinstance(args.output_dir, pathlib.Path):
        args.output_dir = pathlib.Path(args.output_dir)

    # Validate required-ish args (kept non-required at argparse level so YAML can provide them).
    if args.manifest is None:
        parser.error("--manifest is required (provide on CLI or in --config)")
    if args.output_dir is None:
        parser.error("--output-dir is required (provide on CLI or in --config)")

    # Validate eval/save coupling at argparse time (HF fails late and obscurely otherwise).
    _validate_eval_save_steps(args, parser)

    import torch
    from transformers import set_seed

    # Seed torch/numpy/random/python-hash + HF Trainer's dataloader generator.
    # NOTE: we do NOT call torch.use_deterministic_algorithms(True) — many MPS
    # ops would fail. Same-seed runs match within ~2% on MPS, ~0.5% on CPU.
    set_seed(args.seed)
    # Belt-and-suspenders: older transformers releases of set_seed didn't cover
    # MPS RNG. Harmless on newer versions; correct on older ones.
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)

    # Resolve tracking + project env BEFORE we instantiate Trainer.
    args.report_to_resolved = _resolve_report_to(args.report_to)
    if "wandb" in args.report_to_resolved:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # Resolve warmup (ratio wins over steps when > 0).
    args.warmup_ratio_resolved, args.warmup_steps_resolved = _resolve_warmup(args)

    is_whisper = _is_whisper_model(args.model_id, args.model_type)
    model_type_resolved = "whisper" if is_whisper else "ctc"
    target_modules = _default_target_modules(model_type_resolved, args.lora_target_modules)

    # Precision auto-detection.
    #   - bf16: CUDA-only feature; auto-on when CUDA is available.
    #   - fp16: Apple-MPS-friendly mixed precision; auto-on when MPS is available
    #     and bf16 is not in play. CPU stays at fp32.
    if args.bf16 is None:
        args.bf16 = bool(torch.cuda.is_available())
    if args.fp16 is None:
        args.fp16 = (not args.bf16) and bool(torch.backends.mps.is_available())

    print(f"Model: {args.model_id}")
    print(f"Type: {model_type_resolved} (detected={args.model_type == 'auto'})")
    print(f"LoRA target_modules: {target_modules}")
    print(f"Precision: bf16={args.bf16}, fp16={args.fp16}")
    print(f"Seed: {args.seed}")
    print(f"Scheduler: {args.lr_scheduler_type} (warmup_ratio={args.warmup_ratio_resolved}, "
          f"warmup_steps={args.warmup_steps_resolved})")
    print(f"Tracking: report_to={args.report_to_resolved or '[]'}")

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

    # Eval strategy: explicit --eval-strategy wins; legacy fallback is "steps when eval_ds present, else no".
    effective_eval_strategy = args.eval_strategy
    if effective_eval_strategy == "no" and eval_ds is not None:
        effective_eval_strategy = "steps"

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs if args.max_steps == 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps_resolved,
        warmup_ratio=args.warmup_ratio_resolved,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=10,
        save_steps=args.save_steps,
        eval_strategy=effective_eval_strategy,
        eval_steps=args.eval_steps if effective_eval_strategy != "no" else None,
        load_best_model_at_end=args.load_best_model_at_end and eval_ds is not None,
        metric_for_best_model="eval_loss" if (args.load_best_model_at_end and eval_ds is not None) else None,
        greater_is_better=False if (args.load_best_model_at_end and eval_ds is not None) else None,
        report_to=args.report_to_resolved,
        run_name=args.run_name,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        data_seed=args.seed,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        if eval_ds is None:
            print("Warning: --early-stopping-patience > 0 but no --eval-manifest; ignoring.", file=sys.stderr)
        elif not args.load_best_model_at_end:
            print("Warning: --early-stopping-patience > 0 but --load-best-model-at-end is false; ignoring.",
                  file=sys.stderr)
        else:
            from transformers import EarlyStoppingCallback
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=callbacks or None,
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

    effective_eval_strategy = args.eval_strategy
    if effective_eval_strategy == "no" and eval_ds is not None:
        effective_eval_strategy = "steps"

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs if args.max_steps == 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps_resolved,
        warmup_ratio=args.warmup_ratio_resolved,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=10,
        save_steps=args.save_steps,
        eval_strategy=effective_eval_strategy,
        eval_steps=args.eval_steps if effective_eval_strategy != "no" else None,
        load_best_model_at_end=args.load_best_model_at_end and eval_ds is not None,
        metric_for_best_model="eval_loss" if (args.load_best_model_at_end and eval_ds is not None) else None,
        greater_is_better=False if (args.load_best_model_at_end and eval_ds is not None) else None,
        report_to=args.report_to_resolved,
        run_name=args.run_name,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        data_seed=args.seed,
        predict_with_generate=False,  # speeds up training; switch on for WER eval at large scale
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        if eval_ds is None:
            print("Warning: --early-stopping-patience > 0 but no --eval-manifest; ignoring.", file=sys.stderr)
        elif not args.load_best_model_at_end:
            print("Warning: --early-stopping-patience > 0 but --load-best-model-at-end is false; ignoring.",
                  file=sys.stderr)
        else:
            from transformers import EarlyStoppingCallback
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=callbacks or None,
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
