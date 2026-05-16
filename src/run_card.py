"""Run-card emission: per-adapter lineage for reproducible experiments.

A run card is the small JSON we write next to every trained LoRA adapter so
that, six months from now, we can answer "what produced this adapter?":

    {
      "git_sha": "...",
      "config_hash": "...",          # canonical hash of all argparse-resolved args
      "data_hash":   "...",          # hash of sorted (audio_path, text) tuples
      "seed": 42,
      "hostname": "...",
      "wall_clock_s": 12345.6,
      "peak_mem_gb": 18.7,
      "peak_mem_source": "mps_driver",  # mps_driver | cuda | psutil_rss
      "status": "completed",         # completed | crashed | interrupted
      "device": "mps",
      ...
    }

Design notes:

* ``data_hash`` covers (audio_path, text) tuples sorted lex — NOT audio file
  contents. Hashing 200+ audio files would dominate the training script's
  startup. Audio files are gitignored, so swap-on-disk is an operational risk,
  not a code-level one. If you need stronger guarantees, layer a content hash
  on top in a Phase 6+ refactor.

* ``write_run_card`` is written to be safe to call in a ``finally`` block.
  A run that crashed mid-train is forensically more valuable than no card.
  Pass ``status="crashed"`` and whatever partial state you have; missing
  fields land as ``None``.

* Peak memory is best-effort. MPS driver allocation is the most accurate
  number on Apple Silicon; CUDA's ``max_memory_allocated`` is canonical on
  NVIDIA; otherwise we fall back to RSS which over-counts (includes the
  Python process, not just tensors).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# Hashing helpers
# -----------------------------------------------------------------------------

def config_hash(args: Any) -> str:
    """SHA256 of canonical-JSON dump of every argparse attr.

    Filters out attrs whose names start with ``_`` (internal namespace) and
    coerces non-JSON values (Path, set) to strings/lists so the dump is stable
    across runs and Python versions.
    """
    payload: dict[str, Any] = {}
    src = vars(args) if hasattr(args, "__dict__") else dict(args)
    for k, v in sorted(src.items()):
        if k.startswith("_"):
            continue
        payload[k] = _to_jsonable(v)
    return _sha256_of(payload)


def data_hash(records: list[dict] | None) -> str | None:
    """SHA256 of sorted ``(audio, text)`` tuples drawn from manifest records.

    Returns ``None`` when ``records`` is None or empty so the caller can
    distinguish "we didn't measure" from "the manifest was empty".
    """
    if not records:
        return None
    pairs = sorted(
        (str(rec.get("audio", "")), str(rec.get("text", "")))
        for rec in records
    )
    return _sha256_of(pairs)


def _to_jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_jsonable(val) for k, val in sorted(v.items())}
    if isinstance(v, set):
        return sorted(_to_jsonable(x) for x in v)
    if isinstance(v, Path):
        return str(v)
    return str(v)


def _sha256_of(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# -----------------------------------------------------------------------------
# Environment probes
# -----------------------------------------------------------------------------

def git_sha(cwd: Path | None = None) -> str:
    """Current HEAD's short SHA, or ``"unknown"`` if not in a git work tree."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode("ascii", errors="replace").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "unknown"


@dataclass(frozen=True)
class PeakMem:
    gb: float | None
    source: str  # "mps_driver" | "cuda" | "psutil_rss" | "unavailable"


def peak_memory_gb() -> PeakMem:
    """Best-effort peak-memory probe. Never raises."""
    # CUDA: canonical when present.
    try:
        import torch
        if torch.cuda.is_available():
            bytes_ = int(torch.cuda.max_memory_allocated())
            return PeakMem(gb=bytes_ / (1024 ** 3), source="cuda")
    except Exception:
        pass

    # MPS driver allocation (recent torch on Apple Silicon).
    try:
        import torch
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            fn = getattr(torch.mps, "driver_allocated_memory", None)
            if fn is not None:
                bytes_ = int(fn())
                return PeakMem(gb=bytes_ / (1024 ** 3), source="mps_driver")
    except Exception:
        pass

    # Fallback: RSS via psutil. Over-counts (whole process), but always available.
    try:
        import psutil
        rss = psutil.Process().memory_info().rss
        return PeakMem(gb=rss / (1024 ** 3), source="psutil_rss")
    except Exception:
        return PeakMem(gb=None, source="unavailable")


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _final_losses(trainer_state: Any) -> tuple[float | None, float | None]:
    """Extract last train/eval loss from a transformers TrainerState (or None)."""
    if trainer_state is None:
        return None, None
    log_history = getattr(trainer_state, "log_history", None) or []
    train_loss: float | None = None
    eval_loss: float | None = None
    for entry in log_history:
        if not isinstance(entry, dict):
            continue
        if "loss" in entry and isinstance(entry["loss"], (int, float)):
            train_loss = float(entry["loss"])
        if "eval_loss" in entry and isinstance(entry["eval_loss"], (int, float)):
            eval_loss = float(entry["eval_loss"])
    return train_loss, eval_loss


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def write_run_card(
    output_dir: str | Path,
    *,
    args: Any,
    train_records: list[dict] | None,
    eval_records: list[dict] | None,
    trainer_state: Any = None,
    wall_clock_s: float | None = None,
    peak_mem: PeakMem | None = None,
    status: str = "completed",
    device: str | None = None,
    scores: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Emit ``<output_dir>/run_card.json``. Safe to call from a ``finally`` block.

    Returns the path written. Creates the directory if missing.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pm = peak_mem if peak_mem is not None else peak_memory_gb()
    final_train, final_eval = _final_losses(trainer_state)

    card: dict[str, Any] = {
        "git_sha": git_sha(),
        "config_hash": config_hash(args),
        "data_hash": data_hash(train_records),
        "eval_data_hash": data_hash(eval_records),
        "seed": getattr(args, "seed", None),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": _pkg_version("torch"),
        "transformers_version": _pkg_version("transformers"),
        "peft_version": _pkg_version("peft"),
        "device": device,
        "wall_clock_s": wall_clock_s,
        "peak_mem_gb": pm.gb,
        "peak_mem_source": pm.source,
        "status": status,
        "train_n_clips": len(train_records) if train_records is not None else None,
        "eval_n_clips": len(eval_records) if eval_records is not None else None,
        "final_train_loss": final_train,
        "final_eval_loss": final_eval,
        "scores": scores,
        "args": {k: _to_jsonable(v) for k, v in sorted(vars(args).items())
                 if hasattr(args, "__dict__") and not k.startswith("_")},
    }
    if extra:
        for k, v in extra.items():
            card[k] = _to_jsonable(v)

    path = output_dir / "run_card.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    return path
