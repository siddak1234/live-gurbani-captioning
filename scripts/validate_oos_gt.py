#!/usr/bin/env python3
"""Validate hand-corrected OOS GT JSONs before scoring.

The OOS v1 pack is the promotion gate for Phase 2.9. Machine-generated drafts
live under ``eval_data/oos_v1/drafts/``; only hand-corrected labels belong in
``eval_data/oos_v1/test/``. This script prevents accidental scoring of drafts
or half-labeled files.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import wave
from dataclasses import dataclass
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.bootstrap_oos_gt import OosCase, load_cases  # noqa: E402

DEFAULT_CASES = REPO_ROOT / "eval_data" / "oos_v1" / "cases.yaml"
DEFAULT_AUDIO_DIR = REPO_ROOT / "eval_data" / "oos_v1" / "audio"
DEFAULT_GT_DIR = REPO_ROOT / "eval_data" / "oos_v1" / "test"
HUMAN_STATUS = "HUMAN_CORRECTED_V1"
DRAFT_MARKER = "DRAFT_FROM_ORACLE_ENGINE"


@dataclass
class ValidationResult:
    case_id: str
    errors: list[str]


def wav_duration_s(path: pathlib.Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return float(wav.getnframes()) / float(wav.getframerate())


def _as_number(value: Any, label: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label} must be numeric")
        return None
    return float(value)


def validate_case(
    case: OosCase,
    *,
    gt_dir: pathlib.Path,
    audio_dir: pathlib.Path,
    require_human_status: bool = True,
) -> ValidationResult:
    errors: list[str] = []
    gt_path = gt_dir / f"{case.case_id}.json"
    audio_path = audio_dir / f"{case.case_id}_16k.wav"

    if not gt_path.exists():
        return ValidationResult(case.case_id, [f"missing GT file: {gt_path}"])
    if not audio_path.exists():
        errors.append(f"missing audio file: {audio_path}")

    try:
        gt = json.loads(gt_path.read_text())
    except json.JSONDecodeError as e:
        return ValidationResult(case.case_id, [f"invalid JSON: {e}"])

    if gt.get("video_id") != case.case_id:
        errors.append(f"video_id must be {case.case_id!r}, got {gt.get('video_id')!r}")
    if int(gt.get("shabad_id", -1)) != case.shabad_id:
        errors.append(f"shabad_id must be {case.shabad_id}, got {gt.get('shabad_id')!r}")

    status = str(gt.get("curation_status", ""))
    if DRAFT_MARKER in status or "DRAFT" in status.upper():
        errors.append("curation_status still marks this file as a draft")
    if require_human_status and status != HUMAN_STATUS:
        errors.append(f"curation_status must be {HUMAN_STATUS!r} after hand correction")

    total_duration = _as_number(gt.get("total_duration"), "total_duration", errors)
    uem = gt.get("uem")
    if not isinstance(uem, dict):
        errors.append("uem must be an object with start/end")
        uem_start = uem_end = None
    else:
        uem_start = _as_number(uem.get("start"), "uem.start", errors)
        uem_end = _as_number(uem.get("end"), "uem.end", errors)
        if uem_start is not None and uem_end is not None and uem_end <= uem_start:
            errors.append("uem.end must be greater than uem.start")

    if audio_path.exists() and total_duration is not None:
        audio_duration = wav_duration_s(audio_path)
        if abs(audio_duration - total_duration) > 1.0:
            errors.append(
                f"total_duration {total_duration:.3f}s differs from audio duration "
                f"{audio_duration:.3f}s by >1s"
            )
    if total_duration is not None and uem_end is not None and uem_end > total_duration + 0.5:
        errors.append("uem.end exceeds total_duration")

    segments = gt.get("segments")
    if not isinstance(segments, list) or not segments:
        errors.append("segments must be a non-empty list")
        return ValidationResult(case.case_id, errors)

    prev_start = -1.0
    for i, seg in enumerate(segments):
        prefix = f"segments[{i}]"
        if not isinstance(seg, dict):
            errors.append(f"{prefix} must be an object")
            continue
        start = _as_number(seg.get("start"), f"{prefix}.start", errors)
        end = _as_number(seg.get("end"), f"{prefix}.end", errors)
        if start is not None and end is not None:
            if end <= start:
                errors.append(f"{prefix}.end must be greater than start")
            if start < -0.25:
                errors.append(f"{prefix}.start is negative")
            if total_duration is not None and end > total_duration + 0.5:
                errors.append(f"{prefix}.end exceeds total_duration")
            if start < prev_start - 0.001:
                errors.append(f"{prefix}.start is out of order")
            prev_start = start
        line_idx = seg.get("line_idx")
        if isinstance(line_idx, bool) or not isinstance(line_idx, int) or line_idx < 0:
            errors.append(f"{prefix}.line_idx must be a non-negative integer")
        if seg.get("verse_id") in (None, ""):
            errors.append(f"{prefix}.verse_id is required")
        text = seg.get("banidb_gurmukhi")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{prefix}.banidb_gurmukhi is required")

    return ValidationResult(case.case_id, errors)


def validate_all(
    *,
    cases_path: pathlib.Path,
    gt_dir: pathlib.Path,
    audio_dir: pathlib.Path,
    require_human_status: bool = True,
) -> list[ValidationResult]:
    cases = load_cases(cases_path)
    results = [
        validate_case(
            case,
            gt_dir=gt_dir,
            audio_dir=audio_dir,
            require_human_status=require_human_status,
        )
        for case in cases
    ]

    expected = {case.case_id for case in cases}
    extra = sorted(p.stem for p in gt_dir.glob("*.json") if p.stem not in expected)
    for stem in extra:
        results.append(ValidationResult(stem, [f"unexpected GT file: {gt_dir / (stem + '.json')}"]))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=pathlib.Path, default=DEFAULT_CASES)
    parser.add_argument("--gt-dir", type=pathlib.Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--audio-dir", type=pathlib.Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument(
        "--allow-draft-status",
        action="store_true",
        help="Testing escape hatch; production scoring should never pass this.",
    )
    args = parser.parse_args()

    results = validate_all(
        cases_path=args.cases.resolve(),
        gt_dir=args.gt_dir.resolve(),
        audio_dir=args.audio_dir.resolve(),
        require_human_status=not args.allow_draft_status,
    )
    failures = [r for r in results if r.errors]
    if failures:
        print("OOS GT validation FAILED")
        for result in failures:
            print(f"\n{result.case_id}:")
            for error in result.errors:
                print(f"  - {error}")
        print(
            "\nDrafts must be hand-corrected into eval_data/oos_v1/test/ and marked "
            f"curation_status={HUMAN_STATUS!r} before scoring."
        )
        return 1

    print(f"OOS GT validation passed: {len(results)} case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
