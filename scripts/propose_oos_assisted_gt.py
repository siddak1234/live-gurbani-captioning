#!/usr/bin/env python3
"""Create a machine-assisted OOS GT proposal set.

This script writes diagnostic labels, not gold labels. It starts from the
editable OOS working files under ``eval_data/oos_v1/test/`` and applies
conservative machine-silver edits using the same evidence as
``audit_oos_assisted_gt.py``:

- cached BaniDB corpus text;
- local Whisper ASR chunks;
- optional YouTube auto-caption JSON3 files.

Output goes to ``eval_data/oos_v1/assisted_test/`` and is marked
``MACHINE_ASSISTED_V1_NOT_GOLD``. It is meant for diagnostic scoring only.
Never use it for promotion.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rapidfuzz import fuzz  # noqa: E402

from scripts.audit_oos_assisted_gt import (  # noqa: E402
    CaptionChunk,
    AsrChunk,
    _overlap,
    load_asr_chunks,
    load_caption_chunks,
    load_corpus,
)
from scripts.bootstrap_oos_gt import OosCase, load_cases  # noqa: E402
from src.matcher import normalize  # noqa: E402


DEFAULT_CASES = REPO_ROOT / "eval_data" / "oos_v1" / "cases.yaml"
DEFAULT_INPUT_DIR = REPO_ROOT / "eval_data" / "oos_v1" / "test"
DEFAULT_OUT_DIR = REPO_ROOT / "eval_data" / "oos_v1" / "assisted_test"
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus_cache"
DEFAULT_ASR_CACHE = REPO_ROOT / "asr_cache"
DEFAULT_CAPTIONS_DIR = pathlib.Path("/private/tmp/oos_subs")
ASSISTED_STATUS = "MACHINE_ASSISTED_V1_NOT_GOLD"


@dataclass(frozen=True)
class EvidenceMatch:
    line_idx: int
    score: float
    text: str
    source: str


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _score(a: str, b: str) -> float:
    return float(fuzz.WRatio(normalize(a), normalize(b)))


def _line_by_idx(corpus: list[dict]) -> dict[int, dict]:
    return {int(row.get("line_idx", -1)): row for row in corpus}


def _best_line(text: str, corpus: list[dict], *, source: str) -> EvidenceMatch | None:
    if not text.strip():
        return None
    best: EvidenceMatch | None = None
    for row in corpus:
        idx = int(row.get("line_idx", -1))
        if idx == 0:
            continue
        cand = str(row.get("banidb_gurmukhi") or row.get("transliteration_english") or "")
        score = _score(text, cand)
        if best is None or score > best.score:
            best = EvidenceMatch(idx, score, cand, source)
    return best


def _text_overlapping(start: float, end: float, chunks: list[AsrChunk | CaptionChunk]) -> str:
    texts: list[str] = []
    for chunk in chunks:
        if _overlap(start, end, chunk.start, chunk.end) > 0.25:
            texts.append(chunk.text)
    return " ".join(texts)


def _evidence_for_window(
    *,
    start: float,
    end: float,
    corpus: list[dict],
    asr_chunks: list[AsrChunk],
    captions: list[CaptionChunk],
) -> EvidenceMatch | None:
    candidates: list[EvidenceMatch] = []
    asr_text = _text_overlapping(start, end, asr_chunks)
    asr_match = _best_line(asr_text, corpus, source="asr") if asr_text else None
    if asr_match:
        candidates.append(asr_match)
    cap_text = _text_overlapping(start, end, captions)
    cap_match = _best_line(cap_text, corpus, source="caption") if cap_text else None
    if cap_match:
        candidates.append(cap_match)
    if not candidates:
        return None
    return max(candidates, key=lambda m: m.score)


def _segment_payload(start: float, end: float, corpus_row: dict, *, shabad_id: int) -> dict[str, Any]:
    return {
        "start": round(float(start), 3),
        "end": round(float(end), 3),
        "line_idx": int(corpus_row["line_idx"]),
        "shabad_id": shabad_id,
        "verse_id": corpus_row.get("verse_id"),
        "banidb_gurmukhi": corpus_row.get("banidb_gurmukhi"),
    }


def _copy_line_fields(segment: dict[str, Any], corpus_row: dict) -> dict[str, Any]:
    out = dict(segment)
    out["line_idx"] = int(corpus_row["line_idx"])
    if corpus_row.get("shabad_id"):
        out["shabad_id"] = corpus_row.get("shabad_id")
    out["verse_id"] = corpus_row.get("verse_id")
    out["banidb_gurmukhi"] = corpus_row.get("banidb_gurmukhi")
    return out


def propose_case(
    case: OosCase,
    *,
    input_dir: pathlib.Path,
    corpus_dir: pathlib.Path,
    asr_cache: pathlib.Path,
    captions_dir: pathlib.Path,
    min_gap_s: float = 10.0,
    gap_score_threshold: float = 58.0,
    replace_score_threshold: float = 60.0,
    replace_margin: float = 8.0,
) -> tuple[dict[str, Any], list[str]]:
    src = input_dir / f"{case.case_id}.json"
    payload = _load_json(src)
    corpus = load_corpus(corpus_dir, case.shabad_id)
    corpus_by_idx = _line_by_idx(corpus)
    asr_chunks = load_asr_chunks(asr_cache, case.case_id)
    captions = load_caption_chunks(captions_dir, case)
    total_duration = float(payload.get("total_duration", case.expected_duration_s))

    notes: list[str] = []
    proposed_segments: list[dict[str, Any]] = []
    segments = sorted(list(payload.get("segments") or []), key=lambda s: float(s.get("start", 0.0)))

    prev_end = 0.0
    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))

        if start - prev_end >= min_gap_s:
            match = _evidence_for_window(
                start=prev_end,
                end=start,
                corpus=corpus,
                asr_chunks=asr_chunks,
                captions=captions,
            )
            if match and match.score >= gap_score_threshold and match.line_idx in corpus_by_idx:
                proposed = _segment_payload(
                    prev_end,
                    start,
                    corpus_by_idx[match.line_idx],
                    shabad_id=case.shabad_id,
                )
                proposed["machine_assisted"] = {
                    "action": "insert_from_gap_evidence",
                    "source": match.source,
                    "score": round(match.score, 2),
                    "reason": "ASR/caption evidence inside unlabeled gap matched corpus line",
                }
                proposed_segments.append(proposed)
                notes.append(
                    f"inserted line {match.line_idx} for gap {prev_end:.1f}-{start:.1f}s "
                    f"from {match.source} score {match.score:.1f}"
                )

        current = dict(segment)
        target = str(current.get("banidb_gurmukhi", ""))
        current_idx = int(current.get("line_idx", -1))
        current_score = -1.0
        best = _evidence_for_window(
            start=start,
            end=end,
            corpus=corpus,
            asr_chunks=asr_chunks,
            captions=captions,
        )
        if best:
            current_score = _score(_text_overlapping(start, end, asr_chunks + captions), target)
        if (
            best
            and best.line_idx != current_idx
            and best.line_idx in corpus_by_idx
            and best.score >= replace_score_threshold
            and best.score >= current_score + replace_margin
        ):
            current = _copy_line_fields(current, corpus_by_idx[best.line_idx])
            current["machine_assisted"] = {
                "action": "replace_line_from_evidence",
                "source": best.source,
                "score": round(best.score, 2),
                "previous_line_idx": current_idx,
                "reason": "ASR/caption evidence matched a different corpus line more strongly",
            }
            notes.append(
                f"replaced {start:.1f}-{end:.1f}s line {current_idx} -> {best.line_idx} "
                f"from {best.source} score {best.score:.1f}"
            )
        proposed_segments.append(current)
        prev_end = max(prev_end, end)

    if total_duration - prev_end >= min_gap_s:
        match = _evidence_for_window(
            start=prev_end,
            end=total_duration,
            corpus=corpus,
            asr_chunks=asr_chunks,
            captions=captions,
        )
        if match and match.score >= gap_score_threshold and match.line_idx in corpus_by_idx:
            proposed = _segment_payload(
                prev_end,
                total_duration,
                corpus_by_idx[match.line_idx],
                shabad_id=case.shabad_id,
            )
            proposed["machine_assisted"] = {
                "action": "insert_from_gap_evidence",
                "source": match.source,
                "score": round(match.score, 2),
                "reason": "ASR/caption evidence inside trailing unlabeled gap matched corpus line",
            }
            proposed_segments.append(proposed)
            notes.append(
                f"inserted line {match.line_idx} for trailing gap {prev_end:.1f}-{total_duration:.1f}s "
                f"from {match.source} score {match.score:.1f}"
            )

    out = dict(payload)
    out["curation_status"] = ASSISTED_STATUS
    out["gold_review_required"] = True
    out["machine_assisted_proposal"] = {
        "source": "scripts/propose_oos_assisted_gt.py",
        "input_dir": str(input_dir),
        "signals": ["cached BaniDB corpus", "local Whisper ASR cache", "optional YouTube captions"],
        "notes": notes,
    }
    out["segments"] = sorted(proposed_segments, key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))
    return out, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=pathlib.Path, default=DEFAULT_CASES)
    parser.add_argument("--input-dir", type=pathlib.Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--corpus-dir", type=pathlib.Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--asr-cache", type=pathlib.Path, default=DEFAULT_ASR_CACHE)
    parser.add_argument("--captions-dir", type=pathlib.Path, default=DEFAULT_CAPTIONS_DIR)
    args = parser.parse_args()

    cases = load_cases(args.cases.resolve())
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing machine-assisted OOS proposal files to {out_dir}")

    for case in cases:
        payload, notes = propose_case(
            case,
            input_dir=args.input_dir.resolve(),
            corpus_dir=args.corpus_dir.resolve(),
            asr_cache=args.asr_cache.resolve(),
            captions_dir=args.captions_dir.resolve(),
        )
        out_path = out_dir / f"{case.case_id}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"  {case.case_id}: {len(notes)} machine edit(s) -> {out_path}")
        for note in notes:
            print(f"    - {note}")
    print("\nDiagnostic only: these files are not gold GT and must not be promoted without review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
