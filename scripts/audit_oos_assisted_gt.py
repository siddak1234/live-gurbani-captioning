#!/usr/bin/env python3
"""Machine-assisted cross-check for OOS working GT files.

This script does not certify ground truth. It triangulates the editable
``eval_data/oos_v1/test/*.json`` files against:

1. the cached BaniDB corpus for each shabad;
2. local Whisper ASR chunks in ``asr_cache``;
3. optional YouTube auto-caption JSON3 files when available.

The output is a reviewer aid: concrete mismatches, low-confidence segments,
large gaps, and online-caption hints. The final gate remains
``make validate-oos-gt`` with ``curation_status=HUMAN_CORRECTED_V1``.
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

from scripts.bootstrap_oos_gt import OosCase, load_cases  # noqa: E402
from src.matcher import normalize  # noqa: E402


DEFAULT_CASES = REPO_ROOT / "eval_data" / "oos_v1" / "cases.yaml"
DEFAULT_GT_DIR = REPO_ROOT / "eval_data" / "oos_v1" / "test"
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus_cache"
DEFAULT_ASR_CACHE = REPO_ROOT / "asr_cache"
DEFAULT_CAPTIONS_DIR = pathlib.Path("/private/tmp/oos_subs")
DEFAULT_OUT = REPO_ROOT / "diagnostics" / "oos_v1_assisted_crosscheck.md"


@dataclass(frozen=True)
class AsrChunk:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class CaptionChunk:
    start: float
    end: float
    text: str
    source: str


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _score(a: str, b: str) -> float:
    return float(fuzz.WRatio(normalize(a), normalize(b)))


def load_corpus(path: pathlib.Path, shabad_id: int) -> list[dict]:
    corpus_path = path / f"{shabad_id}.json"
    data = _load_json(corpus_path)
    return list(data.get("lines") or [])


def load_asr_chunks(cache_dir: pathlib.Path, case_id: str) -> list[AsrChunk]:
    candidates = sorted(cache_dir.glob(f"{case_id}_16k__medium__pa.json"))
    if not candidates:
        candidates = sorted(cache_dir.glob(f"{case_id}_16k__*.json"))
    if not candidates:
        return []
    rows = _load_json(candidates[0])
    return [
        AsrChunk(float(r.get("start", 0.0)), float(r.get("end", 0.0)), str(r.get("text", "")))
        for r in rows
    ]


def _caption_text(event: dict) -> str:
    return "".join(str(seg.get("utf8", "")) for seg in event.get("segs") or []).strip()


def load_caption_chunks(captions_dir: pathlib.Path, case: OosCase) -> list[CaptionChunk]:
    if not captions_dir.exists():
        return []
    paths = sorted(captions_dir.glob(f"{case.source_video_id}.*.json3"))
    chunks: list[CaptionChunk] = []
    offset = float(case.clip_start_s)
    for path in paths:
        try:
            data = _load_json(path)
        except json.JSONDecodeError:
            continue
        for event in data.get("events", []):
            text = _caption_text(event)
            if not text:
                continue
            start = float(event.get("tStartMs", 0)) / 1000.0 - offset
            end = start + float(event.get("dDurationMs", 0)) / 1000.0
            if end < 0 or start > case.expected_duration_s:
                continue
            chunks.append(
                CaptionChunk(
                    max(0.0, start),
                    min(case.expected_duration_s, end),
                    text,
                    path.name,
                )
            )
    # JSON3 can contain translated + original tracks that duplicate each other.
    seen: set[tuple[float, float, str]] = set()
    deduped: list[CaptionChunk] = []
    for chunk in sorted(chunks, key=lambda c: (c.start, c.end, c.text)):
        key = (round(chunk.start, 2), round(chunk.end, 2), chunk.text)
        if key not in seen:
            seen.add(key)
            deduped.append(chunk)
    return deduped


def _text_overlapping(start: float, end: float, chunks: list[AsrChunk | CaptionChunk]) -> str:
    texts: list[str] = []
    for chunk in chunks:
        if _overlap(start, end, chunk.start, chunk.end) > 0.25:
            texts.append(chunk.text)
    return " ".join(texts)


def _gap_finding(
    *,
    start: float,
    end: float,
    corpus: list[dict],
    asr_chunks: list[AsrChunk],
    captions: list[CaptionChunk],
) -> str:
    evidence: list[str] = []
    asr_text = _text_overlapping(start, end, asr_chunks)
    if asr_text:
        best_idx, best_score, best_text = _best_corpus_line(asr_text, corpus)
        if best_idx is not None and best_score >= 55.0:
            evidence.append(f"ASR hears likely line {best_idx} ({best_score:.1f}): `{best_text}`")
        else:
            evidence.append(f"ASR hears text: `{asr_text}`")
    cap_text = _text_overlapping(start, end, captions)
    if cap_text:
        best_idx, best_score, best_text = _best_corpus_line(cap_text, corpus)
        if best_idx is not None and best_score >= 55.0:
            evidence.append(f"captions suggest line {best_idx} ({best_score:.1f}): `{best_text}`")
        else:
            evidence.append(f"captions contain text: `{cap_text}`")
    return "; ".join(evidence)


def _best_corpus_line(text: str, corpus: list[dict]) -> tuple[int | None, float, str]:
    best_idx: int | None = None
    best_score = -1.0
    best_text = ""
    for row in corpus:
        idx = int(row.get("line_idx", -1))
        if idx == 0:
            continue
        cand = str(row.get("banidb_gurmukhi") or row.get("transliteration_english") or "")
        score = _score(text, cand)
        if score > best_score:
            best_idx, best_score, best_text = idx, score, cand
    return best_idx, best_score, best_text


def _segment_findings(
    *,
    segment: dict,
    corpus: list[dict],
    asr_chunks: list[AsrChunk],
    captions: list[CaptionChunk],
) -> list[str]:
    findings: list[str] = []
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", 0.0))
    target = str(segment.get("banidb_gurmukhi", ""))
    line_idx = int(segment.get("line_idx", -1))

    asr_text = _text_overlapping(start, end, asr_chunks)
    if asr_text:
        asr_score = _score(asr_text, target)
        best_idx, best_score, best_text = _best_corpus_line(asr_text, corpus)
        if asr_score < 55:
            findings.append(f"low ASR/text agreement ({asr_score:.1f}) for `{target}` vs ASR `{asr_text}`")
        if best_idx is not None and best_idx != line_idx and best_score >= max(60.0, asr_score + 8.0):
            findings.append(
                f"ASR better matches corpus line {best_idx} ({best_score:.1f}) than GT line {line_idx} "
                f"({asr_score:.1f}); best text `{best_text}`"
            )
    else:
        findings.append("no overlapping ASR chunk found")

    cap_text = _text_overlapping(start, end, captions)
    if cap_text:
        cap_score = _score(cap_text, target)
        best_idx, best_score, best_text = _best_corpus_line(cap_text, corpus)
        if cap_score < 55:
            findings.append(f"low caption/text agreement ({cap_score:.1f}) for `{target}` vs captions `{cap_text}`")
        if best_idx is not None and best_idx != line_idx and best_score >= max(60.0, cap_score + 8.0):
            findings.append(
                f"captions better match corpus line {best_idx} ({best_score:.1f}) than GT line {line_idx} "
                f"({cap_score:.1f}); best text `{best_text}`"
            )

    return findings


def audit_case(
    case: OosCase,
    *,
    gt_dir: pathlib.Path,
    corpus_dir: pathlib.Path,
    asr_cache: pathlib.Path,
    captions_dir: pathlib.Path,
) -> str:
    gt_path = gt_dir / f"{case.case_id}.json"
    gt = _load_json(gt_path)
    corpus = load_corpus(corpus_dir, case.shabad_id)
    asr_chunks = load_asr_chunks(asr_cache, case.case_id)
    captions = load_caption_chunks(captions_dir, case)
    segments = list(gt.get("segments") or [])

    lines: list[str] = []
    lines.append(f"## {case.case_id} — shabad {case.shabad_id}")
    lines.append("")
    lines.append(f"- Status: `{gt.get('curation_status', '')}`")
    lines.append(f"- Segments: {len(segments)}")
    lines.append(f"- ASR chunks: {len(asr_chunks)}")
    lines.append(f"- Online caption chunks: {len(captions)}")
    if captions:
        sources = sorted({c.source for c in captions})
        lines.append(f"- Caption sources: {', '.join(f'`{s}`' for s in sources)}")

    gap_rows: list[str] = []
    prev_end = 0.0
    for seg in segments:
        start = float(seg.get("start", 0.0))
        if start - prev_end > 10.0:
            finding = _gap_finding(
                start=prev_end,
                end=start,
                corpus=corpus,
                asr_chunks=asr_chunks,
                captions=captions,
            )
            gap_rows.append(f"| {prev_end:.1f}-{start:.1f} | {finding or 'no ASR/caption evidence'} |")
        prev_end = float(seg.get("end", start))
    if float(gt.get("total_duration", case.expected_duration_s)) - prev_end > 10.0:
        end = float(gt.get("total_duration", case.expected_duration_s))
        finding = _gap_finding(
            start=prev_end,
            end=end,
            corpus=corpus,
            asr_chunks=asr_chunks,
            captions=captions,
        )
        gap_rows.append(f"| {prev_end:.1f}-{end:.1f} | {finding or 'no ASR/caption evidence'} |")
    if gap_rows:
        lines.append("")
        lines.append("| Large unlabeled gap | Machine evidence inside gap |")
        lines.append("|---:|---|")
        lines.extend(gap_rows)

    segment_rows: list[str] = []
    for i, seg in enumerate(segments, start=1):
        findings = _segment_findings(
            segment=seg,
            corpus=corpus,
            asr_chunks=asr_chunks,
            captions=captions,
        )
        if findings:
            segment_rows.append(
                f"| {i} | {float(seg.get('start', 0.0)):.1f}-{float(seg.get('end', 0.0)):.1f} "
                f"| {seg.get('line_idx')} | {'<br>'.join(findings)} |"
            )

    if segment_rows:
        lines.append("")
        lines.append("| # | Time | GT line | Machine concern |")
        lines.append("|---:|---:|---:|---|")
        lines.extend(segment_rows)
    else:
        lines.append("- Machine concerns: none above thresholds.")

    if captions:
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Online caption snippets inside clip</summary>")
        lines.append("")
        lines.append("| Time | Text |")
        lines.append("|---:|---|")
        for chunk in captions[:40]:
            lines.append(f"| {chunk.start:.2f}-{chunk.end:.2f} | {chunk.text} |")
        if len(captions) > 40:
            lines.append(f"| ... | {len(captions) - 40} more caption chunks omitted |")
        lines.append("")
        lines.append("</details>")

    return "\n".join(lines)


def render_report(
    cases: list[OosCase],
    *,
    gt_dir: pathlib.Path,
    corpus_dir: pathlib.Path,
    asr_cache: pathlib.Path,
    captions_dir: pathlib.Path,
) -> str:
    header = "\n".join([
        "# OOS v1 assisted GT cross-check",
        "",
        "This is a machine-assist report, not ground truth. It compares the editable",
        "`eval_data/oos_v1/test/*.json` files against cached shabad corpus text,",
        "local Whisper ASR chunks, and optional YouTube auto-caption JSON3 files.",
        "Use it to focus review; do not use it to bypass `HUMAN_CORRECTED_V1`.",
        "",
    ])
    case_sections = [
        audit_case(
            case,
            gt_dir=gt_dir,
            corpus_dir=corpus_dir,
            asr_cache=asr_cache,
            captions_dir=captions_dir,
        )
        for case in cases
    ]
    return "\n\n".join([header] + case_sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=pathlib.Path, default=DEFAULT_CASES)
    parser.add_argument("--gt-dir", type=pathlib.Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--corpus-dir", type=pathlib.Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--asr-cache", type=pathlib.Path, default=DEFAULT_ASR_CACHE)
    parser.add_argument("--captions-dir", type=pathlib.Path, default=DEFAULT_CAPTIONS_DIR)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    report = render_report(
        load_cases(args.cases.resolve()),
        gt_dir=args.gt_dir.resolve(),
        corpus_dir=args.corpus_dir.resolve(),
        asr_cache=args.asr_cache.resolve(),
        captions_dir=args.captions_dir.resolve(),
    )
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    try:
        shown = out.relative_to(REPO_ROOT)
    except ValueError:
        shown = out
    print(f"wrote: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
