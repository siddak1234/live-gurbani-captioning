#!/usr/bin/env python3
"""Bootstrap draft OOS GT JSONs from the OOS case manifest.

This script intentionally writes to ``eval_data/oos_v1/drafts/``, not
``eval_data/oos_v1/test/``. Drafts are oracle-engine predictions meant to save
manual labeling time. They are NOT ground truth and must be hand-corrected
before promotion into ``test/``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import wave
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.engine import EngineConfig, predict  # noqa: E402


DEFAULT_CASES = REPO_ROOT / "eval_data" / "oos_v1" / "cases.yaml"
DEFAULT_AUDIO_DIR = REPO_ROOT / "eval_data" / "oos_v1" / "audio"
DEFAULT_DRAFT_DIR = REPO_ROOT / "eval_data" / "oos_v1" / "drafts"
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus_cache"
DEFAULT_ASR_CACHE = REPO_ROOT / "asr_cache"
DEFAULT_ENGINE_CONFIG = REPO_ROOT / "configs" / "inference" / "v3_2.yaml"


@dataclass(frozen=True)
class OosCase:
    case_id: str
    shabad_id: int
    source_url: str
    source_video_id: str
    clip_start_s: float
    clip_end_s: float
    role: str = ""
    opening_line: str = ""
    source_title: str = ""
    rationale: str = ""

    @property
    def expected_duration_s(self) -> float:
        return float(self.clip_end_s) - float(self.clip_start_s)


def _parse_blend(spec) -> dict[str, float] | None:
    if not spec:
        return None
    if isinstance(spec, dict):
        return spec
    out: dict[str, float] = {}
    for part in str(spec).split(","):
        name, weight = part.split(":")
        out[name.strip()] = float(weight)
    return out


def load_engine_config(path: pathlib.Path) -> EngineConfig:
    import yaml

    raw = yaml.safe_load(path.read_text()) or {}
    raw["blend"] = _parse_blend(raw.get("blend"))
    raw["blind_blend"] = _parse_blend(raw.get("blind_blend"))
    if "asr_cache_dir" in raw and raw["asr_cache_dir"]:
        raw["asr_cache_dir"] = pathlib.Path(raw["asr_cache_dir"])
    known = set(EngineConfig().__dataclass_fields__.keys())
    unknown = set(raw) - known
    if unknown:
        print(f"warning: ignoring unknown engine-config keys: {sorted(unknown)}", file=sys.stderr)
    return EngineConfig(**{k: v for k, v in raw.items() if k in known})


def load_cases(path: pathlib.Path) -> list[OosCase]:
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    rows = data.get("cases") or []
    cases: list[OosCase] = []
    seen: set[str] = set()
    for i, row in enumerate(rows, start=1):
        try:
            case = OosCase(
                case_id=str(row["case_id"]),
                shabad_id=int(row["shabad_id"]),
                source_url=str(row["source_url"]),
                source_video_id=str(row["source_video_id"]),
                clip_start_s=float(row["clip_start_s"]),
                clip_end_s=float(row["clip_end_s"]),
                role=str(row.get("role", "")),
                opening_line=str(row.get("opening_line", "")),
                source_title=str(row.get("source_title", "")),
                rationale=str(row.get("rationale", "")),
            )
        except KeyError as e:
            raise ValueError(f"case row {i} missing required key: {e.args[0]}") from e
        if "/" in case.case_id or "\\" in case.case_id:
            raise ValueError(f"case_id must be a filename stem, got: {case.case_id!r}")
        if case.case_id in seen:
            raise ValueError(f"duplicate case_id: {case.case_id}")
        if case.clip_end_s <= case.clip_start_s:
            raise ValueError(f"{case.case_id}: clip_end_s must be greater than clip_start_s")
        seen.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"no cases found in {path}")
    return cases


def wav_duration_s(path: pathlib.Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return float(wav.getnframes()) / float(wav.getframerate())


def load_corpora(corpus_dir: pathlib.Path) -> dict[int, list[dict]]:
    corpora: dict[int, list[dict]] = {}
    for path in sorted(corpus_dir.glob("*.json")):
        data = json.loads(path.read_text())
        corpora[int(data["shabad_id"])] = data["lines"]
    return corpora


def draft_payload(case: OosCase, duration_s: float, segments: list[dict]) -> dict:
    return {
        "video_id": case.case_id,
        "shabad_id": case.shabad_id,
        "total_duration": round(duration_s, 3),
        "uem": {"start": 0.0, "end": round(duration_s, 3)},
        "source_url": case.source_url,
        "source_video_id": case.source_video_id,
        "source_clip": {
            "start_s": case.clip_start_s,
            "end_s": case.clip_end_s,
        },
        "curation_status": "DRAFT_FROM_ORACLE_ENGINE__HAND_CORRECT_BEFORE_COMMIT",
        "curation_notes": (
            "Machine-generated oracle bootstrap only. Correct boundaries and "
            "verify every verse_id/banidb_gurmukhi before copying to test/."
        ),
        "segments": segments,
    }


def bootstrap_one(
    case: OosCase,
    *,
    audio_dir: pathlib.Path,
    draft_dir: pathlib.Path,
    corpora: dict[int, list[dict]],
    config: EngineConfig,
) -> bool:
    audio_path = audio_dir / f"{case.case_id}_16k.wav"
    if not audio_path.exists():
        print(f"  error: missing audio {audio_path}", file=sys.stderr)
        return False
    if case.shabad_id not in corpora:
        print(f"  error: missing corpus_cache/{case.shabad_id}.json", file=sys.stderr)
        return False

    duration_s = wav_duration_s(audio_path)
    result = predict(audio_path, corpora, shabad_id=case.shabad_id, uem_start=0.0, config=config)
    segments = [
        {
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "line_idx": s.line_idx,
            "shabad_id": s.shabad_id,
            "verse_id": s.verse_id,
            "banidb_gurmukhi": s.banidb_gurmukhi,
        }
        for s in result.segments
    ]

    draft_dir.mkdir(parents=True, exist_ok=True)
    out_path = draft_dir / f"{case.case_id}.json"
    out_path.write_text(
        json.dumps(draft_payload(case, duration_s, segments), ensure_ascii=False, indent=2) + "\n"
    )
    print(f"  {case.case_id}: shabad={case.shabad_id} chunks={result.n_chunks} "
          f"segments={len(segments)} -> {out_path.relative_to(REPO_ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=pathlib.Path, default=DEFAULT_CASES)
    parser.add_argument("--audio-dir", type=pathlib.Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--draft-dir", type=pathlib.Path, default=DEFAULT_DRAFT_DIR)
    parser.add_argument("--corpus-dir", type=pathlib.Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--asr-cache-dir", type=pathlib.Path, default=DEFAULT_ASR_CACHE)
    parser.add_argument("--engine-config", type=pathlib.Path, default=DEFAULT_ENGINE_CONFIG)
    args = parser.parse_args()

    cases = load_cases(args.cases.resolve())
    corpora = load_corpora(args.corpus_dir.resolve())
    if not corpora:
        print(f"error: no corpus files in {args.corpus_dir}", file=sys.stderr)
        return 1

    config = load_engine_config(args.engine_config.resolve())
    if config.asr_cache_dir is None:
        config.asr_cache_dir = args.asr_cache_dir.resolve()

    print(f"Bootstrapping {len(cases)} OOS draft GT file(s) from {args.cases}")
    print("Mode: ORACLE shabad ID; output is draft-only, not ground truth.\n")

    failures: list[str] = []
    for case in cases:
        if not bootstrap_one(
            case,
            audio_dir=args.audio_dir.resolve(),
            draft_dir=args.draft_dir.resolve(),
            corpora=corpora,
            config=config,
        ):
            failures.append(case.case_id)

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1
    print(f"\nall {len(cases)} draft GT file(s) ready in {args.draft_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
