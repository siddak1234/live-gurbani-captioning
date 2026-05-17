#!/usr/bin/env python3
"""Seed editable OOS GT working files from machine-generated drafts.

This is a convenience step, not a validation bypass. The output files are
written under ``eval_data/oos_v1/test/`` because that is where the evaluator
will eventually read from, but they are explicitly marked
``NEEDS_HUMAN_CORRECTION``. ``make validate-oos-gt`` still fails until a human
reviews each file and sets ``curation_status`` to ``HUMAN_CORRECTED_V1``.
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

from scripts.bootstrap_oos_gt import OosCase, load_cases  # noqa: E402


DEFAULT_CASES = REPO_ROOT / "eval_data" / "oos_v1" / "cases.yaml"
DEFAULT_DRAFT_DIR = REPO_ROOT / "eval_data" / "oos_v1" / "drafts"
DEFAULT_TEST_DIR = REPO_ROOT / "eval_data" / "oos_v1" / "test"
WORKING_STATUS = "NEEDS_HUMAN_CORRECTION"
HUMAN_STATUS = "HUMAN_CORRECTED_V1"


@dataclass(frozen=True)
class PrepareResult:
    case_id: str
    status: str
    path: pathlib.Path
    message: str = ""


def _rel(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{_rel(path)} is invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{_rel(path)} must contain a JSON object")
    return data


def _working_payload(case: OosCase, draft_path: pathlib.Path) -> dict[str, Any]:
    payload = _load_json(draft_path)

    if payload.get("video_id") != case.case_id:
        raise ValueError(
            f"{_rel(draft_path)} video_id must be {case.case_id!r}, "
            f"got {payload.get('video_id')!r}"
        )
    if int(payload.get("shabad_id", -1)) != case.shabad_id:
        raise ValueError(
            f"{_rel(draft_path)} shabad_id must be {case.shabad_id}, "
            f"got {payload.get('shabad_id')!r}"
        )

    prior_status = str(payload.get("curation_status", ""))
    payload["curation_status"] = WORKING_STATUS
    payload["draft_source"] = _rel(draft_path)
    payload["draft_curation_status"] = prior_status
    payload["human_review_required"] = True
    payload["review_protocol"] = (
        "Listen to the clipped audio, correct every segment boundary and "
        "canonical line field, then set curation_status to "
        f"{HUMAN_STATUS!r} only after review."
    )
    return payload


def prepare_case(
    case: OosCase,
    *,
    draft_dir: pathlib.Path,
    test_dir: pathlib.Path,
    force: bool = False,
) -> PrepareResult:
    draft_path = draft_dir / f"{case.case_id}.json"
    test_path = test_dir / f"{case.case_id}.json"
    if not draft_path.exists():
        return PrepareResult(case.case_id, "missing", draft_path, "draft file is missing")
    if test_path.exists() and not force:
        return PrepareResult(case.case_id, "skipped", test_path, "already exists; use --force")

    payload = _working_payload(case, draft_path)
    test_dir.mkdir(parents=True, exist_ok=True)
    test_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return PrepareResult(case.case_id, "written", test_path)


def prepare_all(
    cases: list[OosCase],
    *,
    draft_dir: pathlib.Path,
    test_dir: pathlib.Path,
    force: bool = False,
) -> list[PrepareResult]:
    return [
        prepare_case(case, draft_dir=draft_dir, test_dir=test_dir, force=force)
        for case in cases
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=pathlib.Path, default=DEFAULT_CASES)
    parser.add_argument("--draft-dir", type=pathlib.Path, default=DEFAULT_DRAFT_DIR)
    parser.add_argument("--test-dir", type=pathlib.Path, default=DEFAULT_TEST_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing working files under test/.",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases.resolve())
    results = prepare_all(
        cases,
        draft_dir=args.draft_dir.resolve(),
        test_dir=args.test_dir.resolve(),
        force=args.force,
    )

    for result in results:
        suffix = f" ({result.message})" if result.message else ""
        print(f"{result.status:>7}: {result.case_id} -> {_rel(result.path)}{suffix}")

    missing = [r for r in results if r.status == "missing"]
    if missing:
        print("\nerror: missing draft files; run make bootstrap-oos-gt first", file=sys.stderr)
        return 1

    written = sum(1 for r in results if r.status == "written")
    skipped = sum(1 for r in results if r.status == "skipped")
    print(
        f"\nPrepared {written} working file(s), skipped {skipped}. "
        f"Next: edit eval_data/oos_v1/test/*.json and set "
        f"curation_status={HUMAN_STATUS!r} only after review."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
