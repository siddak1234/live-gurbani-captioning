#!/usr/bin/env python3
"""Render a local HTML review pack for OOS GT curation.

The generated page is a human-labeling aid, not source data. It reads committed
case metadata and draft JSONs, links to local gitignored audio, and gives the
reviewer one place to play each draft segment while correcting the GT files that
will eventually live under ``eval_data/oos_v1/test/``.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.bootstrap_oos_gt import DEFAULT_CASES, OosCase, load_cases  # noqa: E402

DEFAULT_DRAFT_DIR = REPO_ROOT / "eval_data" / "oos_v1" / "drafts"
DEFAULT_AUDIO_DIR = REPO_ROOT / "eval_data" / "oos_v1" / "audio"
DEFAULT_TEST_DIR = REPO_ROOT / "eval_data" / "oos_v1" / "test"
DEFAULT_REVIEW_DIR = REPO_ROOT / "eval_data" / "oos_v1" / "review"


def display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def segment_rows(case_id: str, segments: list[dict]) -> str:
    rows: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td><button type=\"button\" onclick=\"playSegment('{html.escape(case_id)}',{start:.3f},{end:.3f})\">Play</button></td>"
            f"<td>{start:.3f}</td>"
            f"<td>{end:.3f}</td>"
            f"<td>{html.escape(str(seg.get('line_idx', '')))}</td>"
            f"<td>{html.escape(str(seg.get('verse_id', '')))}</td>"
            f"<td class=\"gurmukhi\">{html.escape(str(seg.get('banidb_gurmukhi', '')))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_case(
    case: OosCase,
    *,
    draft: dict,
    audio_rel: str,
    draft_path: pathlib.Path,
    test_path: pathlib.Path,
) -> str:
    status = str(draft.get("curation_status", ""))
    segments = draft.get("segments") or []
    total_duration = draft.get("total_duration", draft.get("uem", {}).get("end", ""))
    return f"""
<section class="case" id="{html.escape(case.case_id)}">
  <header>
    <h2>{html.escape(case.case_id)} · shabad {case.shabad_id}</h2>
    <p><strong>{html.escape(case.role)}</strong> — {html.escape(case.rationale)}</p>
    <p class="opening">{html.escape(case.opening_line)}</p>
    <p><a href="{html.escape(case.source_url)}">source recording</a> · clip {case.clip_start_s:.1f}s–{case.clip_end_s:.1f}s · draft segments: {len(segments)}</p>
  </header>
  <audio id="audio-{html.escape(case.case_id)}" controls preload="metadata" src="{html.escape(audio_rel)}"></audio>
  <div class="paths">
    <code>draft: {html.escape(str(draft_path))}</code>
    <code>save corrected GT: {html.escape(str(test_path))}</code>
  </div>
  <details open>
    <summary>Correction checklist</summary>
    <ul>
      <li>Listen through the full clipped audio once before editing boundaries.</li>
      <li>For each row, play the segment and adjust start/end to the sung line, not the model's guess.</li>
      <li>Delete extra rows, add missing repeated rows, and keep starts in ascending order.</li>
      <li>Verify every <code>verse_id</code> and <code>banidb_gurmukhi</code> against the shabad corpus.</li>
      <li>Set <code>curation_status</code> to <code>HUMAN_CORRECTED_V1</code>.</li>
      <li>Keep <code>total_duration</code> as <code>{html.escape(str(total_duration))}</code> unless the audio file changes.</li>
    </ul>
  </details>
  <p class="status">Current draft status: <code>{html.escape(status)}</code></p>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Audio</th><th>Start</th><th>End</th><th>Line</th><th>Verse</th><th>Draft Text</th>
      </tr>
    </thead>
    <tbody>
      {segment_rows(case.case_id, segments)}
    </tbody>
  </table>
</section>
"""


def render_index(
    cases: list[OosCase],
    *,
    draft_dir: pathlib.Path,
    audio_dir: pathlib.Path,
    test_dir: pathlib.Path,
    out_path: pathlib.Path,
) -> str:
    sections: list[str] = []
    for case in cases:
        draft_path = draft_dir / f"{case.case_id}.json"
        audio_path = audio_dir / f"{case.case_id}_16k.wav"
        test_path = test_dir / f"{case.case_id}.json"
        if not draft_path.exists():
            raise FileNotFoundError(f"missing draft: {draft_path}")
        draft = json.loads(draft_path.read_text())
        audio_rel = pathlib.Path(
            os.path.relpath(audio_path.resolve(), start=out_path.resolve().parent)
        ).as_posix()
        sections.append(
            render_case(
                case,
                draft=draft,
                audio_rel=audio_rel,
                draft_path=pathlib.Path(display_path(draft_path)),
                test_path=pathlib.Path(display_path(test_path)),
            )
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OOS v1 GT Review Pack</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.45; color: #17202a; }}
    h1 {{ margin-bottom: 4px; }}
    .case {{ border-top: 2px solid #d6dde5; padding-top: 24px; margin-top: 32px; }}
    .opening {{ font-size: 1.15rem; }}
    audio {{ width: min(760px, 100%); display: block; margin: 12px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #d6dde5; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f3f6f9; text-align: left; }}
    button {{ cursor: pointer; }}
    code {{ background: #f3f6f9; padding: 2px 4px; border-radius: 3px; }}
    .paths code {{ display: block; margin: 4px 0; }}
    .status {{ color: #7a3b00; }}
    .gurmukhi {{ font-size: 1.05rem; }}
  </style>
</head>
<body>
  <h1>OOS v1 GT Review Pack</h1>
  <p>This page is a labeling aid. It does not create ground truth. Run <code>make prepare-oos-review</code> to seed editable working copies under <code>eval_data/oos_v1/test/</code>, correct them against the audio, then run <code>make validate-oos-gt</code>.</p>
  <p>Current goal: prove whether <code>phase2_9_loop_align</code> generalizes beyond the four-shabad paired benchmark before promoting or scaling Phase 3.</p>
  {"".join(sections)}
  <script>
    function playSegment(caseId, start, end) {{
      const audio = document.getElementById('audio-' + caseId);
      audio.currentTime = start;
      audio.play();
      const stop = () => {{
        if (audio.currentTime >= end) {{
          audio.pause();
          audio.removeEventListener('timeupdate', stop);
        }}
      }};
      audio.addEventListener('timeupdate', stop);
    }}
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=pathlib.Path, default=DEFAULT_CASES)
    parser.add_argument("--draft-dir", type=pathlib.Path, default=DEFAULT_DRAFT_DIR)
    parser.add_argument("--audio-dir", type=pathlib.Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--test-dir", type=pathlib.Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_REVIEW_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.html"
    html_text = render_index(
        load_cases(args.cases.resolve()),
        draft_dir=args.draft_dir.resolve(),
        audio_dir=args.audio_dir.resolve(),
        test_dir=args.test_dir.resolve(),
        out_path=out_path,
    )
    out_path.write_text(html_text)
    print(f"review pack: {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
