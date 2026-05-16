"""Tests for content-based benchmark holdout (Phase 1.E).

The kirtan dataset's ``canonical_shabad_id`` namespace (3-char tokens
like ``"JS2"``) doesn't align with BaniDB's integer IDs the benchmark
uses (``{4377, 1821, 1341, 3712}``). So the only reliable contamination
check at the row level is text-based: does the line match any canonical
Gurmukhi line published in the benchmark's GT JSONs?

These tests pin that contract for ``load_benchmark_lines()``:

  - Loads canonical line strings from benchmark GT JSONs
  - Normalizes via ``src.matcher.normalize`` so the filter is robust to
    capitalization / whitespace / punctuation drift
  - Returns an empty set when benchmark dir is unavailable (graceful
    no-op for the training-machine workflow)
  - Skips malformed JSON without crashing
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import yaml  # noqa: F401
    from src.matcher import normalize  # noqa: F401
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

if _HAS_DEPS:
    from scripts.pull_dataset import load_benchmark_lines  # noqa: E402
    from src.matcher import normalize  # noqa: E402,F811


def _write_gt(dir_: Path, case_id: str, shabad_id: int, lines: list[str]) -> None:
    """Write a benchmark-shaped GT JSON to a synthetic test dir."""
    gt = {
        "video_id": case_id,
        "shabad_id": shabad_id,
        "segments": [
            {"start": float(i * 5), "end": float((i + 1) * 5),
             "line_idx": i, "banidb_gurmukhi": text}
            for i, text in enumerate(lines)
        ],
    }
    (dir_ / f"{case_id}.json").write_text(json.dumps(gt, ensure_ascii=False))


@unittest.skipUnless(_HAS_DEPS, "PyYAML and src.matcher (unidecode+rapidfuzz) required")
class TestLoadBenchmarkLines(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.bench = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_loads_canonical_gurmukhi_lines(self):
        _write_gt(self.bench, "case_001", 4377, [
            "ਪੋਥੀ ਪਰਮੇਸਰ ਕਾ ਥਾਨੁ",
            "ਸਾਧਸੰਗਿ ਗਾਵਹਿ ਗੁਣ ਗੋਬਿੰਦ",
        ])
        lines = load_benchmark_lines(self.bench)
        self.assertEqual(len(lines), 2)
        # All entries are normalized (via src.matcher.normalize)
        self.assertIn(normalize("ਪੋਥੀ ਪਰਮੇਸਰ ਕਾ ਥਾਨੁ"), lines)
        self.assertIn(normalize("ਸਾਧਸੰਗਿ ਗਾਵਹਿ ਗੁਣ ਗੋਬਿੰਦ"), lines)

    def test_deduplicates_across_files(self):
        """Same line in two GT files (e.g. same shabad across cold variants)
        should land as one entry in the set."""
        _write_gt(self.bench, "case_001", 4377, ["ਪੋਥੀ ਪਰਮੇਸਰ ਕਾ ਥਾਨੁ"])
        _write_gt(self.bench, "case_002", 4377, ["ਪੋਥੀ ਪਰਮੇਸਰ ਕਾ ਥਾਨੁ"])
        lines = load_benchmark_lines(self.bench)
        self.assertEqual(len(lines), 1)

    def test_unavailable_benchmark_dir_returns_empty(self):
        """Per CLAUDE.md, training machines don't require the benchmark repo.
        Missing dir -> empty set + stderr warning, not a crash."""
        nonexistent = Path("/nonexistent/path/should/not/exist")
        lines = load_benchmark_lines(nonexistent)
        self.assertEqual(lines, set())

    def test_default_path_resolution(self):
        """Calling with no argument should attempt the canonical sibling
        path (../live-gurbani-captioning-benchmark-v1/test). If that
        location exists on this machine, lines are loaded; otherwise empty.
        Either way, no crash."""
        lines = load_benchmark_lines()
        self.assertIsInstance(lines, set)

    def test_skips_malformed_json(self):
        """A malformed GT file should be skipped, not crash the loader."""
        _write_gt(self.bench, "case_good", 4377, ["ਪੋਥੀ ਪਰਮੇਸਰ ਕਾ ਥਾਨੁ"])
        (self.bench / "case_bad.json").write_text("not valid json {{{")
        lines = load_benchmark_lines(self.bench)
        # Good file's line still loaded
        self.assertEqual(len(lines), 1)

    def test_segments_without_banidb_gurmukhi_fall_back_to_text(self):
        """The benchmark schema's primary field is banidb_gurmukhi but the
        loader also accepts text as a fallback so we don't lose signal on
        older GT files."""
        gt = {
            "video_id": "case_x",
            "shabad_id": 1821,
            "segments": [
                {"start": 0.0, "end": 5.0, "line_idx": 0,
                 "text": "ਫਾਲਬੈਕ ਟੈਕਸਟ"},
            ],
        }
        (self.bench / "case_x.json").write_text(json.dumps(gt, ensure_ascii=False))
        lines = load_benchmark_lines(self.bench)
        self.assertIn(normalize("ਫਾਲਬੈਕ ਟੈਕਸਟ"), lines)

    def test_empty_lines_filtered(self):
        """Empty/whitespace-only line strings should not enter the set."""
        gt = {
            "video_id": "case_e",
            "shabad_id": 4377,
            "segments": [
                {"start": 0, "end": 5, "line_idx": 0, "banidb_gurmukhi": ""},
                {"start": 5, "end": 10, "line_idx": 1, "banidb_gurmukhi": "   "},
                {"start": 10, "end": 15, "line_idx": 2, "banidb_gurmukhi": "ਠੀਕ ਲਾਈਨ"},
            ],
        }
        (self.bench / "case_e.json").write_text(json.dumps(gt, ensure_ascii=False))
        lines = load_benchmark_lines(self.bench)
        self.assertEqual(len(lines), 1)
        self.assertIn(normalize("ਠੀਕ ਲਾਈਨ"), lines)


@unittest.skipUnless(_HAS_DEPS, "deps required")
class TestContentHoldoutSemantics(unittest.TestCase):
    """End-to-end check on the live benchmark repo (when present).

    Runs against the actual paired benchmark at the canonical sibling path.
    Skipped if the repo isn't available (training-machine workflow)."""

    def test_real_benchmark_returns_nonempty_lines(self):
        canonical = _REPO_ROOT.parent / "live-gurbani-captioning-benchmark-v1" / "test"
        if not canonical.exists():
            self.skipTest("paired benchmark repo not at canonical sibling path")
        lines = load_benchmark_lines(canonical)
        # Benchmark has 4 shabads with multiple pangtis each; expect at least 10 lines
        self.assertGreater(len(lines), 10,
                           f"only {len(lines)} canonical lines loaded — benchmark JSON shape may have drifted")
        # All entries are normalized (no raw Gurmukhi)
        for line in list(lines)[:3]:
            self.assertEqual(line, normalize(line))


if __name__ == "__main__":
    unittest.main()
