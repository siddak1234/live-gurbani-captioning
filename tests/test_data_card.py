"""Tests for the data_card.md emitter in pull_dataset.py.

The data card is the per-pull lineage doc. These tests pin its
**structure** (sections present, table format, conditional sections),
not the precise wording. If a future PR adds a new section, add a
test; if a future PR re-words a section header, update both.

PyYAML is needed because the import path goes through
scripts/pull_dataset.py which imports yaml at module scope. Skipped
on bare-clone envs.

Run:
    python -m unittest tests.test_data_card -v
    make test
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import yaml  # noqa: F401
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

if _HAS_YAML:
    from scripts.pull_dataset import _build_data_card  # noqa: E402


def _args(**kw):
    """SimpleNamespace stand-in for argparse.Namespace with sensible defaults."""
    defaults = dict(
        num_samples=50, min_score=0.8,
        min_duration_s=1.0, max_duration_s=30.0,
        split_by="shabad", split_seed=42,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _manifest(shabad_counts: dict[str, int], dur: float = 10.0) -> list[dict]:
    out = []
    for sid, n in shabad_counts.items():
        for i in range(n):
            out.append({
                "audio": f"clips/{sid}_{i:03d}.wav",
                "text": f"line-{sid}-{i}",
                "source": "kirtan",
                "shabad_id": sid,
                "video_id": f"vid_{sid}",
                "score": 0.9,
                "duration_s": dur,
            })
    return out


@unittest.skipUnless(_HAS_YAML, "PyYAML not installed")
class TestBuildDataCardWithSplits(unittest.TestCase):
    """Data card when --split-by shabad produced 3 manifests."""

    def setUp(self):
        from scripts.pull_dataset import _split_by_shabad
        self.manifest = _manifest({f"s{i:03d}": 5 for i in range(20)})
        self.splits = _split_by_shabad(self.manifest)
        self.rejections = {
            "holdout_shabad": 3, "holdout_video": 1, "score_low": 12,
            "dur_short": 2, "dur_long": 0, "simran": 4,
        }
        self.card = _build_data_card(
            out_dir=Path("/tmp/run_v1"),
            source_key="kirtan",
            source_id="surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical",
            args=_args(),
            splits=self.splits,
            manifest=self.manifest,
            rejections=self.rejections,
            holdout_shabads={"4377", "1821", "1341", "3712", 4377, 1821, 1341, 3712},
            holdout_videos={"IZOsmkdmmcg", "kZhIA8P6xWI", "kchMJPK9Axs", "zOtIpxMT9hU"},
            apply_holdout=True,
        )

    def test_header_has_title_and_metadata(self):
        self.assertIn("# Data card — run_v1", self.card)
        self.assertIn("surindersinghssj/gurbani-kirtan", self.card)
        self.assertIn("kirtan", self.card)
        self.assertIn("num_samples=50", self.card)
        self.assertIn("split_by=shabad", self.card)
        self.assertIn("split_seed=42", self.card)

    def test_holdout_section_lists_benchmark_ids(self):
        """Header section must explicitly enumerate the held-out shabads
        and videos — half the point of the card is auditing what was NOT
        scraped."""
        for sid in ("4377", "1821", "1341", "3712"):
            self.assertIn(sid, self.card)
        for vid in ("IZOsmkdmmcg", "kZhIA8P6xWI", "kchMJPK9Axs", "zOtIpxMT9hU"):
            self.assertIn(vid, self.card)

    def test_split_summary_table_present(self):
        self.assertIn("## Split summary", self.card)
        self.assertIn("| Split | Clips | Hours | Unique shabads | Unique videos |", self.card)
        # All three split names appear in the table
        for split in ("train", "val", "test"):
            self.assertIn(f"| {split} |", self.card)

    def test_no_manifest_summary_when_splits_present(self):
        """The single-manifest section is the alternative path; when splits
        are present, only the split table should appear."""
        self.assertNotIn("## Manifest summary", self.card)

    def test_rejections_table_lists_all_reasons(self):
        self.assertIn("## Rejection counts", self.card)
        for reason in ("score_low", "dur_short", "dur_long",
                       "holdout_shabad", "holdout_video", "simran"):
            self.assertIn(f"| {reason} |", self.card)
        # Counts surface
        self.assertIn("| score_low | 12 |", self.card)
        self.assertIn("| holdout_shabad | 3 |", self.card)

    def test_top_shabads_section(self):
        """When splits exist, the top-shabads section uses the train pool."""
        self.assertIn("Top shabads in train (by hours)", self.card)
        # At least one shabad listed
        self.assertIn("shabad `s", self.card)

    def test_diversity_guardrail_section(self):
        self.assertIn("## Diversity guardrail", self.card)
        self.assertIn("val unique shabads:", self.card)
        self.assertIn("test unique shabads:", self.card)


@unittest.skipUnless(_HAS_YAML, "PyYAML not installed")
class TestBuildDataCardSingleManifest(unittest.TestCase):
    """Data card when --split-by none (legacy single manifest)."""

    def setUp(self):
        self.manifest = _manifest({"a": 5, "b": 5, "c": 5})
        self.card = _build_data_card(
            out_dir=Path("/tmp/run_v2"),
            source_key="kirtan",
            source_id="surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical",
            args=_args(split_by="none"),
            splits=None,  # <- the difference vs. split case
            manifest=self.manifest,
            rejections={"holdout_shabad": 0, "holdout_video": 0, "score_low": 0,
                        "dur_short": 0, "dur_long": 0, "simran": 0},
            holdout_shabads={"4377", "1821", "1341", "3712"},
            holdout_videos={"IZOsmkdmmcg", "kZhIA8P6xWI", "kchMJPK9Axs", "zOtIpxMT9hU"},
            apply_holdout=True,
        )

    def test_manifest_summary_present_when_no_splits(self):
        self.assertIn("## Manifest summary", self.card)
        self.assertNotIn("## Split summary", self.card)
        self.assertIn("Clips: 15", self.card)

    def test_top_shabads_uses_full_manifest(self):
        self.assertIn("Top shabads (by hours)", self.card)
        self.assertNotIn("Top shabads in train", self.card)

    def test_no_diversity_section_for_unsplit(self):
        """Diversity guardrail is meaningful only for val/test, which don't
        exist in the unsplit case."""
        self.assertNotIn("## Diversity guardrail", self.card)


@unittest.skipUnless(_HAS_YAML, "PyYAML not installed")
class TestDataCardHoldoutOff(unittest.TestCase):
    def test_states_holdout_not_enforced(self):
        """For general Punjabi sources (indicvoices, commonvoice), the
        card should clearly say holdout isn't applied — otherwise a
        reader might assume the same 4 shabads were filtered."""
        card = _build_data_card(
            out_dir=Path("/tmp/iv_v1"),
            source_key="indicvoices",
            source_id="ai4bharat/IndicVoices",
            args=_args(split_by="none"),
            splits=None,
            manifest=_manifest({"misc": 10}),
            rejections={k: 0 for k in ("holdout_shabad", "holdout_video", "score_low",
                                       "dur_short", "dur_long", "simran")},
            holdout_shabads=set(),
            holdout_videos=set(),
            apply_holdout=False,
        )
        self.assertIn("Not enforced", card)


@unittest.skipUnless(_HAS_YAML, "PyYAML not installed")
class TestDataCardDiversityWarning(unittest.TestCase):
    def test_warning_emoji_on_low_diversity(self):
        """If val or test ends up with <3 unique shabads, the card flags
        it with a warning glyph so reviewers spot it."""
        from scripts.pull_dataset import _split_by_shabad
        # 4 shabads → val and test will have 0-1 each
        manifest = _manifest({f"s{i}": 5 for i in range(4)})
        # We expect this to print a stderr warning during split; ignore that here
        import io
        from contextlib import redirect_stderr
        with redirect_stderr(io.StringIO()):
            splits = _split_by_shabad(manifest)
        card = _build_data_card(
            out_dir=Path("/tmp/run_x"),
            source_key="kirtan",
            source_id="surindersinghssj/gurbani-kirtan-yt-captions-300h-canonical",
            args=_args(),
            splits=splits,
            manifest=manifest,
            rejections={k: 0 for k in ("holdout_shabad", "holdout_video", "score_low",
                                       "dur_short", "dur_long", "simran")},
            holdout_shabads={"4377", "1821", "1341", "3712"},
            holdout_videos=set(),
            apply_holdout=True,
        )
        # The warning glyph indicates the low-diversity case
        self.assertIn("⚠️", card)


if __name__ == "__main__":
    unittest.main()
