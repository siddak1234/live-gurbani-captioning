"""Holdout regression guard for scripts/pull_dataset.py.

After PR 1.A, the holdout (4 benchmark shabads + benchmark/OOS videos) is loaded from
configs/datasets.yaml rather than hardcoded. These tests are the safety net:

1. Today's YAML produces EXACTLY the expected shabad + recording holdout.
   If someone edits the YAML wrong, this test catches it.
2. _is_held_out behaves correctly for both str and int shabad_id forms
   (HF parquet's canonical_shabad_id can come back as either type).
3. enforce=False sources (general Punjabi data) don't apply holdout.

PyYAML is the only non-stdlib dep here — tests skip gracefully when it
isn't importable, so a fresh clone (pre-make-install) still runs the rest
of the suite green. On any dev/training machine, yaml is in requirements
and these tests run.

Run:
    python -m unittest tests.test_pull_dataset_holdout -v
    make test
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import yaml  # noqa: F401
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

if _HAS_YAML:
    from scripts.pull_dataset import (  # noqa: E402
        _is_held_out,
        get_holdout,
        load_dataset_config,
    )


# Authoritative reference — these four IDs are the paired benchmark's shabads.
# If the benchmark ever changes them, this constant changes too AND the
# YAML must change to match. Keeping the truth in two places (here + YAML)
# is the regression guard: a one-sided edit breaks the test.
EXPECTED_BENCHMARK_SHABADS_STR = {"4377", "1821", "1341", "3712"}
EXPECTED_HOLDOUT_VIDEOS = {
    # Paired benchmark recordings.
    "IZOsmkdmmcg", "kZhIA8P6xWI", "kchMJPK9Axs", "zOtIpxMT9hU",
    # OOS v1 source recordings selected 2026-05-16.
    "hhpYbZ9_jH4", "ZdZ5sBLcjr0", "9SNXYPEVE60", "kZnV63eQOeM", "yr6Y3gzjAu4",
}


@unittest.skipUnless(_HAS_YAML, "PyYAML not installed; install via requirements*.txt")
class TestDatasetConfigLoading(unittest.TestCase):
    def test_loads_and_returns_dict(self):
        cfg = load_dataset_config()
        self.assertIsInstance(cfg, dict)
        self.assertIn("sources", cfg)
        self.assertIn("schema_version", cfg)

    def test_lists_expected_gurbani_sources(self):
        """The three SURT sources must exist in the registry; the script
        depends on their keys to look up HF repo IDs."""
        cfg = load_dataset_config()
        for required in ("kirtan", "sehaj", "sehajpath"):
            self.assertIn(required, cfg["sources"], f"missing source: {required}")


@unittest.skipUnless(_HAS_YAML, "PyYAML not installed")
class TestGetHoldout(unittest.TestCase):
    def test_kirtan_holdout_matches_benchmark(self):
        """REGRESSION GUARD: kirtan's holdout must exactly equal the
        benchmark's 4 shabads plus every locked eval recording. If the YAML drifts, this
        test fails loudly — much better than discovering the model
        was trained on benchmark data three weeks later."""
        shabads, videos, enforce = get_holdout("kirtan")
        self.assertTrue(enforce, "kirtan must enforce holdout")

        # shabads set contains BOTH str and int forms of each id
        for sid_str in EXPECTED_BENCHMARK_SHABADS_STR:
            self.assertIn(sid_str, shabads, f"missing str form: {sid_str}")
            self.assertIn(int(sid_str), shabads, f"missing int form: {sid_str}")

        self.assertEqual(videos, EXPECTED_HOLDOUT_VIDEOS)

    def test_sehaj_and_sehajpath_match_kirtan(self):
        """All Gurbani sources currently share the same holdout. If we
        ever introduce per-source holdout drift (e.g., sehaj develops
        its own benchmark), update this test deliberately."""
        kirtan = get_holdout("kirtan")
        for source in ("sehaj", "sehajpath"):
            other = get_holdout(source)
            self.assertEqual(kirtan[0], other[0], f"{source} shabads diverged from kirtan")
            self.assertEqual(kirtan[1], other[1], f"{source} videos diverged from kirtan")
            self.assertTrue(other[2], f"{source} must enforce holdout")

    def test_general_punjabi_sources_skip_holdout(self):
        """indicvoices/commonvoice are general Punjabi speech — they have
        no shabad metadata, so holdout doesn't apply. The YAML expresses
        this with enforce: false."""
        for source in ("indicvoices", "commonvoice"):
            _, _, enforce = get_holdout(source)
            self.assertFalse(enforce, f"{source} should not enforce holdout")

    def test_unknown_source_returns_safe_defaults(self):
        """Asking for a source the YAML doesn't define returns empty sets
        + enforce=False. Lets new pullers be added without YAML edits as
        a transitional state — they just won't have holdout."""
        shabads, videos, enforce = get_holdout("does_not_exist")
        self.assertEqual(shabads, set())
        self.assertEqual(videos, set())
        self.assertFalse(enforce)

    def test_custom_cfg_override(self):
        """Passing an explicit cfg dict bypasses the YAML loader — used
        by other tests to construct synthetic configs without writing
        temp files."""
        custom = {
            "sources": {
                "synthetic": {
                    "holdout": {
                        "enforce": True,
                        "shabad_ids": ["999"],
                        "video_ids": ["abc"],
                    }
                }
            }
        }
        shabads, videos, enforce = get_holdout("synthetic", cfg=custom)
        self.assertTrue(enforce)
        self.assertEqual(shabads, {"999", 999})
        self.assertEqual(videos, {"abc"})


@unittest.skipUnless(_HAS_YAML, "PyYAML not installed")
class TestIsHeldOut(unittest.TestCase):
    """_is_held_out is the per-row filter inside _run_surt_puller. These
    tests check it against the *actual* YAML-resolved sets (not synthetic
    fixtures) so they double-verify the YAML→runtime path end-to-end."""

    @classmethod
    def setUpClass(cls):
        cls.shabads, cls.videos, _ = get_holdout("kirtan")

    def test_benchmark_shabad_id_as_int(self):
        """Parquet column may return shabad_id as int."""
        held, reason = _is_held_out(4377, "different_video",
                                    shabads=self.shabads, videos=self.videos)
        self.assertTrue(held)
        self.assertEqual(reason, "shabad")

    def test_benchmark_shabad_id_as_str(self):
        """Or as str — both must trigger holdout."""
        held, reason = _is_held_out("4377", "different_video",
                                    shabads=self.shabads, videos=self.videos)
        self.assertTrue(held)
        self.assertEqual(reason, "shabad")

    def test_non_benchmark_shabad_passes(self):
        held, reason = _is_held_out(9999, "non_benchmark_video",
                                    shabads=self.shabads, videos=self.videos)
        self.assertFalse(held)
        self.assertEqual(reason, "")

    def test_benchmark_video_blocks_even_with_non_benchmark_shabad(self):
        """A benchmark video with a different shabad still must be dropped
        — same recording, different snippet would still leak audio
        signature into training."""
        held, reason = _is_held_out(9999, "IZOsmkdmmcg",
                                    shabads=self.shabads, videos=self.videos)
        self.assertTrue(held)
        self.assertEqual(reason, "video")

    def test_oos_video_blocks_even_with_non_benchmark_shabad(self):
        """Once a recording becomes OOS eval material, it must also be
        excluded from all future training pulls."""
        held, reason = _is_held_out(9999, "hhpYbZ9_jH4",
                                    shabads=self.shabads, videos=self.videos)
        self.assertTrue(held)
        self.assertEqual(reason, "video")

    def test_shabad_check_wins_over_video_check(self):
        """Order-of-check matters for the data card's rejection tally —
        a row that hits BOTH filters is counted under shabad. This is a
        behavioral lock-in, not an unchangeable contract; if the order
        flips, update the test deliberately."""
        held, reason = _is_held_out(4377, "IZOsmkdmmcg",
                                    shabads=self.shabads, videos=self.videos)
        self.assertTrue(held)
        self.assertEqual(reason, "shabad")


if __name__ == "__main__":
    unittest.main()
