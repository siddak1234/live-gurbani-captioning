from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_corpus import collect_shabad_ids, resolve_shabad_ids  # noqa: E402


def _write_gt(dir_: Path, name: str, shabad_id: int) -> None:
    (dir_ / f"{name}.json").write_text(json.dumps({
        "video_id": name,
        "shabad_id": shabad_id,
        "segments": [],
    }))


class TestBuildCorpusIds(unittest.TestCase):
    def test_collect_shabad_ids_dedupes_and_sorts_gt_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            gt_dir = Path(tmp)
            _write_gt(gt_dir, "case_b", 3712)
            _write_gt(gt_dir, "case_a", 1341)
            _write_gt(gt_dir, "case_c", 1341)

            self.assertEqual(collect_shabad_ids(gt_dir), [1341, 3712])

    def test_resolve_shabad_ids_adds_explicit_oos_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            gt_dir = Path(tmp)
            _write_gt(gt_dir, "case_a", 1341)

            self.assertEqual(
                resolve_shabad_ids(gt_dir, [5621, 1341, 1821]),
                [1341, 1821, 5621],
            )


if __name__ == "__main__":
    unittest.main()
