"""Tests for M4 Pro compute audit helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.audit_m4pro_compute import parse_memory_gb  # noqa: E402


class TestAuditM4ProCompute(unittest.TestCase):
    def test_parse_memory_gb(self):
        self.assertEqual(parse_memory_gb("48 GB"), 48.0)
        self.assertEqual(parse_memory_gb("128 GB"), 128.0)
        self.assertIsNone(parse_memory_gb(None))
        self.assertIsNone(parse_memory_gb("unknown"))


if __name__ == "__main__":
    unittest.main()
