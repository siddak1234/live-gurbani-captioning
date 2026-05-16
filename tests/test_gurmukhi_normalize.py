"""Contract tests for src/matcher.py:normalize().

normalize() is NOT a Gurmukhi-preserving normalizer — it romanizes
(unidecode) + canonicalizes for matching. ASR text from Whisper and
canonical Gurmukhi from BaniDB both flow through this function so the
fuzzy matcher compares apples to apples.

These tests lock in the function's contract:
  1. Idempotent (repeat calls don't drift)
  2. Romanizes Gurmukhi to ASCII
  3. Lowercases
  4. Strips parentheticals like "(Naam)"
  5. Strips pangti markers and 'rahaau'
  6. Collapses whitespace

The whole matcher (88% benchmark score) depends on this function's
behavior staying stable. If a future edit changes any of these rules,
update the tests deliberately AND re-score the matcher.

Requires the ``unidecode`` package (already in requirements.txt).
Skipped on bare-clone envs where it isn't importable.

Run:
    python -m unittest tests.test_gurmukhi_normalize -v
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
    from src.matcher import normalize  # noqa: E402
    _HAS_MATCHER = True
except ImportError:
    _HAS_MATCHER = False


@unittest.skipUnless(_HAS_MATCHER, "src.matcher not importable (missing unidecode dep?)")
class TestNormalizeContract(unittest.TestCase):

    def test_idempotent(self):
        """normalize(normalize(x)) == normalize(x) — repeat calls must
        not drift. Critical because the matcher may double-normalize at
        boundaries between cache layers."""
        samples = [
            "ਨਾਮੁ ਜਪਹੁ",
            "naam japah",
            "Hello World",
            "(Rahaau) sat naam",
            "   leading and trailing   ",
            "",
        ]
        for s in samples:
            once = normalize(s)
            twice = normalize(once)
            self.assertEqual(once, twice, f"not idempotent on {s!r}: {once!r} → {twice!r}")

    def test_romanizes_gurmukhi_to_ascii(self):
        """The matcher needs ASR text and Gurmukhi text to land in the
        same space. unidecode does the conversion."""
        result = normalize("ਨਾਮੁ ਜਪਹੁ")
        self.assertTrue(result.isascii(), f"expected ASCII, got {result!r}")
        self.assertGreater(len(result), 0, "Gurmukhi input shouldn't normalize to empty")

    def test_lowercases(self):
        self.assertEqual(normalize("HELLO World"), "hello world")
        self.assertEqual(normalize("Naam Japah"), "naam japah")

    def test_strips_parentheticals(self):
        """Parenthetical pronunciation hints get dropped — they're not
        part of the line text the ASR will produce."""
        self.assertNotIn("rahaau-hint", normalize("naam (rahaau-hint) japah"))
        self.assertNotIn("(", normalize("text (note) here"))
        self.assertNotIn(")", normalize("text (note) here"))

    def test_strips_rahaau(self):
        """The word 'rahaau' is a structural marker in shabads (means
        'pause'); the ASR doesn't transcribe it as content."""
        result = normalize("japah rahaau naam")
        self.assertNotIn("rahaau", result)
        # The surrounding words survive
        self.assertIn("japah", result)
        self.assertIn("naam", result)

    def test_collapses_whitespace(self):
        """Multi-space and leading/trailing whitespace gets normalized
        to single spaces with no padding."""
        self.assertEqual(normalize("  hello    world  "), "hello world")
        self.assertEqual(normalize("a\nb\tc"), "a b c")

    def test_strips_non_word_chars(self):
        """Punctuation and stray symbols vanish — ASR doesn't emit them
        and the canonical lines don't depend on them."""
        result = normalize("hello, world! how's it going?")
        self.assertNotIn(",", result)
        self.assertNotIn("!", result)
        self.assertNotIn("?", result)
        # Underscores are \w so they DO survive — document this quirk via test
        self.assertEqual(normalize("a_b"), "a_b")

    def test_handles_empty_string(self):
        self.assertEqual(normalize(""), "")

    def test_handles_only_whitespace(self):
        self.assertEqual(normalize("   \t\n  "), "")


if __name__ == "__main__":
    unittest.main()
