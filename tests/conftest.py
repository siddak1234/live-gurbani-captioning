"""Test setup shared across the suite.

Adds the repo root to ``sys.path`` so ``from src.run_card import ...`` and
``from scripts...`` resolve without needing an installed package or
``PYTHONPATH`` gymnastics. ``conftest.py`` is auto-loaded by pytest; for
plain ``python -m unittest discover`` runs, the same import is performed
by each test module via ``_path_setup`` at the top of the file.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
