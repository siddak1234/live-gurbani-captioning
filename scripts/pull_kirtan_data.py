#!/usr/bin/env python3
"""DEPRECATED: thin back-compat shim for ``scripts/pull_dataset.py kirtan``.

The unified data-pull entrypoint moved to ``scripts/pull_dataset.py`` with
subcommands per source. This shim translates the old flat-flag CLI to the new
subcommand form so existing call sites and submission notes keep working.

New code should call ``python scripts/pull_dataset.py kirtan ...`` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

if __name__ == "__main__":
    # Prepend "kirtan" subcommand to argv and dispatch.
    sys.argv = [sys.argv[0], "kirtan", *sys.argv[1:]]
    from pull_dataset import main
    raise SystemExit(main())
