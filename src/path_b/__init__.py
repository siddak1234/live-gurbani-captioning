"""Path B: CTC phoneme scoring + loop-aware HMM line tracker.

Empty scaffold — implementation lives in subsequent commits. The shared
infrastructure (audio download, BaniDB corpus, scoring/visualization,
submission folder layout) is identical to Path A and lives at the repo top
level. Only the engine internals differ.

Path A (frozen at 86.5% blind+live) lives in `src/asr.py`, `src/matcher.py`,
`src/smoother.py`, `src/shabad_id.py`, run via `scripts/run_path_a.py`.
"""
