"""Top-of-repo CLI tools.

Marked as a regular package (this ``__init__.py`` is intentional) so
``from scripts.pull_dataset import ...`` resolves to our local files
rather than getting shadowed by site-packages.

Reason: several pip deps (whisperkit, numba, accelerate as observed on
Python 3.12 venv) ship a top-level ``scripts/__init__.py`` into
site-packages. Without our own ``__init__.py``, Python's import
machinery treats our ``./scripts/`` as a namespace package, which loses
precedence to a regular package later on sys.path — and our test imports
fail with ``ModuleNotFoundError: No module named 'scripts.pull_dataset'``.
A regular package (this file) takes precedence over the namespace fallback
even when sys.path has both.

This file is deliberately empty of exports — it exists for the import
resolution behavior, not to namespace anything.
"""
