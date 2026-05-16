"""Project source tree.

Defensively marked as a regular package so ``from src.run_card import ...``,
``from src.matcher import ...``, etc. resolve to our local files even
if a future pip dep ships a ``src/__init__.py`` into site-packages.
Same pattern, same reason as ``scripts/__init__.py``.

This file is deliberately empty of exports.
"""
