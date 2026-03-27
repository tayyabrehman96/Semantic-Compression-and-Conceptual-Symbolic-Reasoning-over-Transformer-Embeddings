"""Backward compatibility: `python -m src.run_experiment` (flat flags). Prefer `python -m src run`."""

from __future__ import annotations

from .cli import main_legacy_flat

if __name__ == "__main__":
    main_legacy_flat()
