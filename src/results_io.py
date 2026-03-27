"""Timestamped run folders: metrics CSV + JSON manifest for reproducibility."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from . import config


def _try_git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def make_run_dir(base: Path | None = None) -> Path:
    base = base or config.RESULTS_DIR
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run = base / f"run_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    return run


def build_manifest(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    m: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "random_seed": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "data_csv": str(config.DATA_CSV),
        "text_column": config.TEXT_COLUMN,
        "label_column": config.LABEL_COLUMN,
        "git_commit": _try_git_commit(),
    }
    if extra:
        m.update(extra)
    return m


def save_run(
    run_dir: Path,
    manifest: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
    filename: str = "metrics_sklearn.csv",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    pd.DataFrame(metrics_rows).to_csv(run_dir / filename, index=False)
