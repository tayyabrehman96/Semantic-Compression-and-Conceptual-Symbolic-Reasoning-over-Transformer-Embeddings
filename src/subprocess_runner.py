"""Run embedding and training in separate OS processes (releases GPU RAM between stages)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_isolated_pipeline(
    work_dir: Path,
    embed: str,
    mode: str,
    kmeans: int,
    n_jobs: int | None,
    results_parent: Path | None,
) -> None:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    artifact = work_dir / "stage_embed.npz"

    cmd1 = [
        sys.executable,
        "-m",
        "src",
        "cache-embed",
        "--embed",
        embed,
        "--out",
        str(artifact),
    ]
    subprocess.run(cmd1, check=True)

    cmd2 = [sys.executable, "-m", "src", "train", "--artifact", str(artifact), "--mode", mode]
    if kmeans > 0:
        cmd2.extend(["--kmeans", str(kmeans)])
    if n_jobs is not None:
        cmd2.extend(["--n-jobs", str(n_jobs)])
    if results_parent is not None:
        cmd2.extend(["--out", str(results_parent)])
    subprocess.run(cmd2, check=True)
