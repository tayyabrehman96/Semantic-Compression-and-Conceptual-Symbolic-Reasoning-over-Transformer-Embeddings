"""
Micro-program 02: E5 embeddings + K-Means distance features + sklearn heads.

Run from repository root:
  python examples/run_02_e5_kmeans.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "run",
            "--mode",
            "sklearn",
            "--embed",
            "e5",
            "--kmeans",
            "8",
        ],
        cwd=root,
        check=True,
    )


if __name__ == "__main__":
    main()
