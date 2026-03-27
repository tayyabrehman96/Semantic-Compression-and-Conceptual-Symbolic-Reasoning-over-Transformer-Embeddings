"""
Micro-program 03: same as production two-step OS process (embed cache, then train).

Frees GPU memory between embedding and training.

Run from repository root:
  python examples/run_03_two_process_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    work = root / "_work"
    work.mkdir(exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "run-subprocess",
            "--work-dir",
            str(work),
            "--embed",
            "sentence_transformer",
            "--mode",
            "sklearn",
        ],
        cwd=root,
        check=True,
    )


if __name__ == "__main__":
    main()
