"""
Micro-program 01: full sklearn pipeline (sentence-transformer embeddings, parallel heads).

Run from repository root:
  python examples/run_01_sklearn_default.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-m", "src", "run", "--mode", "sklearn", "--embed", "sentence_transformer"],
        cwd=root,
        check=True,
    )


if __name__ == "__main__":
    main()
