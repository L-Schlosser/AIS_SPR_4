"""End-to-end pipeline: download → export ONNX → benchmark."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    py = sys.executable
    run([py, "download_model.py"])
    run([py, "export_onnx.py"])
    run([py, "benchmark.py", "--output", "benchmark_results.json"])
    run([py, "inference.py", "--document-type", "invoice"])
    print("\nPipeline complete. Deploy models/onnx/model.onnx to Flutter.")


if __name__ == "__main__":
    main()
