#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or dry-run Group A sequentially")
    parser.add_argument("matrix", type=Path, default=Path("configs/ablations/group_a.yaml"), nargs="?")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    mode = "--execute" if args.execute else "--dry-run"
    for config in matrix["experiments"]:
        command = ["python", "-m", "mamaguardnet.cli", "train", config, mode]
        print(" ".join(command), flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

