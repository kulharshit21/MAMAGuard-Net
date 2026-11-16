#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from mamaguardnet.data.external_lock import assert_external_allowed, consume_external
from mamaguardnet.utils.hashing import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Hash-bound, one-time external evaluation wrapper")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=Path("runs/external_yunnan.lock.json"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    config_hash, checkpoint_hash = sha256_file(args.config), sha256_file(args.checkpoint)
    assert_external_allowed(args.state, config_hash=config_hash, checkpoint_hash=checkpoint_hash)
    command = [
        sys.executable, str(Path(__file__).with_name("evaluate_predictions.py")),
        "--predictions", str(args.predictions), "--targets", str(args.targets),
        "--output-csv", str(args.output_csv),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode:
        return result.returncode
    consume_external(args.state, config_hash=config_hash, checkpoint_hash=checkpoint_hash)
    print("External token consumed after successful evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

