#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mamaguardnet.data.external_lock import freeze_external
from mamaguardnet.utils.hashing import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze config/checkpoint for one external evaluation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=Path("runs/external_yunnan.lock.json"))
    args = parser.parse_args()
    freeze_external(
        args.state,
        config_hash=sha256_file(args.config),
        checkpoint_hash=sha256_file(args.checkpoint),
    )
    print(f"Frozen one-time external token: {args.state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

