#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from mamaguardnet.data.cases import read_case_manifest
from mamaguardnet.nnunet.staging import stage_mamamia_raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage three-phase MAMA-MIA for nnU-Net")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--raw-root", type=Path, default=os.getenv("nnUNet_raw"))
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="MAMAMIA3Phase")
    parser.add_argument("--mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    args = parser.parse_args()
    if args.raw_root is None:
        raise SystemExit("Set nnUNet_raw or pass --raw-root")
    destination = stage_mamamia_raw(
        read_case_manifest(args.manifest), args.raw_root, dataset_id=args.dataset_id,
        dataset_name=args.dataset_name, mode=args.mode,
    )
    print(f"Staged dataset: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
