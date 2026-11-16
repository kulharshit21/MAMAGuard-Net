#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mamaguardnet.data.cases import read_case_manifest
from mamaguardnet.data.splits import make_patient_splits
from mamaguardnet.utils.atomic import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--holdout-source", action="append", default=[])
    args = parser.parse_args()
    document = make_patient_splits(
        read_case_manifest(args.manifest), seed=args.seed, holdout_sources=tuple(args.holdout_source)
    )
    atomic_write_json(args.output, document)
    print(f"Wrote locked split {document['sha256']} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

