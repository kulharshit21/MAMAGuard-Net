#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mamaguardnet.data.cases import read_case_manifest
from mamaguardnet.data.protocols import protocol_document
from mamaguardnet.utils.atomic import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Create locked MAMA-MIA research splits")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    document = protocol_document(
        read_case_manifest(args.manifest), folds=args.folds, seed=args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "protocol_splits.json", document)
    atomic_write_json(args.output_dir / "splits_final.json", document["nnunet_folds"])
    summary = {
        "sha256": document["sha256"],
        "case_count": document["case_count"],
        "source_counts": document["source_counts"],
        "fold_validation_counts": [len(fold["val"]) for fold in document["nnunet_folds"]],
        "source_holdout_test_counts": {
            source: len(split["test"])
            for source, split in document["source_holdouts"].items()
        },
        "external_cases_included": document["external_cases_included"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
