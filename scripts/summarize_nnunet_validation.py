#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mamaguardnet.evaluation.nnunet_summary import summarize_nnunet_validation


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    manifest = {record["case_id"]: record for record in records}
    if not records or len(manifest) != len(records):
        raise ValueError(f"Manifest is empty or has duplicate case IDs: {path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize an existing nnU-Net validation result")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    if not 0 <= args.fold < len(splits):
        raise ValueError(f"Fold {args.fold} is absent from {args.splits}")
    expected = splits[args.fold].get("val", splits[args.fold].get("validation"))
    if not isinstance(expected, list):
        raise ValueError("Split fold lacks val/validation case list")
    report = summarize_nnunet_validation(summary, read_manifest(args.manifest), expected)
    document = {
        "source_summary": str(args.summary.resolve()),
        "manifest": str(args.manifest.resolve()),
        "splits": str(args.splits.resolve()),
        "fold": args.fold,
        **report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
