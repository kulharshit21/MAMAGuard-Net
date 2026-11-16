#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mamaguardnet.evaluation.reporting import (
    summarize_rows_by_group,
    validate_case_coverage,
)


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, object]] = []
        for row in csv.DictReader(handle):
            converted: dict[str, object] = {}
            for key, value in row.items():
                if key in {"case_id", "source", "patient_id", "dataset_role"}:
                    converted[key] = value
                elif value in {"True", "False"}:
                    converted[key] = value == "True"
                else:
                    try:
                        converted[key] = float(value)
                    except (TypeError, ValueError):
                        converted[key] = value
            rows.append(converted)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if not records or "case_id" not in records[0]:
        raise ValueError(f"Manifest lacks case_id records: {path}")
    result = {record["case_id"]: record for record in records}
    if len(result) != len(records):
        raise ValueError(f"Duplicate case IDs in manifest: {path}")
    return result


def load_expected_case_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("validation", value.get("case_ids"))
        if not isinstance(value, list):
            raise ValueError("Expected-case JSON must be a list or contain validation/case_ids")
        return [str(case_id) for case_id in value]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create patient-level and source-level summaries from evaluated metrics"
    )
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-cases", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_csv(args.metrics_csv)
    manifest = read_manifest(args.manifest)
    for row in rows:
        case_id = str(row["case_id"])
        if case_id not in manifest:
            raise ValueError(f"Metrics case is absent from manifest: {case_id}")
        record = manifest[case_id]
        row["source"] = record.get("source", "UNKNOWN")
        row["patient_id"] = record.get("patient_id", case_id)
        row["dataset_role"] = record.get("dataset_role", "UNKNOWN")

    expected = load_expected_case_ids(args.expected_cases)
    coverage = validate_case_coverage(rows, expected or [str(row["case_id"]) for row in rows])
    if not coverage["complete"]:
        raise ValueError(f"Patient coverage check failed: {coverage}")

    report = {
        "metrics_csv": str(args.metrics_csv.resolve()),
        "manifest": str(args.manifest.resolve()),
        "coverage": coverage,
        **summarize_rows_by_group(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
