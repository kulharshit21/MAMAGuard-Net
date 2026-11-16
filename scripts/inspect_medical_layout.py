#!/usr/bin/env python3
"""Inspect the concrete MAMA-MIA and 100-case secondary dataset layouts."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def normalized_name(name: str, case_id: str) -> str:
    return re.sub(re.escape(case_id), "{case}", name, flags=re.IGNORECASE)


def file_record(path: Path, root: Path) -> dict:
    return {"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size}


def read_csv_shape(path: Path, sample_rows: int = 3) -> dict:
    if not path.is_file():
        return {"exists": False, "path": str(path)}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        count = 0
        for row in reader:
            count += 1
            if len(rows) < sample_rows:
                rows.append(row)
    return {
        "exists": True,
        "path": str(path),
        "columns": reader.fieldnames,
        "row_count": count,
        "sample_rows": rows,
    }


def inspect_primary(root: Path) -> dict:
    images = root / "images"
    cases = sorted((path for path in images.iterdir() if path.is_dir()), key=lambda path: path.name) if images.is_dir() else []
    patterns: Counter[str] = Counter()
    counts: Counter[int] = Counter()
    samples: dict[str, list[dict]] = {}
    for case in cases:
        files = sorted(path for path in case.rglob("*") if path.is_file() and path.name not in {"manifest.csv", "SYNAPSE_METADATA_MANIFEST.tsv"})
        counts[len(files)] += 1
        for path in files:
            patterns[normalized_name(path.name, case.name)] += 1
        if len(samples) < 5:
            samples[case.name] = [file_record(path, root) for path in files]

    segmentation = {}
    for kind in ("expert", "automatic"):
        directory = root / "segmentations" / kind
        files = sorted(path for path in directory.rglob("*") if path.is_file()) if directory.is_dir() else []
        segmentation[kind] = {
            "count": len(files),
            "samples": [file_record(path, root) for path in files[:10]],
        }

    info_dir = root / "patient_info_files"
    info_files = sorted(info_dir.glob("*.json")) if info_dir.is_dir() else []
    info_sample = None
    if info_files:
        info_sample = {
            "path": str(info_files[0].relative_to(root)),
            "content": json.loads(info_files[0].read_text(encoding="utf-8")),
        }
    return {
        "case_count": len(cases),
        "case_file_count_distribution": dict(sorted(counts.items())),
        "filename_patterns": dict(patterns.most_common()),
        "case_samples": samples,
        "segmentations": segmentation,
        "manifest": read_csv_shape(root / "manifest.csv"),
        "train_test_splits": read_csv_shape(root / "train_test_splits.csv"),
        "quality_scores": read_csv_shape(root / "preliminary_automatic_segmentations_quality_scores.csv"),
        "patient_info_count": len(info_files),
        "patient_info_sample": info_sample,
    }


def locate_secondary(root: Path) -> Path | None:
    candidates = [
        root / "Breast_Cancer_DCE-MRI_Data" / "Breast_Cancer_DCE-MRI_Data",
        root / "Breast_Cancer_DCE-MRI_Data",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(path.is_dir() and path.name.isdigit() for path in candidate.iterdir()):
            return candidate
    return None


def inspect_secondary(root: Path) -> dict:
    secondary = locate_secondary(root)
    if secondary is None:
        return {"exists": False}
    case_roots = sorted(
        (path for path in secondary.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    patterns: Counter[str] = Counter()
    counts: Counter[int] = Counter()
    cases = {}
    for outer in case_roots:
        case_id = outer.name
        inner = outer / case_id
        case_root = inner if inner.is_dir() else outer
        files = sorted(
            path for path in case_root.rglob("*")
            if path.is_file() and not path.name.startswith(".mamaguard_")
        )
        counts[len(files)] += 1
        for path in files:
            patterns[normalized_name(path.name, case_id)] += 1
        cases[case_id] = [file_record(path, root) for path in files]
    return {
        "exists": True,
        "root": str(secondary),
        "case_count": len(case_roots),
        "case_file_count_distribution": dict(sorted(counts.items())),
        "filename_patterns": dict(patterns.most_common()),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "data_root": str(args.data_root.resolve()),
        "primary_mamamia": inspect_primary(args.data_root),
        "secondary_100_case": inspect_secondary(args.data_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "primary_cases": result["primary_mamamia"]["case_count"],
        "primary_patterns": result["primary_mamamia"]["filename_patterns"],
        "secondary_cases": result["secondary_100_case"].get("case_count"),
        "secondary_patterns": result["secondary_100_case"].get("filename_patterns"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
