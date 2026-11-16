from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from mamaguardnet.data.cases import CaseRecord, validate_case_records

PHASE = re.compile(r"P(\d+)\.nii\.gz$", re.IGNORECASE)


def _source_from_case(case_id: str, metadata_path: Path | None = None) -> str:
    if metadata_path and metadata_path.is_file():
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
        source = document.get("imaging_data", {}).get("dataset")
        if source:
            return str(source).upper()
    return case_id.split("_", 1)[0].upper()


def discover_mamamia(data_root: Path) -> tuple[list[CaseRecord], list[str]]:
    images_root = data_root / "images"
    expert_root = data_root / "segmentations" / "expert"
    info_root = data_root / "patient_info_files"
    if not images_root.is_dir():
        raise FileNotFoundError(f"MAMA-MIA images directory not found: {images_root}")
    records: list[CaseRecord] = []
    issues: list[str] = []
    for case_dir in sorted(path for path in images_root.iterdir() if path.is_dir()):
        case_id = case_dir.name
        phases = sorted(case_dir.glob(f"{case_id}_[0-9][0-9][0-9][0-9].nii.gz"))
        by_index = {path.name[-11:-7]: path for path in phases}
        required = [by_index.get(f"{index:04d}") for index in range(3)]
        mask = expert_root / f"{case_id}.nii.gz"
        missing = [
            name
            for name, path in zip(("0000", "0001", "0002"), required, strict=True)
            if path is None
        ]
        if missing or not mask.is_file():
            issues.append(
                f"{case_id}: missing phases={missing or 'none'}, expert_mask={mask.is_file()}"
            )
            continue
        records.append(CaseRecord(
            case_id=case_id,
            patient_id=case_id,
            source=_source_from_case(case_id, info_root / f"{case_id}.json"),
            pre=str(required[0].resolve()),
            post_early=str(required[1].resolve()),
            post_late=str(required[2].resolve()),
            mask=str(mask.resolve()),
            available_phases=tuple(str(path.resolve()) for path in phases),
            dataset_role="development",
        ))
    validate_case_records(records)
    return records, issues


def locate_secondary_root(data_root: Path) -> Path:
    candidates = [
        data_root / "Breast_Cancer_DCE-MRI_Data" / "Breast_Cancer_DCE-MRI_Data",
        data_root / "Breast_Cancer_DCE-MRI_Data",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(path.is_dir() and path.name.isdigit() for path in candidate.iterdir()):
            return candidate
    raise FileNotFoundError("Could not locate the numeric 100-case secondary cohort")


def discover_yunnan(data_root: Path) -> tuple[list[CaseRecord], list[str]]:
    root = locate_secondary_root(data_root)
    records: list[CaseRecord] = []
    issues: list[str] = []
    case_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    for outer in case_dirs:
        raw_id = outer.name
        case_root = outer / raw_id if (outer / raw_id).is_dir() else outer
        phases = sorted(
            (path for path in case_root.glob("P*.nii.gz") if PHASE.fullmatch(path.name)),
            key=lambda path: int(PHASE.fullmatch(path.name).group(1)),
        )
        by_index = {int(PHASE.fullmatch(path.name).group(1)): path for path in phases}
        gt = case_root / "GT.nii.gz"
        breast_mask = case_root / "Breast_mask.nii.gz"
        missing = [f"P{index}" for index in range(3) if index not in by_index]
        if missing or not gt.is_file() or not breast_mask.is_file():
            issues.append(
                f"{raw_id}: missing={missing}, GT={gt.is_file()}, breast_mask={breast_mask.is_file()}"
            )
            continue
        case_id = f"YUNNAN_{int(raw_id):03d}"
        records.append(CaseRecord(
            case_id=case_id,
            patient_id=case_id,
            source="YUNNAN",
            pre=str(by_index[0].resolve()),
            post_early=str(by_index[1].resolve()),
            post_late=str(by_index[2].resolve()),
            mask=str(gt.resolve()),
            breast_mask=str(breast_mask.resolve()),
            available_phases=tuple(str(path.resolve()) for path in phases),
            dataset_role="locked_external_test",
        ))
    validate_case_records(records)
    return records, issues


def layout_summary(records: list[CaseRecord], issues: list[str]) -> dict:
    return {
        "case_count": len(records),
        "source_counts": dict(Counter(record.source for record in records)),
        "available_phase_count_distribution": dict(
            Counter(len(record.available_phases) for record in records)
        ),
        "issues": issues,
    }
