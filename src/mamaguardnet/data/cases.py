from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    patient_id: str
    source: str
    pre: str
    post_early: str
    post_late: str
    mask: str | None = None
    breast_mask: str | None = None
    available_phases: tuple[str, ...] = ()
    dataset_role: str = "development"


REQUIRED_COLUMNS = {"case_id", "patient_id", "source", "pre", "post_early", "post_late"}


def read_case_manifest(path: Path) -> list[CaseRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        records = [
            CaseRecord(
                case_id=row["case_id"].strip(),
                patient_id=row["patient_id"].strip(),
                source=row["source"].strip(),
                pre=row["pre"].strip(),
                post_early=row["post_early"].strip(),
                post_late=row["post_late"].strip(),
                mask=(row.get("mask") or "").strip() or None,
                breast_mask=(row.get("breast_mask") or "").strip() or None,
                available_phases=tuple(
                    item for item in (row.get("available_phases") or "").split(";") if item
                ),
                dataset_role=(row.get("dataset_role") or "development").strip(),
            )
            for row in reader
        ]
    validate_case_records(records)
    return records


def validate_case_records(records: list[CaseRecord]) -> None:
    case_ids = [record.case_id for record in records]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_id values must be unique")
    for record in records:
        if not record.patient_id or not record.source:
            raise ValueError(f"Missing patient/source identity for {record.case_id}")
        for phase in (record.pre, record.post_early, record.post_late):
            if not phase:
                raise ValueError(f"Missing required phase for {record.case_id}")


def write_case_manifest(path: Path, records: list[CaseRecord]) -> None:
    validate_case_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id", "patient_id", "source", "pre", "post_early", "post_late",
        "mask", "breast_mask", "available_phases", "dataset_role",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "case_id": record.case_id,
                "patient_id": record.patient_id,
                "source": record.source,
                "pre": record.pre,
                "post_early": record.post_early,
                "post_late": record.post_late,
                "mask": record.mask or "",
                "breast_mask": record.breast_mask or "",
                "available_phases": ";".join(record.available_phases),
                "dataset_role": record.dataset_role,
            })
