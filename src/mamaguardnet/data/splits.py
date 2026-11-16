from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict

from mamaguardnet.data.cases import CaseRecord
from mamaguardnet.utils.hashing import hash_object


def _patient_groups(records: list[CaseRecord]) -> dict[str, list[CaseRecord]]:
    groups: dict[str, list[CaseRecord]] = defaultdict(list)
    for record in records:
        groups[record.patient_id].append(record)
    return groups


def make_patient_splits(
    records: list[CaseRecord], *, seed: int = 20261031, val_fraction: float = 0.15,
    test_fraction: float = 0.15, holdout_sources: tuple[str, ...] = (),
) -> dict:
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Invalid validation/test fractions")
    holdout = set(holdout_sources)
    by_patient = _patient_groups(records)
    source_holdout = {
        patient for patient, cases in by_patient.items() if any(case.source in holdout for case in cases)
    }
    mixed = {
        patient for patient, cases in by_patient.items()
        if any(case.source in holdout for case in cases) and any(case.source not in holdout for case in cases)
    }
    if mixed:
        raise ValueError(f"Patients span holdout and development sources: {sorted(mixed)}")
    development = sorted(set(by_patient) - source_holdout)
    random.Random(seed).shuffle(development)
    n_test = round(len(development) * test_fraction)
    n_val = round(len(development) * val_fraction)
    patient_split = {
        "test": set(development[:n_test]),
        "val": set(development[n_test:n_test + n_val]),
        "train": set(development[n_test + n_val:]),
        "external": source_holdout,
    }
    result = {
        name: sorted(
            (asdict(case) for patient in patients for case in by_patient[patient]),
            key=lambda row: row["case_id"],
        )
        for name, patients in patient_split.items()
    }
    validate_no_overlap(result)
    metadata = {
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "holdout_sources": sorted(holdout),
        "patient_counts": {name: len(patients) for name, patients in patient_split.items()},
    }
    return {"metadata": metadata, "splits": result, "sha256": hash_object({"metadata": metadata, "splits": result})}


def validate_no_overlap(split_document: dict) -> None:
    splits = split_document.get("splits", split_document)
    patients = {name: {row["patient_id"] for row in rows} for name, rows in splits.items()}
    names = list(patients)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = patients[left] & patients[right]
            if overlap:
                raise ValueError(f"Patient leakage between {left} and {right}: {sorted(overlap)}")
