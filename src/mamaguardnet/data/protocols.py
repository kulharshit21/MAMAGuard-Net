from __future__ import annotations

import random
from collections import Counter, defaultdict

from mamaguardnet.data.cases import CaseRecord
from mamaguardnet.utils.hashing import hash_object


def source_balanced_folds(
    records: list[CaseRecord], *, folds: int = 5, seed: int = 20261031,
) -> list[dict[str, list[str]]]:
    if folds < 2:
        raise ValueError("At least two folds are required")
    if any(record.dataset_role != "development" for record in records):
        raise ValueError("Only development cases may enter training folds")
    patients = [record.patient_id for record in records]
    if len(patients) != len(set(patients)):
        raise ValueError("Expected one case per patient for nnU-Net folds")
    by_source: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_source[record.source].append(record.case_id)
    validation: list[list[str]] = [[] for _ in range(folds)]
    for source in sorted(by_source):
        cases = sorted(by_source[source])
        random.Random(f"{seed}:{source}").shuffle(cases)
        for index, case_id in enumerate(cases):
            validation[index % folds].append(case_id)
    all_cases = {record.case_id for record in records}
    result = []
    for fold in range(folds):
        val = sorted(validation[fold])
        train = sorted(all_cases - set(val))
        if set(train) & set(val) or set(train) | set(val) != all_cases:
            raise AssertionError(f"Invalid partition in fold {fold}")
        result.append({"train": train, "val": val})
    return result


def source_holdout_protocols(records: list[CaseRecord]) -> dict[str, dict[str, list[str]]]:
    if any(record.dataset_role != "development" for record in records):
        raise ValueError("External cases cannot enter source holdout protocols")
    sources = sorted({record.source for record in records})
    return {
        source: {
            "train": sorted(record.case_id for record in records if record.source != source),
            "test": sorted(record.case_id for record in records if record.source == source),
        }
        for source in sources
    }


def protocol_document(records: list[CaseRecord], *, folds: int = 5, seed: int = 20261031) -> dict:
    nnunet_folds = source_balanced_folds(records, folds=folds, seed=seed)
    source_holdouts = source_holdout_protocols(records)
    core = {
        "seed": seed,
        "fold_count": folds,
        "case_count": len(records),
        "source_counts": dict(Counter(record.source for record in records)),
        "nnunet_folds": nnunet_folds,
        "source_holdouts": source_holdouts,
        "external_cases_included": False,
    }
    return {**core, "sha256": hash_object(core)}
