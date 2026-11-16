import pytest

from mamaguardnet.data.cases import CaseRecord
from mamaguardnet.data.splits import make_patient_splits, validate_no_overlap


def _case(index: int, source: str = "dev") -> CaseRecord:
    return CaseRecord(f"c{index}", f"p{index}", source, "pre", "early", "late", "mask")


def test_patient_splits_are_deterministic_and_external_source_locked():
    records = [_case(index) for index in range(20)] + [_case(100, "external")]
    first = make_patient_splits(records, holdout_sources=("external",))
    second = make_patient_splits(records, holdout_sources=("external",))
    assert first == second
    assert {row["patient_id"] for row in first["splits"]["external"]} == {"p100"}
    validate_no_overlap(first)


def test_overlap_validator_fails_closed():
    with pytest.raises(ValueError, match="leakage"):
        validate_no_overlap({"train": [{"patient_id": "p1"}], "test": [{"patient_id": "p1"}]})


def test_mixed_source_patient_is_rejected():
    records = [
        CaseRecord("a", "p1", "dev", "p", "e", "l"),
        CaseRecord("b", "p1", "external", "p", "e", "l"),
    ]
    with pytest.raises(ValueError, match="span"):
        make_patient_splits(records, holdout_sources=("external",))

