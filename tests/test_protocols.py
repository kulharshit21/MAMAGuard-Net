from mamaguardnet.data.cases import CaseRecord
from mamaguardnet.data.protocols import protocol_document, source_balanced_folds


def record(case_id: str, source: str, role: str = "development") -> CaseRecord:
    return CaseRecord(case_id, case_id, source, "p0", "p1", "p2", dataset_role=role)


def test_source_balanced_folds_cover_each_case_once_as_validation():
    records = [
        *[record(f"A_{index}", "A") for index in range(7)],
        *[record(f"B_{index}", "B") for index in range(4)],
    ]
    folds = source_balanced_folds(records, folds=3, seed=4)
    validation = [case for fold in folds for case in fold["val"]]
    assert sorted(validation) == sorted(item.case_id for item in records)
    for fold in folds:
        assert not set(fold["train"]) & set(fold["val"])
        assert len(fold["train"]) + len(fold["val"]) == len(records)


def test_protocol_excludes_external_cases_and_has_source_holdouts():
    records = [record("A1", "A"), record("A2", "A"), record("B1", "B")]
    document = protocol_document(records, folds=2)
    assert document["source_holdouts"]["A"]["test"] == ["A1", "A2"]
    assert document["source_holdouts"]["A"]["train"] == ["B1"]
    assert document["external_cases_included"] is False

    try:
        source_balanced_folds([*records, record("Y1", "YUNNAN", "locked_external_test")])
    except ValueError as error:
        assert "development" in str(error)
    else:
        raise AssertionError("External case was accepted into training folds")
