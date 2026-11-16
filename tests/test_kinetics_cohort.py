import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_kinetics_cohort.py"
SPEC = importlib.util.spec_from_file_location("audit_kinetics_cohort", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selection_is_stratified_deterministic_and_excludes_nondevelopment():
    records = [
        {"case_id": f"{source}_{index}", "source": source, "dataset_role": "development"}
        for source in ("A", "B")
        for index in range(5)
    ]
    records.append({"case_id": "X_0", "source": "X", "dataset_role": "locked_external_test"})
    first = MODULE.select_records(records, per_source=2, seed=9)
    second = MODULE.select_records(list(reversed(records)), per_source=2, seed=9)
    assert first == second
    assert len(first) == 4
    assert {record["source"] for record in first} == {"A", "B"}
    assert MODULE.case_seed(9, "A_1") == MODULE.case_seed(9, "A_1")
    assert MODULE.case_seed(9, "A_1") != MODULE.case_seed(9, "A_2")
