import json
from pathlib import Path

import yaml

from mamaguardnet.orchestration.fold_gate import inspect_fold_gate


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    preprocessed = tmp_path / "preprocessed"
    dataset = preprocessed / "Dataset501_Test"
    dataset.mkdir(parents=True)
    (dataset / "splits_final.json").write_text(
        json.dumps([{"train": ["case_c"], "val": ["case_a", "case_b"]}]),
        encoding="utf-8",
    )

    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "experiment": "A_test",
                "dataset": 501,
                "configuration": "3d_fullres",
                "plans": "Plans",
                "trainer": "Trainer",
            }
        ),
        encoding="utf-8",
    )
    return preprocessed, tmp_path / "results", config


def test_fold_gate_requires_complete_exact_validation_artifacts(tmp_path: Path):
    preprocessed, results, config = _fixture(tmp_path)
    fold_root = (
        results
        / "A_test"
        / "seed_7"
        / "Dataset501_Test"
        / "Trainer__Plans__3d_fullres"
        / "fold_0"
    )
    validation = fold_root / "validation"
    validation.mkdir(parents=True)

    incomplete = inspect_fold_gate(
        results_root=results,
        preprocessed_root=preprocessed,
        config_paths=[config],
        fold=0,
        seed=7,
    )
    assert not incomplete["ready"]

    (fold_root / "checkpoint_final.pth").write_bytes(b"0" * 1_000_000)
    (validation / "summary.json").write_text('{"dice": 0.75}', encoding="utf-8")
    (validation / "case_a.nii.gz").write_bytes(b"prediction")
    (validation / "case_b.nii.gz").write_bytes(b"prediction")

    complete = inspect_fold_gate(
        results_root=results,
        preprocessed_root=preprocessed,
        config_paths=[config],
        fold=0,
        seed=7,
    )
    assert complete["ready"]
    assert complete["experiments"][0]["observed_predictions"] == 2


def test_fold_gate_rejects_nonfinite_summary_and_unexpected_prediction(tmp_path: Path):
    preprocessed, results, config = _fixture(tmp_path)
    fold_root = (
        results
        / "A_test"
        / "seed_7"
        / "Dataset501_Test"
        / "Trainer__Plans__3d_fullres"
        / "fold_0"
    )
    validation = fold_root / "validation"
    validation.mkdir(parents=True)
    (fold_root / "checkpoint_final.pth").write_bytes(b"0" * 1_000_000)
    (validation / "summary.json").write_text('{"dice": NaN}', encoding="utf-8")
    for case in ("case_a", "case_b", "case_x"):
        (validation / f"{case}.nii.gz").write_bytes(b"prediction")

    report = inspect_fold_gate(
        results_root=results,
        preprocessed_root=preprocessed,
        config_paths=[config],
        fold=0,
        seed=7,
    )
    assert not report["ready"]
    assert report["experiments"][0]["unexpected_cases"] == ["case_x"]
    assert "validation summary contains a non-finite number" in report["experiments"][0]["issues"]
