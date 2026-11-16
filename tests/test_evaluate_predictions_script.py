import json
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_predictions.py"


def _save(path: Path, shape: tuple[int, int, int], affine: np.ndarray) -> None:
    data = np.zeros(shape, dtype=np.float32)
    data[1, 1, 1] = 1
    nib.save(nib.Nifti1Image(data, affine), path)


def test_evaluator_fails_closed_then_allows_explicit_nearest_resampling(tmp_path):
    predictions, targets = tmp_path / "predictions", tmp_path / "targets"
    predictions.mkdir()
    targets.mkdir()
    _save(predictions / "case.nii.gz", (3, 3, 3), np.eye(4))
    shifted = np.eye(4)
    shifted[0, 3] = 0.25
    _save(targets / "case.nii.gz", (4, 4, 4), shifted)
    output = tmp_path / "metrics.csv"
    command = [
        sys.executable,
        str(SCRIPT),
        "--predictions",
        str(predictions),
        "--targets",
        str(targets),
        "--output-csv",
        str(output),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")

    failed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert failed.returncode != 0
    assert "GeometryMismatchError" in failed.stderr

    completed = subprocess.run(
        [*command, "--resample-prediction", "--workers", "2", "--progress-every", "1"],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["case_count"] == 1
    assert summary["metrics"]["dice"]["all_finite"]
    assert summary["metrics"]["evaluation_seconds"]["mean"] >= 0
