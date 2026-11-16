from pathlib import Path

import pytest

from mamaguardnet.cli import _assert_training_data_ready


def test_training_gate_requires_reviewed_plans_and_splits(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("nnUNet_preprocessed", str(tmp_path))
    dataset = tmp_path / "Dataset501_MAMAMIA"
    dataset.mkdir()
    with pytest.raises(SystemExit):
        _assert_training_data_ready(501)
    (dataset / "nnUNetPlans.json").write_text("{}")
    with pytest.raises(SystemExit):
        _assert_training_data_ready(501)
    (dataset / "splits_final.json").write_text("[]")
    _assert_training_data_ready(501)

    with pytest.raises(SystemExit):
        _assert_training_data_ready(501, "nnUNetResEncUNetLPlans")
    (dataset / "nnUNetResEncUNetLPlans.json").write_text("{}")
    _assert_training_data_ready(501, "nnUNetResEncUNetLPlans")
