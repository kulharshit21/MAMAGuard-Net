from pathlib import Path

import pytest

from mamaguardnet.orchestration.registry import DuplicateRun, RunRegistry


def test_duplicate_configuration_is_refused(tmp_path: Path):
    registry = RunRegistry(tmp_path / "runs.json")
    first = registry.register({"x": 1})
    with pytest.raises(DuplicateRun):
        registry.register({"x": 1})
    assert registry.update_status(first["run_id"], "running")["status"] == "running"

