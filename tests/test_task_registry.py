from pathlib import Path

import pytest

from mamaguardnet.orchestration.task_registry import (
    DuplicateTask,
    InvalidTransition,
    TaskRegistry,
)


def task_spec(task_id: str, *, task_type: str = "ENGINEERING", dependencies=None) -> dict:
    return {
        "task_id": task_id,
        "task_type": task_type,
        "scientific_hypothesis": "A prespecified testable hypothesis",
        "dependencies": dependencies or [],
        "priority": 1,
        "owner": "research-team",
        "code_snapshot_hash": "a" * 64,
        "data_split_hash": "b" * 64,
        "configuration_hash": "c" * 64,
        "resources": {"cpu_cores": 2, "ram_gib": 4, "gpus": 0},
        "resource_authorized": True,
        "expected_outputs": ["report.json"],
        "test_requirements": ["unit tests"],
        "launch_approved": task_type == "ENGINEERING",
        "status": "PLANNED",
    }


def test_task_registry_rejects_duplicate_and_invalid_transition(tmp_path: Path):
    registry = TaskRegistry(tmp_path / "tasks.json")
    registry.register(task_spec("T1"))
    with pytest.raises(DuplicateTask):
        registry.register(task_spec("T1"))
    duplicate_science = task_spec("T1-DIFFERENT-ID")
    with pytest.raises(DuplicateTask, match="Scientifically identical"):
        registry.register(duplicate_science)
    with pytest.raises(InvalidTransition):
        registry.transition("T1", "RUNNING")


def test_gpu_research_cannot_run_without_explicit_approval(tmp_path: Path):
    registry = TaskRegistry(tmp_path / "tasks.json")
    spec = task_spec("E1", task_type="GPU_RESEARCH")
    spec["launch_approved"] = False
    registry.register(spec)
    registry.transition("E1", "READY_FOR_TEST")
    registry.transition("E1", "TESTING")
    registry.transition("E1", "TEST_PASSED")
    registry.transition("E1", "WAITING_FOR_APPROVAL")
    with pytest.raises(InvalidTransition, match="explicit launch approval"):
        registry.transition("E1", "RUNNING")


def test_dependencies_must_complete_before_running(tmp_path: Path):
    registry = TaskRegistry(tmp_path / "tasks.json")
    registry.register(task_spec("D1"))
    registry.register(task_spec("D2", dependencies=["D1"]))
    for task_id in ("D1", "D2"):
        registry.transition(task_id, "READY_FOR_TEST")
        registry.transition(task_id, "TESTING")
        registry.transition(task_id, "TEST_PASSED")
    with pytest.raises(InvalidTransition, match="Incomplete dependencies"):
        registry.transition("D2", "RUNNING")
    registry.transition("D1", "COMPLETED", result_location="result.json")
    assert registry.transition("D2", "RUNNING")["status"] == "RUNNING"


def test_external_evaluation_requires_lock_hash(tmp_path: Path):
    registry = TaskRegistry(tmp_path / "tasks.json")
    spec = task_spec("E7", task_type="EXTERNAL_EVALUATION")
    spec["launch_approved"] = True
    registry.register(spec)
    registry.transition("E7", "READY_FOR_TEST")
    registry.transition("E7", "TESTING")
    registry.transition("E7", "TEST_PASSED")
    with pytest.raises(InvalidTransition, match="hash-bound lock"):
        registry.transition("E7", "RUNNING")
