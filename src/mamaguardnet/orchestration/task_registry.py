from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mamaguardnet.utils.atomic import atomic_write_json
from mamaguardnet.utils.hashing import hash_object

TASK_STATUSES = {
    "PLANNED",
    "READY_FOR_TEST",
    "TESTING",
    "TEST_PASSED",
    "WAITING_FOR_DEPENDENCY",
    "WAITING_FOR_GPU",
    "WAITING_FOR_APPROVAL",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "BLOCKED",
    "INVALIDATED",
    "SKIPPED_WITH_REASON",
}

TASK_TYPES = {"ENGINEERING", "GPU_RESEARCH", "EXTERNAL_EVALUATION"}

TERMINAL_STATUSES = {"COMPLETED", "INVALIDATED", "SKIPPED_WITH_REASON"}

TRANSITIONS = {
    "PLANNED": {
        "READY_FOR_TEST",
        "WAITING_FOR_DEPENDENCY",
        "WAITING_FOR_APPROVAL",
        "BLOCKED",
        "SKIPPED_WITH_REASON",
    },
    "READY_FOR_TEST": {"TESTING", "BLOCKED", "SKIPPED_WITH_REASON"},
    "TESTING": {"TEST_PASSED", "FAILED", "BLOCKED"},
    "TEST_PASSED": {
        "WAITING_FOR_DEPENDENCY",
        "WAITING_FOR_GPU",
        "WAITING_FOR_APPROVAL",
        "RUNNING",
        "COMPLETED",
        "BLOCKED",
    },
    "WAITING_FOR_DEPENDENCY": {
        "WAITING_FOR_GPU",
        "WAITING_FOR_APPROVAL",
        "RUNNING",
        "BLOCKED",
    },
    "WAITING_FOR_GPU": {"RUNNING", "WAITING_FOR_APPROVAL", "BLOCKED"},
    "WAITING_FOR_APPROVAL": {"WAITING_FOR_GPU", "RUNNING", "BLOCKED"},
    "RUNNING": {"COMPLETED", "FAILED", "BLOCKED", "INVALIDATED"},
    "FAILED": {"READY_FOR_TEST", "WAITING_FOR_DEPENDENCY", "BLOCKED"},
    "BLOCKED": {"READY_FOR_TEST", "WAITING_FOR_DEPENDENCY", "WAITING_FOR_APPROVAL"},
    "COMPLETED": {"INVALIDATED"},
    "INVALIDATED": set(),
    "SKIPPED_WITH_REASON": set(),
}

REQUIRED_FIELDS = {
    "task_id",
    "task_type",
    "scientific_hypothesis",
    "dependencies",
    "priority",
    "owner",
    "code_snapshot_hash",
    "data_split_hash",
    "configuration_hash",
    "resources",
    "resource_authorized",
    "expected_outputs",
    "test_requirements",
    "launch_approved",
    "status",
}


class DuplicateTask(RuntimeError):
    pass


class InvalidTask(RuntimeError):
    pass


class InvalidTransition(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_spec(spec: Mapping[str, Any]) -> None:
    missing = REQUIRED_FIELDS - set(spec)
    if missing:
        raise InvalidTask(f"Missing required task fields: {sorted(missing)}")
    if spec["task_type"] not in TASK_TYPES:
        raise InvalidTask(f"Unsupported task type: {spec['task_type']}")
    if spec["status"] not in TASK_STATUSES:
        raise InvalidTask(f"Unsupported task status: {spec['status']}")
    if not isinstance(spec["task_id"], str) or not spec["task_id"].strip():
        raise InvalidTask("task_id must be a non-empty string")
    if not isinstance(spec["dependencies"], list):
        raise InvalidTask("dependencies must be a list")
    if not isinstance(spec["resources"], dict):
        raise InvalidTask("resources must be a mapping")
    if not isinstance(spec["priority"], int):
        raise InvalidTask("priority must be an integer")
    for field in ("resource_authorized", "launch_approved"):
        if not isinstance(spec[field], bool):
            raise InvalidTask(f"{field} must be boolean")


class TaskRegistry:
    """Atomic registry for engineering, GPU-research and external-evaluation tasks."""

    def __init__(self, path: Path, *, lock_timeout_seconds: float = 10.0):
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self.lock_timeout_seconds = lock_timeout_seconds

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "tasks": []}
        document = json.loads(self.path.read_text(encoding="utf-8"))
        if document.get("schema_version") != 1 or not isinstance(document.get("tasks"), list):
            raise InvalidTask(f"Unsupported registry document: {self.path}")
        return document

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.lock_timeout_seconds
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(descriptor, f"pid={os.getpid()}\n".encode())
            except FileExistsError as error:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out acquiring registry lock: {self.lock_path}"
                    ) from error
                time.sleep(0.05)
        try:
            yield
        finally:
            os.close(descriptor)
            self.lock_path.unlink(missing_ok=True)

    def register(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        _validate_spec(spec)
        task = dict(spec)
        immutable = {key: value for key, value in task.items() if key not in {"status", "task_id"}}
        task_hash = hash_object(immutable)
        now = _utc_now()
        task.update(
            {
                "task_hash": task_hash,
                "created_at_utc": now,
                "updated_at_utc": now,
                "start_timestamp_utc": None,
                "end_timestamp_utc": None,
                "checkpoint_resume": None,
                "failure_reason": None,
                "result_location": None,
                "history": [{"status": task["status"], "timestamp_utc": now}],
            }
        )
        with self._lock():
            document = self._read()
            for existing in document["tasks"]:
                if existing["task_id"] == task["task_id"]:
                    raise DuplicateTask(f"Task ID already registered: {task['task_id']}")
                if existing["task_hash"] == task_hash:
                    raise DuplicateTask(
                        f"Scientifically identical task already registered: {existing['task_id']}"
                    )
            document["tasks"].append(task)
            atomic_write_json(self.path, document)
        return task

    def list(self) -> list[dict[str, Any]]:
        return self._read()["tasks"]

    def get(self, task_id: str) -> dict[str, Any]:
        for task in self.list():
            if task["task_id"] == task_id:
                return task
        raise KeyError(task_id)

    def transition(
        self,
        task_id: str,
        status: str,
        *,
        failure_reason: str | None = None,
        checkpoint_resume: str | None = None,
        result_location: str | None = None,
    ) -> dict[str, Any]:
        if status not in TASK_STATUSES:
            raise InvalidTransition(f"Unsupported task status: {status}")
        with self._lock():
            document = self._read()
            task = next((item for item in document["tasks"] if item["task_id"] == task_id), None)
            if task is None:
                raise KeyError(task_id)
            current = task["status"]
            if status not in TRANSITIONS[current]:
                raise InvalidTransition(f"Invalid transition: {current} -> {status}")
            if status == "RUNNING":
                if not task["resource_authorized"]:
                    raise InvalidTransition("RUNNING requires resource authorization")
                if task["task_type"] in {"GPU_RESEARCH", "EXTERNAL_EVALUATION"} and not task[
                    "launch_approved"
                ]:
                    raise InvalidTransition(f"{task['task_type']} requires explicit launch approval")
                if task["task_type"] == "EXTERNAL_EVALUATION" and not task.get(
                    "external_lock_hash"
                ):
                    raise InvalidTransition("External evaluation requires a hash-bound lock")
                incomplete = [
                    dependency
                    for dependency in task["dependencies"]
                    if self._dependency_status(document, dependency) != "COMPLETED"
                ]
                if incomplete:
                    raise InvalidTransition(f"Incomplete dependencies: {incomplete}")
            if status in {"FAILED", "BLOCKED", "INVALIDATED", "SKIPPED_WITH_REASON"}:
                if not failure_reason:
                    raise InvalidTransition(f"{status} requires a recorded reason")
            now = _utc_now()
            task["status"] = status
            task["updated_at_utc"] = now
            task["history"].append({"status": status, "timestamp_utc": now})
            if status == "RUNNING" and task["start_timestamp_utc"] is None:
                task["start_timestamp_utc"] = now
            if status in TERMINAL_STATUSES or status == "FAILED":
                task["end_timestamp_utc"] = now
            if failure_reason is not None:
                task["failure_reason"] = failure_reason
            if checkpoint_resume is not None:
                task["checkpoint_resume"] = checkpoint_resume
            if result_location is not None:
                task["result_location"] = result_location
            atomic_write_json(self.path, document)
            return task

    @staticmethod
    def _dependency_status(document: Mapping[str, Any], task_id: str) -> str | None:
        for task in document["tasks"]:
            if task["task_id"] == task_id:
                return str(task["status"])
        return None
