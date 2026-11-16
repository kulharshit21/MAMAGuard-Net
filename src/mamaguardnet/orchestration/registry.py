from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from mamaguardnet.utils.atomic import atomic_write_json
from mamaguardnet.utils.hashing import hash_object


class DuplicateRun(RuntimeError):
    pass


class RunRegistry:
    def __init__(self, path: Path):
        self.path = path

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"runs": []}

    def register(self, config: dict, *, allow_duplicate: bool = False) -> dict:
        document = self._read()
        config_hash = hash_object({key: value for key, value in config.items() if key != "_meta"})
        existing = [run for run in document["runs"] if run["config_hash"] == config_hash]
        if existing and not allow_duplicate:
            raise DuplicateRun(f"Configuration already registered as {existing[-1]['run_id']}")
        timestamp = datetime.now(timezone.utc)
        run = {
            "run_id": f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{config_hash[:10]}",
            "config_hash": config_hash,
            "status": "planned",
            "created_at_utc": timestamp.isoformat(),
            "config": config,
        }
        document["runs"].append(run)
        atomic_write_json(self.path, document)
        return run

    def list(self) -> list[dict]:
        return self._read()["runs"]

    def update_status(self, run_id: str, status: str) -> dict:
        allowed = {"planned", "running", "completed", "failed", "stopped"}
        if status not in allowed:
            raise ValueError(f"Unsupported status: {status}")
        document = self._read()
        for run in document["runs"]:
            if run["run_id"] == run_id:
                run["status"] = status
                run["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                atomic_write_json(self.path, document)
                return run
        raise KeyError(run_id)

