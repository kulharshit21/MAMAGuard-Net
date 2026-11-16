import json
import os
import subprocess
import sys
from pathlib import Path


def test_queue_initializer_registers_approval_gated_tasks(tmp_path: Path):
    output = tmp_path / "tasks.json"
    environment = os.environ.copy()
    source = str((Path.cwd() / "src").resolve())
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source, environment.get("PYTHONPATH", "")) if value
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/initialize_publication_queue.py",
            "--output",
            str(output),
            "--split-hash",
            "d" * 64,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert "Registered 10 tasks" in completed.stdout
    document = json.loads(output.read_text(encoding="utf-8"))
    tasks = {task["task_id"]: task for task in document["tasks"]}
    assert tasks["E1-A5-INDEPENDENT-SMOOTH"]["status"] == "WAITING_FOR_APPROVAL"
    assert tasks["E1-A5-INDEPENDENT-SMOOTH"]["launch_approved"] is False
    assert tasks["E7-YUNNAN-EXTERNAL"]["data_split_hash"] == "SEALED_EXTERNAL_NOT_ACCESSED"
    assert tasks["E1-A5-INDEPENDENT-SMOOTH"]["data_split_hash"] == "d" * 64
