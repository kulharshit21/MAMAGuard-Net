#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

from mamaguardnet.utils.atomic import atomic_write_json
from mamaguardnet.utils.provenance import collect_provenance


def capture(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/generated/environment_audit.json"))
    args = parser.parse_args()
    disk = shutil.disk_usage(Path.cwd())
    report = {
        "read_only": True,
        "provenance": collect_provenance(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
        "commands": [
            capture(["nvidia-smi"]),
            capture(["nvidia-smi", "topo", "-m"]),
            capture(["free", "-b"]),
            capture(["df", "-PT"]),
            capture(["bash", "-lc", "ulimit -a"]),
        ],
    }
    try:
        import torch
        report["torch"] = {
            "version": torch.__version__, "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count(),
        }
    except ImportError:
        report["torch"] = None
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

