from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


def _command(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_provenance() -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_command(["git", "status", "--porcelain"])),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": _command(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv,noheader"]),
    }
