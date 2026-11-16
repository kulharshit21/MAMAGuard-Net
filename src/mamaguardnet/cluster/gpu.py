from __future__ import annotations

import csv
import io
import subprocess
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GPUState:
    index: int
    name: str
    uuid: str
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int
    utilization_percent: int


def query_compute_processes() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    processes = []
    for row in csv.reader(io.StringIO(result.stdout.strip()), skipinitialspace=True):
        if row:
            processes.append({
                "gpu_uuid": row[0].strip(),
                "pid": int(row[1].strip()),
                "used_memory_mib": int(row[2].strip()),
                "process_name": row[3].strip(),
            })
    return processes


def parse_gpu_csv(text: str) -> list[GPUState]:
    states: list[GPUState] = []
    for row in csv.reader(io.StringIO(text.strip()), skipinitialspace=True):
        if not row:
            continue
        values = [item.strip() for item in row]
        if len(values) != 7:
            raise ValueError(f"Expected 7 nvidia-smi columns, received {len(values)}: {values}")
        states.append(GPUState(int(values[0]), values[1], values[2], *(int(value) for value in values[3:])))
    return states


def query_gpus() -> list[GPUState]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return parse_gpu_csv(result.stdout)


def evaluate_gpus(
    states: list[GPUState], requested: list[int], *, min_free_mib: int, max_utilization: int,
    processes: list[dict] | None = None,
) -> dict:
    by_index = {state.index: state for state in states}
    missing = [index for index in requested if index not in by_index]
    selected = [by_index[index] for index in requested if index in by_index]
    busy_uuids = {process["gpu_uuid"] for process in (processes or [])}
    unsafe = [
        state for state in selected
        if state.memory_free_mib < min_free_mib or state.utilization_percent > max_utilization
        or state.uuid in busy_uuids
    ]
    return {
        "safe": not missing and not unsafe,
        "requested": requested,
        "missing": missing,
        "unsafe": [asdict(state) for state in unsafe],
        "compute_processes": [process for process in (processes or []) if process["gpu_uuid"] in {state.uuid for state in selected}],
        "selected": [asdict(state) for state in selected],
        "thresholds": {"min_free_mib": min_free_mib, "max_utilization_percent": max_utilization},
    }


def wait_for_gpus(
    requested: list[int], *, min_free_mib: int = 36_000, max_utilization: int = 10,
    stable_samples: int = 3, interval_seconds: int = 30, timeout_seconds: int = 86_400,
) -> dict:
    deadline, consecutive = time.monotonic() + timeout_seconds, 0
    last: dict = {}
    while time.monotonic() < deadline:
        last = evaluate_gpus(
            query_gpus(), requested, min_free_mib=min_free_mib, max_utilization=max_utilization,
            processes=query_compute_processes(),
        )
        consecutive = consecutive + 1 if last["safe"] else 0
        if consecutive >= stable_samples:
            last["stable_samples"] = consecutive
            return last
        time.sleep(interval_seconds)
    raise TimeoutError(f"GPUs did not become safely available before timeout; last state: {last}")

