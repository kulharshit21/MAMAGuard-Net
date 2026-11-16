#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Short, opt-in CUDA health benchmark; run only on allocated GPUs")
    parser.add_argument("--output", type=Path, default=Path("reports/generated/gpu_benchmark.json"))
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    results = []
    for device_index in range(torch.cuda.device_count()):
        device = torch.device("cuda", device_index)
        torch.cuda.reset_peak_memory_stats(device)
        left = torch.randn((args.size, args.size), device=device, dtype=torch.bfloat16)
        right = torch.randn_like(left)
        for _ in range(2):
            _ = left @ right
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(args.iterations):
            _ = left @ right
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        results.append({
            "logical_device": device_index,
            "name": torch.cuda.get_device_name(device_index),
            "seconds": elapsed,
            "iterations": args.iterations,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

