from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from mamaguardnet.cluster.gpu import (
    evaluate_gpus,
    query_compute_processes,
    query_gpus,
    wait_for_gpus,
)
from mamaguardnet.config import load_config, validate_experiment
from mamaguardnet.data.discovery import inventory
from mamaguardnet.nnunet.commands import build_train_command, shell_join
from mamaguardnet.orchestration.registry import RunRegistry
from mamaguardnet.utils.atomic import atomic_write_json
from mamaguardnet.utils.provenance import collect_provenance


def _assert_training_data_ready(dataset: object, plans: str = "nnUNetPlans") -> None:
    """Fail closed until nnU-Net preprocessing and the reviewed split exist."""
    root_value = os.getenv("nnUNet_preprocessed")
    if not root_value:
        raise SystemExit("Set nnUNet_preprocessed to the validated preprocessing directory")
    root = Path(root_value)
    candidates = list(root.glob(f"Dataset{int(dataset):03d}_*"))
    plans_file = f"{plans}.json"
    ready = [path for path in candidates if (path / plans_file).is_file()
             and (path / "splits_final.json").is_file()]
    if len(ready) != 1:
        raise SystemExit(
            f"Expected one preprocessed Dataset{int(dataset):03d} with {plans_file} "
            f"and reviewed splits_final.json under {root}; found {len(ready)}"
        )


def _gpus(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("GPUs must be comma-separated integer IDs") from error
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("GPU IDs must be unique")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mamaguard")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit-data", help="Read-only dataset inventory")
    audit.add_argument("root", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--hash-complete", action="store_true")

    gpu = sub.add_parser("gpu-preflight", help="Check or wait for requested GPUs")
    gpu.add_argument("--gpus", type=_gpus, default=_gpus("1,5,6,7"))
    gpu.add_argument("--min-free-mib", type=int, default=36_000)
    gpu.add_argument("--max-utilization", type=int, default=10)
    gpu.add_argument("--stable-samples", type=int, default=3)
    gpu.add_argument("--interval-seconds", type=int, default=30)
    gpu.add_argument("--timeout-seconds", type=int, default=86_400)
    gpu.add_argument("--wait", action="store_true")
    gpu.add_argument("--output", type=Path)

    train = sub.add_parser("train", help="Plan or execute an nnU-Net experiment")
    train.add_argument("config", type=Path)
    train.add_argument("--registry", type=Path, default=Path("runs/registry.json"))
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--execute", action="store_true")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--evaluate-only", action="store_true")
    train.add_argument("--allow-duplicate", action="store_true")
    train.add_argument("--fold", type=int, choices=range(5))
    train.add_argument("--seed", type=int)

    runs = sub.add_parser("runs", help="Inspect registered runs")
    runs.add_argument("--registry", type=Path, default=Path("runs/registry.json"))
    runs.add_argument("--status", choices=["planned", "running", "completed", "failed", "stopped"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit-data":
        document = inventory(args.root, hash_complete=args.hash_complete)
        atomic_write_json(args.output, document)
        print(json.dumps(document["summary"], indent=2))
        return 0
    if args.command == "gpu-preflight":
        if args.wait:
            report = wait_for_gpus(
                args.gpus, min_free_mib=args.min_free_mib, max_utilization=args.max_utilization,
                stable_samples=args.stable_samples, interval_seconds=args.interval_seconds,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            report = evaluate_gpus(
                query_gpus(), args.gpus, min_free_mib=args.min_free_mib,
                max_utilization=args.max_utilization, processes=query_compute_processes(),
            )
        if args.output:
            atomic_write_json(args.output, report)
        print(json.dumps(report, indent=2))
        return 0 if report["safe"] else 2
    if args.command == "runs":
        runs = RunRegistry(args.registry).list()
        if args.status:
            runs = [run for run in runs if run["status"] == args.status]
        print(json.dumps(runs, indent=2))
        return 0
    if args.command == "train":
        if args.execute == args.dry_run:
            raise SystemExit("Choose exactly one of --dry-run or --execute")
        config = load_config(args.config)
        if args.fold is not None:
            config["fold"] = args.fold
        if args.seed is not None:
            config["seed"] = args.seed
        if os.getenv("MAMAGUARD_GPUS"):
            config["gpus"] = _gpus(os.environ["MAMAGUARD_GPUS"])
            config["num_gpus"] = len(config["gpus"])
        validate_experiment(config)
        command = build_train_command(config, resume=args.resume, validation_only=args.evaluate_only)
        plan = {
            "command": command,
            "shell": shell_join(command),
            "cuda_visible_devices": config.get("gpus", [1, 5, 6, 7]),
            "provenance": collect_provenance(),
        }
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        _assert_training_data_ready(config["dataset"], config.get("plans", "nnUNetPlans"))
        if not os.getenv("SLURM_JOB_ID"):
            gpu_report = evaluate_gpus(
                query_gpus(), plan["cuda_visible_devices"], min_free_mib=36_000, max_utilization=10,
                processes=query_compute_processes(),
            )
            if not gpu_report["safe"]:
                print(json.dumps(gpu_report, indent=2), file=sys.stderr)
                raise SystemExit("Unsafe GPU state. Use cluster/launch_ddp.sh to wait without preemption.")
        registry = RunRegistry(args.registry)
        run = registry.register(config, allow_duplicate=args.allow_duplicate or args.resume or args.evaluate_only)
        environment = os.environ.copy()
        results_root = environment.get("nnUNet_results")
        if not results_root:
            raise SystemExit("Set nnUNet_results before executing training")
        environment["nnUNet_results"] = str(
            Path(results_root) / config["experiment"] / f"seed_{config.get('seed', 20261031)}"
        )
        if not os.getenv("SLURM_JOB_ID"):
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in plan["cuda_visible_devices"])
        print(json.dumps({"run": run, "plan": plan}, indent=2), flush=True)
        augmentation = config.get("augmentation_parameters", {})
        environment.update({
            "MAMAGUARD_AUG_P": str(augmentation.get("probability", 0.5)),
            "MAMAGUARD_GAIN_LOG_STD": str(augmentation.get("gain_log_std", 0.12)),
            "MAMAGUARD_OFFSET_STD": str(augmentation.get("offset_std", 0.05)),
            "MAMAGUARD_FIELD_GRID": str(augmentation.get("field_grid", 4)),
            "MAMAGUARD_SEED": str(config.get("seed", 20261031)),
        })
        registry.update_status(run["run_id"], "running")
        try:
            return_code = subprocess.call(command, env=environment)
        except (OSError, KeyboardInterrupt):
            registry.update_status(run["run_id"], "failed")
            raise
        registry.update_status(run["run_id"], "completed" if return_code == 0 else "failed")
        return return_code
    return 1


if __name__ == "__main__":
    sys.exit(main())
