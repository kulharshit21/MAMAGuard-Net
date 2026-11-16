from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml


def _finite_json(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    return True


def _dataset_directory(root: Path, dataset_id: int) -> Path:
    matches = sorted(root.glob(f"Dataset{dataset_id:03d}_*"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one Dataset{dataset_id:03d}_* directory under {root}; "
            f"found {len(matches)}"
        )
    return matches[0]


def inspect_fold_gate(
    *,
    results_root: Path,
    preprocessed_root: Path,
    config_paths: list[Path],
    fold: int,
    seed: int,
) -> dict[str, Any]:
    if fold not in range(5):
        raise ValueError("fold must be between 0 and 4")

    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in config_paths]
    dataset_ids = {int(config["dataset"]) for config in configs}
    if len(dataset_ids) != 1:
        raise ValueError("All gated experiments must use the same dataset")

    dataset_id = dataset_ids.pop()
    preprocessed_dataset = _dataset_directory(preprocessed_root, dataset_id)
    splits = json.loads((preprocessed_dataset / "splits_final.json").read_text(encoding="utf-8"))
    if fold >= len(splits):
        raise ValueError(f"Fold {fold} is absent from splits_final.json")
    expected_cases = set(splits[fold]["val"])

    experiments: list[dict[str, Any]] = []
    for config_path, config in zip(config_paths, configs, strict=True):
        experiment = str(config["experiment"])
        trainer = str(config["trainer"])
        plans = str(config["plans"])
        configuration = str(config["configuration"])
        dataset_name = preprocessed_dataset.name
        fold_root = (
            results_root
            / experiment
            / f"seed_{seed}"
            / dataset_name
            / f"{trainer}__{plans}__{configuration}"
            / f"fold_{fold}"
        )
        validation_root = fold_root / "validation"
        final_checkpoints = sorted(fold_root.glob("checkpoint_final.*"))
        summary_path = validation_root / "summary.json"
        prediction_cases = {
            path.name.removesuffix(".nii.gz") for path in validation_root.glob("*.nii.gz")
        }
        missing_cases = sorted(expected_cases - prediction_cases)
        unexpected_cases = sorted(prediction_cases - expected_cases)
        issues: list[str] = []

        if not final_checkpoints:
            issues.append("final checkpoint is missing")
        elif any(path.stat().st_size < 1_000_000 for path in final_checkpoints):
            issues.append("final checkpoint is unexpectedly small")

        summary_finite = False
        if not summary_path.is_file():
            issues.append("validation summary.json is missing")
        else:
            try:
                summary_finite = _finite_json(json.loads(summary_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as error:
                issues.append(f"validation summary is unreadable: {error}")
            else:
                if not summary_finite:
                    issues.append("validation summary contains a non-finite number")

        if missing_cases:
            issues.append(f"{len(missing_cases)} validation predictions are missing")
        if unexpected_cases:
            issues.append(f"{len(unexpected_cases)} unexpected validation predictions exist")

        experiments.append(
            {
                "experiment": experiment,
                "config": str(config_path),
                "fold_root": str(fold_root),
                "expected_predictions": len(expected_cases),
                "observed_predictions": len(prediction_cases),
                "missing_cases": missing_cases,
                "unexpected_cases": unexpected_cases,
                "final_checkpoints": [str(path) for path in final_checkpoints],
                "summary": str(summary_path),
                "summary_finite": summary_finite,
                "ready": not issues,
                "issues": issues,
            }
        )

    return {
        "ready": all(item["ready"] for item in experiments),
        "dataset": dataset_name,
        "fold": fold,
        "seed": seed,
        "expected_validation_cases": len(expected_cases),
        "experiments": experiments,
    }
