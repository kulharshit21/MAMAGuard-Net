"""Audit mirrored primary folds and compute paired out-of-fold Dice on CPU."""
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

from mamaguardnet.orchestration.fold_gate import inspect_fold_gate


root = Path("DGX_MIRROR")
manifest = root / "repository/manifests"
split_path = manifest / "splits_final.json"
splits = json.loads(split_path.read_text())
metadata = {
    row["case_id"]: (row["patient_id"] or row["case_id"], row["source"])
    for row in csv.DictReader((manifest / "mamamia_cases.csv").open(newline=""))
}
configs = [Path("configs/experiments") / name for name in (
    "a0_no_augmentation.yaml", "a1_nnunet_standard.yaml", "a2_independent.yaml",
    "a3_shared.yaml", "a4_kinetics.yaml")]
dataset = "Dataset501_MAMAMIA3Phase"
results = root / "nnUNet_results"
gates = []
hashes = {}
scores = {}
with tempfile.TemporaryDirectory() as tmp:
    split_dir = Path(tmp) / dataset
    split_dir.mkdir()
    (split_dir / split_path.name).write_bytes(split_path.read_bytes())
    for fold, split in enumerate(splits):
        gate = inspect_fold_gate(results_root=results, preprocessed_root=Path(tmp),
                                 config_paths=configs, fold=fold, seed=20261031)
        gates.append(gate)
        for item in gate["experiments"]:
            arm = item["experiment"][:2]
            item["final_checkpoints"] = [
                name for name in item["final_checkpoints"]
                if Path(name).name == "checkpoint_final.pth"
            ]
            if not item["final_checkpoints"] and not any(
                "checkpoint" in issue for issue in item["issues"]
            ):
                item["issues"].append("final checkpoint is missing (partial files ignored)")
                item["ready"] = False
            summary_path = Path(item["summary"])
            if not summary_path.is_file() and (arm, fold) == ("A1", 4):
                remote = "gdrive:MAMAGuard-Net_Continuous_Backup_2026-11-01/"
                remote += summary_path.relative_to(root).as_posix()
                summary = json.loads(subprocess.check_output(
                    ["rclone", "cat", remote, "--retries", "1"], timeout=40))
            else:
                summary = json.loads(summary_path.read_text())
            rows = summary["metric_per_case"]
            values = {Path(row["prediction_file"]).name.removesuffix(".nii.gz"):
                      row["metrics"]["1"]["Dice"] for row in rows}
            assert len(values) == len(rows) and set(values) == set(split["val"]), (arm, fold)
            assert not set(values).intersection(scores.setdefault(arm, {})), (arm, fold)
            scores[arm].update(values)
            if "--hash-checkpoints" in sys.argv:
                for name in item["final_checkpoints"]:
                    digest = hashlib.sha256()
                    with Path(name).open("rb") as stream:
                        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                            digest.update(chunk)
                    hashes[name] = digest.hexdigest()
        gate["ready"] = all(item["ready"] for item in gate["experiments"])

cases = sorted(scores["A1"])
assert all(set(scores[arm]) == set(cases) for arm in scores)
assert all(case in metadata for case in cases)
patients = sorted({metadata[case][0] for case in cases})
patient_index = {patient: i for i, patient in enumerate(patients)}
groups = [patient_index[metadata[case][0]] for case in cases]
counts = np.bincount(groups)
arrays = {arm: np.array([rows[case] for case in cases]) for arm, rows in scores.items()}
assert all(np.all(np.isfinite(a) & (a >= 0) & (a <= 1)) for a in arrays.values())
sources = np.array([metadata[case][1] for case in cases])
rng = np.random.default_rng(20261031)
comparisons = {}
for a, b in (("A4", "A1"), ("A4", "A2"), ("A4", "A3"), ("A3", "A1")):
    difference = arrays[a] - arrays[b]
    sums = np.bincount(groups, weights=difference)
    replicates = []
    for _ in range(20):
        sample = rng.integers(len(patients), size=(500, len(patients)))
        replicates.extend(sums[sample].sum(axis=1) / counts[sample].sum(axis=1))
    comparisons[f"{a}-{b}"] = {
        "mean": float(difference.mean()),
        "paired_patient_bootstrap_95_ci": np.quantile(replicates, [.025, .975]).tolist(),
        "by_source": {source: {"n": int((sources == source).sum()),
                              "mean": float(difference[sources == source].mean())}
                      for source in sorted(set(sources))},
    }
print(json.dumps({
    "fold_gates": gates, "checkpoint_sha256": hashes,
    "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    "cases": len(cases), "patients": len(patients), "bootstrap_replicates": 10000,
    "bootstrap_seed": 20261031,
    "arms": {arm: {"mean": float(a.mean()), "p10": float(np.quantile(a, .1)),
                   "dice_below_0_1": int((a < .1).sum())} for arm, a in arrays.items()},
    "comparisons": comparisons,
}, indent=2))
