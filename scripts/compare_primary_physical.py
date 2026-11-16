"""Pair DGX physical evaluation outputs by development patient."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as stream:
        metadata = {row["case_id"]: row for row in csv.DictReader(stream)}

    arms = {}
    expected_folds = {}
    for fold in range(5):
        gate = json.loads((args.output_dir / f"fold_{fold}.json").read_text())
        if not gate["ready"] or len(gate["experiments"]) != 5:
            raise ValueError(f"Fold {fold} did not pass all five gates")
        for item in gate["experiments"]:
            experiment = item["experiment"]
            path = args.output_dir / f"{experiment}_fold{fold}.csv"
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            cases = {row["case_id"] for row in rows}
            if len(cases) != len(rows) or len(rows) != item["expected_predictions"]:
                raise ValueError(f"Unexpected case count or duplicate cases: {path}")
            if expected_folds.setdefault(fold, cases) != cases:
                raise ValueError(f"Mismatched case IDs: {path}")
            arm = arms.setdefault(experiment[:2], {})
            if cases.intersection(arm):
                raise ValueError(f"Repeated case across folds: {path}")
            arm.update((row["case_id"], row) for row in rows)

    cases = sorted(arms["A1"])
    patients = [metadata[case]["patient_id"] or case for case in cases]
    if len(set(patients)) != len(cases):
        raise ValueError("Repeated patient requires a clustered bootstrap")
    endpoints = ("dice", "surface_dice_2mm", "hd95_mm",
                 "absolute_volume_error_mm3", "severe_failure")
    source = np.array([metadata[case]["source"] for case in cases])
    rng = np.random.default_rng(20261031)
    metrics = {
        arm: {
            key: np.array([
                float(rows[case][key] == "True") if key == "severe_failure"
                else float(rows[case][key]) for case in cases
            ])
            for key in endpoints
        }
        for arm, rows in arms.items()
    }

    report = {"patients": len(cases), "folds": 5, "bootstrap_replicates": 10000,
              "bootstrap_seed": 20261031, "arms": {}, "paired": {}}
    for arm, values in metrics.items():
        report["arms"][arm] = {}
        for key, vector in values.items():
            valid = np.isfinite(vector)
            finite = vector[valid]
            report["arms"][arm][key] = {
                "finite_n": int(valid.sum()), "nonfinite_n": int((~valid).sum()),
                "finite_mean": float(finite.mean()) if finite.size else None,
                "finite_median": float(np.median(finite)) if finite.size else None,
            }
    for a, b in (("A4", "A1"), ("A4", "A2"), ("A4", "A3"), ("A3", "A1")):
        contrast = report["paired"][f"{a}-{b}"] = {}
        for key in endpoints:
            x, y = metrics[a][key], metrics[b][key]
            valid_x, valid_y = np.isfinite(x), np.isfinite(y)
            both = valid_x & valid_y
            diff = x[both] - y[both]
            draws = []
            if diff.size:
                for _ in range(20):
                    index = rng.integers(diff.size, size=(500, diff.size))
                    draws.extend(diff[index].mean(axis=1))
            contrast[key] = {
                "finite_pairs": int(both.sum()),
                "a_only_nonfinite": int((~valid_x & valid_y).sum()),
                "b_only_nonfinite": int((valid_x & ~valid_y).sum()),
                "both_nonfinite": int((~valid_x & ~valid_y).sum()),
                "mean_difference": float(diff.mean()) if diff.size else None,
                "unadjusted_95_ci": np.quantile(draws, [.025, .975]).tolist() if draws else None,
                "source_exploratory": {
                    name: {"n": int(selected.sum()),
                           "mean_difference": float((x[selected] - y[selected]).mean())}
                    for name in sorted(set(source))
                    if (selected := both & (source == name)).any()
                },
            }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
