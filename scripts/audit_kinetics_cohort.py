#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

from mamaguardnet.evaluation.geometry import ImageGeometry, assert_same_geometry
from mamaguardnet.temporal_analysis import (
    audit_kinetics_domains,
    sample_a4_smooth_affine,
    zscore_per_phase,
)


def case_seed(global_seed: int, case_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{case_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def select_records(
    records: list[dict[str, str]],
    *,
    per_source: int,
    seed: int,
) -> list[dict[str, str]]:
    if per_source < 1:
        raise ValueError("per_source must be at least 1")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        if record.get("dataset_role") != "development":
            continue
        source = record["source"]
        if source.upper() == "YUNNAN":
            raise ValueError("Yunnan must not enter a development kinetics audit")
        groups[source].append(record)
    selected = []
    for source, source_records in sorted(groups.items()):
        ordered = sorted(source_records, key=lambda item: item["case_id"])
        rng = random.Random(case_seed(seed, source))
        selected.extend(rng.sample(ordered, min(per_source, len(ordered))))
    return sorted(selected, key=lambda item: (item["source"], item["case_id"]))


def audit_record(job: tuple[dict[str, str], int, float, float, int, float]) -> dict[str, object]:
    record, seed, gain_log_std, offset_std, field_grid, epsilon = job
    started = time.perf_counter()
    images = [nib.load(record[key]) for key in ("pre", "post_early", "post_late")]
    reference = ImageGeometry(images[0].shape, images[0].affine)
    for image in images[1:]:
        assert_same_geometry(reference, ImageGeometry(image.shape, image.affine))
    mask_image = nib.load(record["mask"])
    assert_same_geometry(reference, ImageGeometry(mask_image.shape, mask_image.affine))
    phases = np.stack([np.asarray(image.dataobj, dtype=np.float32) for image in images])
    mask = np.asarray(mask_image.dataobj) > 0
    normalized = zscore_per_phase(phases)
    augmentation = sample_a4_smooth_affine(
        normalized,
        rng=np.random.default_rng(seed),
        gain_log_std=gain_log_std,
        offset_std=offset_std,
        grid=field_grid,
    )
    report = audit_kinetics_domains(
        phases,
        normalized,
        augmentation.image,
        mask=mask,
        epsilon=epsilon,
    )
    original, augmented = report["original_to_normalized"], report["normalized_to_augmented"]
    return {
        "source": record["source"],
        "case_id": record["case_id"],
        "seed": seed,
        "selected_voxels": report["selected_voxels"],
        "original_sign_eligible": original["eligible_voxels"],
        "original_sign_violations": original["sign_violations"],
        "original_ordering_eligible": original["ordering_eligible_voxels"],
        "original_ordering_violations": original["ordering_violations"],
        "a4_sign_eligible": augmented["eligible_voxels"],
        "a4_sign_violations": augmented["sign_violations"],
        "a4_ordering_eligible": augmented["ordering_eligible_voxels"],
        "a4_ordering_violations": augmented["ordering_violations"],
        "a4_valid": augmented["valid"],
        "gain_min": float(augmentation.gain.min()),
        "gain_max": float(augmentation.gain.max()),
        "gain_mean": float(augmentation.gain.mean()),
        "gain_std": float(augmentation.gain.std()),
        "offset_mean": float(augmentation.offset.mean()),
        "offset_std_observed": float(augmentation.offset.std()),
        "elapsed_seconds": time.perf_counter() - started,
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    def one(group: list[dict[str, object]]) -> dict[str, object]:
        totals = {
            key: sum(int(row[key]) for row in group)
            for key in (
                "selected_voxels",
                "original_sign_eligible",
                "original_sign_violations",
                "original_ordering_eligible",
                "original_ordering_violations",
                "a4_sign_eligible",
                "a4_sign_violations",
                "a4_ordering_eligible",
                "a4_ordering_violations",
            )
        }
        totals["case_count"] = len(group)
        totals["original_sign_violation_rate"] = totals["original_sign_violations"] / max(
            totals["original_sign_eligible"], 1
        )
        totals["original_ordering_violation_rate"] = totals[
            "original_ordering_violations"
        ] / max(totals["original_ordering_eligible"], 1)
        totals["a4_sign_violation_rate"] = totals["a4_sign_violations"] / max(
            totals["a4_sign_eligible"], 1
        )
        totals["a4_ordering_violation_rate"] = totals["a4_ordering_violations"] / max(
            totals["a4_ordering_eligible"], 1
        )
        totals["mean_case_seconds"] = float(np.mean([row["elapsed_seconds"] for row in group]))
        return totals

    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(row)
    return {
        "overall": one(rows),
        "by_source": {source: one(group) for source, group in sorted(by_source.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stratified MAMA-MIA A4 kinetics audit")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--gain-log-std", type=float, default=0.12)
    parser.add_argument("--offset-std", type=float, default=0.05)
    parser.add_argument("--field-grid", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1 or args.progress_every < 1:
        raise SystemExit("workers and progress-every must be at least 1")

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    selected = select_records(records, per_source=args.per_source, seed=args.seed)
    jobs = [
        (
            record,
            case_seed(args.seed, record["case_id"]),
            args.gain_log_std,
            args.offset_std,
            args.field_grid,
            args.epsilon,
        )
        for record in selected
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for completed, row in enumerate(executor.map(audit_record, jobs, chunksize=1), start=1):
            rows.append(row)
            if completed % args.progress_every == 0 or completed == len(jobs):
                print(f"Audited {completed}/{len(jobs)} cases", flush=True)

    if not all(bool(row["a4_valid"]) for row in rows):
        raise RuntimeError("A4 invariant violation detected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "kinetic_constraint_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "manifest": str(args.manifest.resolve()),
        "seed": args.seed,
        "per_source_requested": args.per_source,
        "workers": args.workers,
        "source_counts": dict(Counter(str(row["source"]) for row in rows)),
        "parameters": {
            "gain_log_std": args.gain_log_std,
            "offset_std": args.offset_std,
            "field_grid": args.field_grid,
            "epsilon": args.epsilon,
        },
        **aggregate(rows),
    }
    summary_path = args.output_dir / "kinetic_constraint_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

