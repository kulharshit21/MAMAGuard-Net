#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

from mamaguardnet.evaluation.geometry import (
    GeometryMismatchError,
    ImageGeometry,
    assert_same_geometry,
    resample_label_to_geometry,
)
from mamaguardnet.evaluation.metrics import evaluate_case
from mamaguardnet.evaluation.reporting import summarize_rows


def evaluate_job(
    job: tuple[Path, Path, float, bool],
) -> dict[str, float | int | bool | str]:
    prediction_path, target_path, threshold, allow_resampling = job
    started = time.perf_counter()
    prediction_image, target_image = nib.load(prediction_path), nib.load(target_path)
    prediction_data = np.asarray(prediction_image.dataobj)
    target_data = np.asarray(target_image.dataobj)
    if not np.isfinite(prediction_data).all() or not np.isfinite(target_data).all():
        raise ValueError(f"Non-finite voxel value detected for {prediction_path.name}")
    prediction_geometry = ImageGeometry(prediction_data.shape, prediction_image.affine)
    target_geometry = ImageGeometry(target_data.shape, target_image.affine)
    prediction = prediction_data >= threshold
    try:
        assert_same_geometry(prediction_geometry, target_geometry)
    except GeometryMismatchError:
        if not allow_resampling:
            raise
        prediction = resample_label_to_geometry(
            prediction,
            prediction_geometry,
            target_geometry,
        )
    target = target_data > 0
    metrics = evaluate_case(prediction, target, target_geometry.spacing)
    return {
        "case_id": prediction_path.name.removesuffix(".nii.gz"),
        **metrics,
        "evaluation_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate matching NIfTI predictions at patient level")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--resample-prediction",
        action="store_true",
        help="Explicitly resample predictions to target geometry with nearest-neighbor interpolation",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be at least 1")
    paths = sorted(args.predictions.glob("*.nii.gz"))
    if not paths:
        raise SystemExit("No .nii.gz predictions found")
    jobs = []
    for prediction_path in paths:
        target_path = args.targets / prediction_path.name
        if not target_path.exists():
            raise FileNotFoundError(target_path)
        jobs.append((prediction_path, target_path, args.threshold, args.resample_prediction))

    if args.workers == 1:
        iterator = map(evaluate_job, jobs)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        iterator = executor.map(evaluate_job, jobs, chunksize=1)
    rows = []
    try:
        for completed, row in enumerate(iterator, start=1):
            rows.append(row)
            if completed % args.progress_every == 0 or completed == len(jobs):
                print(f"Evaluated {completed}/{len(jobs)} cases", file=sys.stderr, flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_rows(rows)
    args.output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

