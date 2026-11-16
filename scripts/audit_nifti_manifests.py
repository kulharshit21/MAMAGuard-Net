#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from mamaguardnet.data.cases import CaseRecord, read_case_manifest


def geometry(image) -> tuple[tuple[int, ...], tuple[float, ...]]:
    shape = tuple(int(value) for value in image.shape)
    affine = tuple(float(value) for value in np.asarray(image.affine).ravel())
    return shape, affine


def audit_case(record: CaseRecord, *, inspect_label_values: bool) -> dict:
    import nibabel as nib

    paths = [Path(record.pre), Path(record.post_early), Path(record.post_late)]
    if record.mask:
        paths.append(Path(record.mask))
    if record.breast_mask:
        paths.append(Path(record.breast_mask))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return {"case_id": record.case_id, "errors": [f"missing: {path}" for path in missing]}
    loaded = [nib.load(str(path)) for path in paths]
    reference_shape, reference_affine = geometry(loaded[0])
    errors = []
    for path, image in zip(paths[1:], loaded[1:], strict=True):
        shape, affine = geometry(image)
        if shape != reference_shape:
            errors.append(f"shape mismatch {path}: {shape} != {reference_shape}")
        if not np.allclose(affine, reference_affine, rtol=1e-5, atol=1e-4):
            errors.append(f"affine mismatch: {path}")
    labels = {}
    if inspect_label_values:
        offset = 3
        for name, image in zip(("mask", "breast_mask"), loaded[offset:], strict=False):
            values = np.unique(np.asanyarray(image.dataobj))
            labels[name] = [float(value) for value in values[:32]]
            if values.size > 32:
                errors.append(f"{name} has more than 32 unique values")
            if not np.isfinite(values).all():
                errors.append(f"{name} contains non-finite values")
            if not np.any(values > 0):
                errors.append(f"{name} is empty")
            if not set(values.tolist()).issubset({0, 1}):
                errors.append(f"{name} is not binary: {labels[name]}")
    return {
        "case_id": record.case_id,
        "source": record.source,
        "shape": reference_shape,
        "spacing": tuple(float(value) for value in loaded[0].header.get_zooms()[:3]),
        "label_values": labels,
        "errors": errors,
    }


def audit_case_job(job: tuple[CaseRecord, bool]) -> dict:
    record, inspect_label_values = job
    return audit_case(record, inspect_label_values=inspect_label_values)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate NIfTI geometry and binary masks")
    parser.add_argument("manifest", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-label-values", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    records = [record for path in args.manifest for record in read_case_manifest(path)]
    jobs = [(record, not args.skip_label_values) for record in records]
    results = []
    if args.workers == 1:
        iterator = map(audit_case_job, jobs)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        iterator = executor.map(audit_case_job, jobs, chunksize=1)
    try:
        for completed, result in enumerate(iterator, start=1):
            results.append(result)
            if completed % args.progress_every == 0 or completed == len(records):
                failures_so_far = sum(bool(item["errors"]) for item in results)
                print(
                    f"Audited {completed}/{len(records)} cases; "
                    f"failures so far: {failures_so_far}",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    failures = [result for result in results if result["errors"]]
    summary = {
        "manifests": [str(path.resolve()) for path in args.manifest],
        "case_count": len(results),
        "workers": args.workers,
        "failure_count": len(failures),
        "source_counts": dict(Counter(result.get("source", "unknown") for result in results)),
        "shape_counts": dict(Counter(str(result.get("shape")) for result in results)),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "case_count", "failure_count", "source_counts", "shape_counts",
    )}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
