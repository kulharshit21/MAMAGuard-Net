#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from mamaguardnet.evaluation.geometry import ImageGeometry, assert_same_geometry
from mamaguardnet.temporal_analysis import (
    audit_kinetics_domains,
    sample_a4_smooth_affine,
    zscore_per_phase,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit original, independently normalized, and shared-affine DCE domains"
    )
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--early", type=Path, required=True)
    parser.add_argument("--late", type=Path, required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--gain-log-std", type=float, default=0.12)
    parser.add_argument("--offset-std", type=float, default=0.05)
    parser.add_argument("--field-grid", type=int, default=4)
    args = parser.parse_args()

    images = [nib.load(path) for path in (args.pre, args.early, args.late)]
    geometries = [ImageGeometry(image.shape, image.affine) for image in images]
    for geometry in geometries[1:]:
        assert_same_geometry(geometries[0], geometry)
    phases = np.stack([np.asarray(image.dataobj, dtype=np.float32) for image in images])
    selected = None
    if args.mask:
        mask_image = nib.load(args.mask)
        assert_same_geometry(geometries[0], ImageGeometry(mask_image.shape, mask_image.affine))
        selected = np.asarray(mask_image.dataobj) > 0

    normalized = zscore_per_phase(phases)
    augmentation = sample_a4_smooth_affine(
        normalized,
        rng=np.random.default_rng(args.seed),
        gain_log_std=args.gain_log_std,
        offset_std=args.offset_std,
        grid=args.field_grid,
    )
    report = audit_kinetics_domains(
        phases,
        normalized,
        augmentation.image,
        mask=selected,
        epsilon=args.epsilon,
    )
    report["inputs"] = {
        "pre": str(args.pre.resolve()),
        "early": str(args.early.resolve()),
        "late": str(args.late.resolve()),
        "mask": str(args.mask.resolve()) if args.mask else None,
        "seed": args.seed,
    }
    report["augmentation"] = {
        "kind": "A4_shared_smooth_positive_affine_field",
        "gain_log_std": args.gain_log_std,
        "offset_std": args.offset_std,
        "field_grid": args.field_grid,
        "gain_min": float(augmentation.gain.min()),
        "gain_max": float(augmentation.gain.max()),
        "gain_mean": float(augmentation.gain.mean()),
        "gain_std": float(augmentation.gain.std()),
        "offset_min": float(augmentation.offset.min()),
        "offset_max": float(augmentation.offset.max()),
        "offset_mean": float(augmentation.offset.mean()),
        "offset_std_observed": float(augmentation.offset.std()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

