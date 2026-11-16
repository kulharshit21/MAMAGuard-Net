from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from mamaguardnet.data.cases import CaseRecord


def _place(source: Path, destination: Path, mode: str) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"Refusing to overwrite existing staged file: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unsupported staging mode: {mode}")


def stage_mamamia_raw(
    records: list[CaseRecord], raw_root: Path, *, dataset_id: int = 501,
    dataset_name: str = "MAMAMIA3Phase", mode: str = "symlink",
) -> Path:
    development = [record for record in records if record.dataset_role == "development"]
    if not development:
        raise ValueError("No development cases are eligible for nnU-Net staging")
    if any(record.source == "YUNNAN" for record in development):
        raise ValueError("External Yunnan cases cannot be staged as development data")
    dataset = raw_root / f"Dataset{dataset_id:03d}_{dataset_name}"
    images = dataset / "imagesTr"
    labels = dataset / "labelsTr"
    for record in development:
        if not record.mask:
            raise ValueError(f"Expert mask is required for training: {record.case_id}")
        for channel, value in enumerate((record.pre, record.post_early, record.post_late)):
            source = Path(value)
            if not source.is_file():
                raise FileNotFoundError(source)
            _place(source, images / f"{record.case_id}_{channel:04d}.nii.gz", mode)
        mask = Path(record.mask)
        if not mask.is_file():
            raise FileNotFoundError(mask)
        _place(mask, labels / f"{record.case_id}.nii.gz", mode)
    metadata = {
        "channel_names": {"0": "pre", "1": "early_post", "2": "later_post"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(development),
        "file_ending": ".nii.gz",
        "description": "MAMA-MIA expert labels; fixed common phases 0000, 0001, 0002",
    }
    dataset_json = dataset / "dataset.json"
    rendered = json.dumps(metadata, indent=2) + "\n"
    if dataset_json.exists() and dataset_json.read_text(encoding="utf-8") != rendered:
        raise FileExistsError(f"Refusing to replace different metadata: {dataset_json}")
    dataset.mkdir(parents=True, exist_ok=True)
    dataset_json.write_text(rendered, encoding="utf-8")
    return dataset
