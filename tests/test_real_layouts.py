import json
from pathlib import Path

from mamaguardnet.data.cases import read_case_manifest, write_case_manifest
from mamaguardnet.data.layouts import discover_mamamia, discover_yunnan
from mamaguardnet.nnunet.staging import stage_mamamia_raw


def touch(path: Path, data: bytes = b"nifti") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_verified_layouts_build_manifests_and_stage_only_development(tmp_path: Path):
    case_id = "DUKE_001"
    case_dir = tmp_path / "images" / case_id
    for index in range(4):
        touch(case_dir / f"{case_id}_{index:04d}.nii.gz")
    touch(tmp_path / "segmentations" / "expert" / f"{case_id}.nii.gz", b"mask")
    info = tmp_path / "patient_info_files" / f"{case_id}.json"
    info.parent.mkdir(parents=True)
    info.write_text(json.dumps({"imaging_data": {"dataset": "DUKE"}}))

    secondary = tmp_path / "Breast_Cancer_DCE-MRI_Data" / "Breast_Cancer_DCE-MRI_Data" / "64" / "64"
    for phase in (0, 1, 2, 3, 4, 5, 8):
        touch(secondary / f"P{phase}.nii.gz")
    touch(secondary / "GT.nii.gz", b"gt")
    touch(secondary / "Breast_mask.nii.gz", b"breast")

    mamamia, development_issues = discover_mamamia(tmp_path)
    yunnan, external_issues = discover_yunnan(tmp_path)
    assert not development_issues and not external_issues
    assert mamamia[0].available_phases[-1].endswith("_0003.nii.gz")
    assert [Path(value).name for value in yunnan[0].available_phases] == [
        "P0.nii.gz", "P1.nii.gz", "P2.nii.gz", "P3.nii.gz", "P4.nii.gz",
        "P5.nii.gz", "P8.nii.gz",
    ]
    assert yunnan[0].dataset_role == "locked_external_test"

    manifest = tmp_path / "manifests" / "mamamia.csv"
    write_case_manifest(manifest, mamamia)
    restored = read_case_manifest(manifest)
    assert restored == mamamia

    staged = stage_mamamia_raw(restored + yunnan, tmp_path / "nnUNet_raw", mode="hardlink")
    assert (staged / "imagesTr" / f"{case_id}_0002.nii.gz").is_file()
    assert not any(staged.rglob("YUNNAN*"))
    dataset = json.loads((staged / "dataset.json").read_text())
    assert dataset["numTraining"] == 1
    assert dataset["channel_names"] == {"0": "pre", "1": "early_post", "2": "later_post"}
