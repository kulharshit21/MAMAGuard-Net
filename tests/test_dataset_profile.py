from pathlib import Path

from scripts.profile_dataset_tree import profile


def test_profile_detects_medical_files_archives_and_partials(tmp_path: Path):
    cohort = tmp_path / "cohort_a" / "patient_001"
    cohort.mkdir(parents=True)
    (cohort / "pre.nii.gz").write_bytes(b"image")
    (cohort / "lesion_mask.nii.gz").write_bytes(b"mask")
    (tmp_path / "nested.zip").write_bytes(b"archive")
    (tmp_path / "active.part").write_bytes(b"partial")

    result = profile(tmp_path, sample_limit=5)

    assert result["total_files"] == 4
    assert result["extension_counts"][".nii.gz"] == 2
    assert result["archives"][0]["path"] == "nested.zip"
    assert result["partial_files"] == ["active.part"]
    expected = str(Path("cohort_a") / "patient_001" / "lesion_mask.nii.gz")
    assert result["mask_name_candidates"] == [expected]
