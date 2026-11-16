from pathlib import Path

import pytest

from scripts.extract_nested_archives import (
    extracted_members_complete,
    member_is_directory,
    validate_members,
)


def test_archive_members_must_stay_under_destination(tmp_path: Path):
    validate_members(["patient/phase/image.dcm"], tmp_path)
    with pytest.raises(ValueError):
        validate_members(["../../outside"], tmp_path)
    with pytest.raises(ValueError):
        validate_members(["/absolute/path"], tmp_path)


def test_existing_outer_extraction_requires_every_member_and_size(tmp_path: Path):
    folder = tmp_path / "dataset"
    folder.mkdir()
    archive = folder / "1.zip"
    archive.write_bytes(b"1234")
    members = [("dataset", None, True), ("dataset/1.zip", 4, False)]
    assert extracted_members_complete(tmp_path, members)
    assert not extracted_members_complete(tmp_path, [("dataset/1.zip", 5, False)])
    assert not extracted_members_complete(tmp_path, [("dataset/2.zip", 4, False)])


def test_py7zr_directory_flag_supports_property_and_method():
    class MethodInfo:
        def is_directory(self):
            return False

    class PropertyInfo:
        is_directory = True

    assert not member_is_directory(MethodInfo())
    assert member_is_directory(PropertyInfo())
