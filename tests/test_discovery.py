from pathlib import Path

from mamaguardnet.data.discovery import discover, find_duplicates


def test_partial_files_are_never_hashed(tmp_path: Path):
    complete = tmp_path / "case.zip"
    partial = tmp_path / "case2.zip.part"
    complete.write_bytes(b"abc")
    partial.write_bytes(b"abc")
    records = discover(tmp_path, hash_complete=True)
    by_name = {Path(record.path).name: record for record in records}
    assert by_name["case2.zip.part"].status == "partial"
    assert by_name["case2.zip.part"].sha256 is None
    assert by_name["case.zip"].sha256


def test_duplicates_require_content_hash(tmp_path: Path):
    (tmp_path / "a").write_bytes(b"same")
    (tmp_path / "b").write_bytes(b"same")
    assert len(find_duplicates(discover(tmp_path, hash_complete=True))) == 1

