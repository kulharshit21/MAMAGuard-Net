#!/usr/bin/env python3
"""Safely extract one 7z archive and recursively consume verified nested ZIPs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

if __package__:
    from .profile_dataset_tree import profile
else:
    from profile_dataset_tree import profile


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_members(names: list[str], destination: Path) -> None:
    root = destination.resolve()
    for raw_name in names:
        name = PurePosixPath(raw_name.replace("\\", "/"))
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Unsafe archive member path: {raw_name!r}")
        target = (root / Path(*name.parts)).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Archive member escapes extraction root: {raw_name!r}")


def extracted_members_complete(output: Path, members: list[tuple[str, int | None, bool]]) -> bool:
    files_checked = 0
    for raw_name, expected_size, is_directory in members:
        relative = PurePosixPath(raw_name.replace("\\", "/"))
        target = output / Path(*relative.parts)
        if is_directory:
            if not target.is_dir():
                return False
            continue
        files_checked += 1
        if not target.is_file():
            return False
        if expected_size is not None and target.stat().st_size != expected_size:
            return False
    return files_checked > 0


def member_is_directory(item: object) -> bool:
    value = getattr(item, "is_directory", False)
    return bool(value() if callable(value) else value)


def extract_outer(archive_path: Path, output: Path) -> None:
    try:
        import py7zr
    except ImportError as error:
        raise SystemExit(
            "Python fallback requires py7zr. Install it in a venv with: "
            "python3 -m pip install 'py7zr>=1,<2'"
        ) from error

    marker = output / ".mamaguard_outer_extracted.ok"
    if marker.exists():
        print("Outer archive already marked as successfully extracted.", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    print(f"Testing outer archive: {archive_path}", flush=True)
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        listing = archive.list()
        names = [item.filename for item in listing]
        members = [
            (
                item.filename,
                getattr(item, "uncompressed", None),
                member_is_directory(item),
            )
            for item in listing
        ]
        validate_members(names, output)
        bad_member = archive.test()
        if bad_member:
            raise RuntimeError(f"Outer archive integrity test failed at: {bad_member}")
    if extracted_members_complete(output, members):
        print("Existing outer extraction matches the archive member list; resuming.", flush=True)
        marker.write_text(
            f"{archive_path}\nsize={archive_path.stat().st_size}\n",
            encoding="utf-8",
        )
        return
    if any(output.iterdir()):
        raise RuntimeError(
            "Outer extraction is incomplete and the output directory is non-empty. "
            "Preserving it for manual review instead of overwriting files."
        )
    print(f"Extracting outer archive into: {output}", flush=True)
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(path=output)
    if not extracted_members_complete(output, members):
        raise RuntimeError("Outer extraction finished but member verification failed")
    marker.write_text(
        f"{archive_path}\nsize={archive_path.stat().st_size}\n",
        encoding="utf-8",
    )


def extract_zip(zip_path: Path) -> None:
    digest = sha256_file(zip_path)
    destination = zip_path.with_suffix("")
    marker = destination / f".mamaguard_{digest}.ok"
    print(f"Testing nested archive: {zip_path}", flush=True)
    with zipfile.ZipFile(zip_path) as archive:
        validate_members(archive.namelist(), destination)
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"Nested ZIP integrity test failed at {bad_member}: {zip_path}")
        destination.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            print(f"Extracting into: {destination}", flush=True)
            archive.extractall(destination)
    if not any(path.is_file() for path in destination.rglob("*")):
        raise RuntimeError(f"Extraction produced no files; preserving archive: {zip_path}")
    marker.write_text(f"{digest}\n", encoding="utf-8")
    zip_path.unlink()


def consume_nested_zips(output: Path) -> int:
    count = 0
    while True:
        candidates = sorted(path for path in output.rglob("*") if path.is_file() and path.suffix.lower() == ".zip")
        if not candidates:
            return count
        extract_zip(candidates[0])
        count += 1


def write_reports(data_root: Path, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    result = profile(data_root, sample_limit=20)
    (report_dir / "dataset_tree_after_extraction.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    directories = sorted(str(path) for path in data_root.rglob("*") if path.is_dir() and len(path.relative_to(data_root).parts) <= 4)
    (report_dir / "dataset_directories_depth4.txt").write_text("\n".join(directories) + "\n", encoding="utf-8")
    remaining = sorted(
        f"{path}\t{path.stat().st_size} bytes"
        for path in data_root.rglob("*")
        if path.is_file() and (path.suffix.lower() in {".zip", ".part", ".partial"})
    )
    (report_dir / "remaining_archives_or_partials.txt").write_text("\n".join(remaining) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=int, default=100)
    args = parser.parse_args()

    if not args.archive.is_file():
        raise SystemExit(f"Archive not found: {args.archive}")
    free = shutil.disk_usage(args.data_root).free
    required = args.minimum_free_gib * 1024**3
    if free < required:
        raise SystemExit(
            f"Extraction requires {args.minimum_free_gib} GiB free; "
            f"only {free / 1024**3:.1f} GiB is available"
        )
    extract_outer(args.archive, args.output)
    count = consume_nested_zips(args.output)
    print(f"Successfully extracted and removed {count} nested ZIP archives.", flush=True)
    write_reports(args.data_root, args.report_dir)
    print(f"Audit files written under: {args.report_dir}", flush=True)
    print(f"Original outer archive retained at: {args.archive}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
