from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from mamaguardnet.utils.hashing import sha256_file

PARTIAL_SUFFIXES = (".part", ".partial", ".crdownload", ".download", ".tmp")


@dataclass(frozen=True)
class FileRecord:
    path: str
    size_bytes: int
    modified_ns: int
    status: str
    sha256: str | None = None


def is_partial(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in PARTIAL_SUFFIXES)


def discover(root: Path, *, hash_complete: bool = False) -> list[FileRecord]:
    if not root.exists():
        return []
    records: list[FileRecord] = []
    for directory, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            path = Path(directory) / filename
            stat = path.stat()
            status = "partial" if is_partial(path) else "complete_candidate"
            digest = sha256_file(path) if hash_complete and status != "partial" else None
            records.append(
                FileRecord(
                    path=str(path.resolve()),
                    size_bytes=stat.st_size,
                    modified_ns=stat.st_mtime_ns,
                    status=status,
                    sha256=digest,
                )
            )
    return records


def find_duplicates(records: Iterable[FileRecord]) -> list[list[str]]:
    groups: dict[tuple[int, str], list[str]] = {}
    for record in records:
        if record.sha256 and record.status != "partial":
            groups.setdefault((record.size_bytes, record.sha256), []).append(record.path)
    return [paths for paths in groups.values() if len(paths) > 1]


def inventory(root: Path, *, hash_complete: bool = False) -> dict:
    records = discover(root, hash_complete=hash_complete)
    return {
        "root": str(root.resolve()),
        "exists": root.exists(),
        "read_only_scan": True,
        "files": [asdict(record) for record in records],
        "summary": {
            "count": len(records),
            "bytes": sum(record.size_bytes for record in records),
            "partial_count": sum(record.status == "partial" for record in records),
            "hashed_count": sum(record.sha256 is not None for record in records),
        },
        "duplicates": find_duplicates(records),
    }
