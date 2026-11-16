#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

INCLUDED_DIRECTORIES = ("src", "scripts", "configs", "cluster", "tests", "research", "registry")
INCLUDED_ROOT_FILES = (
    "pyproject.toml",
    "README.md",
    "PROJECT_STATUS.md",
    "NEXT_ACTIONS.md",
    "DECISIONS.md",
    "EXPERIMENT_LOG.md",
)
INCLUDED_PROJECT_FILES = (
    "reports/A5_MATCHED_CONTROL_SPECIFICATION.md",
    "reports/PUBLICATION_ENGINE_STATUS.md",
)
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".git"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pth", ".nii", ".gz", ".7z", ".zip"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for directory in INCLUDED_DIRECTORIES:
        base = root / directory
        if base.is_dir():
            candidates.extend(path for path in base.rglob("*") if path.is_file())
    candidates.extend(root / name for name in INCLUDED_ROOT_FILES if (root / name).is_file())
    candidates.extend(root / name for name in INCLUDED_PROJECT_FILES if (root / name).is_file())
    return sorted(
        {
            path.resolve()
            for path in candidates
            if not EXCLUDED_PARTS.intersection(path.parts)
            and not any(path.name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)
        },
        key=lambda path: path.relative_to(root.resolve()).as_posix(),
    )


def create_snapshot(root: Path, output_dir: Path) -> tuple[Path, Path]:
    root = root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = snapshot_files(root)
    if not files:
        raise ValueError(f"No source files found under {root}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = output_dir / f"mamaguard-source-{timestamp}.tar.gz"
    manifest_path = output_dir / f"mamaguard-source-{timestamp}.manifest.json"
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    def portable_metadata(info: tarfile.TarInfo) -> tarfile.TarInfo:
        relative = Path(info.name)
        info.mode = 0o755 if relative.parts and relative.parts[0] == "scripts" else 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        return info

    with tarfile.open(archive_path, "w:gz") as archive:
        for path in files:
            archive.add(
                path,
                arcname=path.relative_to(root).as_posix(),
                recursive=False,
                filter=portable_metadata,
            )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(root),
        "python": sys.version,
        "platform": platform.platform(),
        "file_count": len(entries),
        "archive": archive_path.name,
        "archive_sha256": sha256(archive_path),
        "files": entries,
        "exclusions": "datasets, checkpoints, generated results, caches, archives, and credentials",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return archive_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a credential- and data-free source snapshot")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    archive, manifest = create_snapshot(args.root, args.output_dir)
    print(json.dumps({"archive": str(archive), "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

