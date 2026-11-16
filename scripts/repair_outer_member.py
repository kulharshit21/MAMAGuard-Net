#!/usr/bin/env python3
"""Recover one incomplete member from a verified 7z archive."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path, PurePosixPath


def safe_target(output: Path, member: str) -> Path:
    relative = PurePosixPath(member.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive member: {member!r}")
    root = output.resolve()
    target = (root / Path(*relative.parts)).resolve()
    if root not in target.parents:
        raise ValueError(f"Member escapes output directory: {member!r}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    args = parser.parse_args()

    try:
        import py7zr
    except ImportError as error:
        raise SystemExit("py7zr is required in the active Python environment") from error

    target = safe_target(args.output, args.member)
    backup = target.with_name(target.name + ".incomplete")
    if backup.exists():
        raise SystemExit(f"Quarantine path already exists; inspect it first: {backup}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.replace(backup)

    try:
        print(f"Re-extracting {args.member}", flush=True)
        with py7zr.SevenZipFile(args.archive, mode="r") as archive:
            if args.member not in archive.getnames():
                raise RuntimeError(f"Member is absent from archive: {args.member}")
            archive.extract(path=args.output, targets=[args.member])
        actual_size = target.stat().st_size
        if actual_size != args.expected_size:
            raise RuntimeError(
                f"Recovered size {actual_size} does not match expected {args.expected_size}"
            )
        if target.suffix.lower() == ".zip":
            with zipfile.ZipFile(target) as archive:
                bad_member = archive.testzip()
                if bad_member:
                    raise RuntimeError(f"Recovered ZIP failed CRC at: {bad_member}")
    except Exception:
        if target.exists():
            target.unlink()
        if backup.exists():
            backup.replace(target)
        raise

    if backup.exists():
        backup.unlink()
    print(f"Recovered and verified: {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
