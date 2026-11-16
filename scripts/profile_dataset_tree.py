#!/usr/bin/env python3
"""Read-only structural profile for large medical-imaging dataset trees."""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PARTIAL_SUFFIXES = (".part", ".partial", ".crdownload", ".download", ".tmp")
ARCHIVE_SUFFIXES = (".zip", ".7z", ".tar", ".tar.gz", ".tgz")
MASK_PATTERN = re.compile(r"mask|seg|label|lesion|tumou?r", re.IGNORECASE)
PHASE_PATTERN = re.compile(r"(^|[_\-.])(pre|post|early|late|phase|dce)([_\-.]|$)", re.IGNORECASE)


def logical_suffix(name: str) -> str:
    lower = name.lower()
    for suffix in (".nii.gz", ".tar.gz"):
        if lower.endswith(suffix):
            return suffix
    return Path(lower).suffix or "[no_extension]"


def profile(root: Path, sample_limit: int) -> dict:
    extensions: Counter[str] = Counter()
    extension_bytes: Counter[str] = Counter()
    top_level: Counter[str] = Counter()
    depths: Counter[int] = Counter()
    samples: dict[str, list[str]] = defaultdict(list)
    archives: list[dict] = []
    partials: list[str] = []
    mask_candidates: list[str] = []
    phase_candidates: list[str] = []
    total_files = total_bytes = 0

    if not root.exists():
        return {"root": str(root), "exists": False}

    for directory, _, filenames in os.walk(root):
        base = Path(directory)
        for filename in filenames:
            path = base / filename
            try:
                stat = path.stat()
            except OSError as error:
                samples["[stat_error]"].append(f"{path}: {error}")
                continue
            relative = path.relative_to(root)
            suffix = logical_suffix(filename)
            total_files += 1
            total_bytes += stat.st_size
            extensions[suffix] += 1
            extension_bytes[suffix] += stat.st_size
            top_level[relative.parts[0]] += 1
            depths[len(relative.parts)] += 1
            if len(samples[suffix]) < sample_limit:
                samples[suffix].append(str(relative))
            lower = filename.lower()
            if lower.endswith(PARTIAL_SUFFIXES):
                partials.append(str(relative))
            if lower.endswith(ARCHIVE_SUFFIXES):
                archives.append({"path": str(relative), "size_bytes": stat.st_size})
            if MASK_PATTERN.search(filename) and len(mask_candidates) < 100:
                mask_candidates.append(str(relative))
            if PHASE_PATTERN.search(filename) and len(phase_candidates) < 100:
                phase_candidates.append(str(relative))

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "exists": True,
        "read_only": True,
        "total_files": total_files,
        "total_bytes": total_bytes,
        "top_level_file_counts": dict(top_level.most_common()),
        "extension_counts": dict(extensions.most_common()),
        "extension_bytes": dict(extension_bytes.most_common()),
        "depth_counts": {str(key): value for key, value in sorted(depths.items())},
        "archives": archives,
        "partial_files": partials,
        "mask_name_candidates": mask_candidates,
        "phase_name_candidates": phase_candidates,
        "samples_by_extension": dict(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=20)
    args = parser.parse_args()
    result = profile(args.root.expanduser(), args.sample_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result.get(key) for key in (
        "root", "exists", "total_files", "total_bytes", "top_level_file_counts",
        "extension_counts", "partial_files",
    )}, indent=2))
    return 0 if result.get("exists") else 2


if __name__ == "__main__":
    raise SystemExit(main())
