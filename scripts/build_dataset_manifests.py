#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mamaguardnet.data.cases import write_case_manifest
from mamaguardnet.data.layouts import discover_mamamia, discover_yunnan, layout_summary
from mamaguardnet.utils.atomic import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Build manifests from the verified DGX layouts")
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--allow-count-mismatch", action="store_true")
    args = parser.parse_args()
    mamamia, mamamia_issues = discover_mamamia(args.data_root)
    yunnan, yunnan_issues = discover_yunnan(args.data_root)
    summary = {
        "mamamia": layout_summary(mamamia, mamamia_issues),
        "yunnan": layout_summary(yunnan, yunnan_issues),
        "phase_policy": {
            "mamamia": {"pre": "0000", "early": "0001", "late": "0002"},
            "yunnan": {"pre": "P0", "early": "P1", "late": "P2"},
            "extra_phases": "recorded_for_QC_but_excluded_from_matched_three_phase_model",
            "external_label": "GT.nii.gz remains evaluation-only",
        },
    }
    if not args.allow_count_mismatch and (len(mamamia) != 1506 or len(yunnan) != 100):
        raise SystemExit(
            f"Unexpected case counts: MAMA-MIA={len(mamamia)} (expected 1506), "
            f"Yunnan={len(yunnan)} (expected 100)"
        )
    if mamamia_issues or yunnan_issues:
        raise SystemExit(json.dumps(summary, indent=2))
    write_case_manifest(args.output_dir / "mamamia_cases.csv", mamamia)
    write_case_manifest(args.output_dir / "yunnan_cases.csv", yunnan)
    atomic_write_json(args.output_dir / "layout_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
