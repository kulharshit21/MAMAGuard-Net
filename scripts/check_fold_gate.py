#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mamaguardnet.orchestration.fold_gate import inspect_fold_gate
from mamaguardnet.utils.atomic import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that a matched experiment fold has complete validation artifacts."
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--preprocessed-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = inspect_fold_gate(
        results_root=args.results_root,
        preprocessed_root=args.preprocessed_root,
        config_paths=args.config,
        fold=args.fold,
        seed=args.seed,
    )
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
