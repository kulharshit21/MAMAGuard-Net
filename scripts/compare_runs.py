#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mamaguardnet.statistics.bootstrap import paired_bootstrap_ci


def read_metric(path: Path, metric: str) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["case_id"]: float(row[metric]) for row in csv.DictReader(handle)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--metric", default="dice")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    candidate, baseline = read_metric(args.candidate, args.metric), read_metric(args.baseline, args.metric)
    if set(candidate) != set(baseline):
        raise SystemExit("Paired comparison requires identical case IDs")
    cases = sorted(candidate)
    result = paired_bootstrap_ci(
        np.asarray([candidate[case] for case in cases]),
        np.asarray([baseline[case] for case in cases]),
        samples=args.samples,
        seed=args.seed,
    )
    result.update({"metric": args.metric, "candidate": str(args.candidate), "baseline": str(args.baseline)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

