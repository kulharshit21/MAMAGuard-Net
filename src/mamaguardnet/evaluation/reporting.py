from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def summarize_rows(
    rows: Sequence[Mapping[str, float | int | bool | str]],
) -> dict[str, object]:
    """Summarize case metrics without hiding undefined or infinite values."""
    if not rows:
        raise ValueError("At least one case row is required")

    summary: dict[str, object] = {"case_count": len(rows), "metrics": {}}
    metric_summary: dict[str, object] = summary["metrics"]  # type: ignore[assignment]
    for key, example in rows[0].items():
        if key == "case_id" or isinstance(example, (bool, np.bool_)):
            continue
        if not isinstance(example, (int, float, np.integer, np.floating)):
            continue
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        all_finite = len(finite) == len(values)
        metric_summary[key] = {
            "all_finite": all_finite,
            "finite_count": len(finite),
            "nonfinite_count": len(values) - len(finite),
            "mean": float(np.mean(values)) if all_finite else None,
            "median": float(np.median(values)) if all_finite else None,
            "standard_deviation": float(np.std(values, ddof=1)) if all_finite and len(values) > 1 else 0.0 if all_finite else None,
            "p10": float(np.quantile(values, 0.1)) if all_finite else None,
            "p25": float(np.quantile(values, 0.25)) if all_finite else None,
            "p75": float(np.quantile(values, 0.75)) if all_finite else None,
            "minimum": float(np.min(values)) if all_finite else None,
            "maximum": float(np.max(values)) if all_finite else None,
            "finite_mean": float(np.mean(finite)) if len(finite) else None,
            "finite_median": float(np.median(finite)) if len(finite) else None,
            "finite_standard_deviation": (
                float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0 if len(finite) else None
            ),
            "finite_p10": float(np.quantile(finite, 0.1)) if len(finite) else None,
        }

    for key, example in rows[0].items():
        if isinstance(example, (bool, np.bool_)):
            summary[f"{key}_count"] = int(sum(bool(row[key]) for row in rows))
            summary[f"{key}_rate"] = float(np.mean([bool(row[key]) for row in rows]))
    return summary


def summarize_rows_by_group(
    rows: Sequence[Mapping[str, float | int | bool | str]],
    *,
    group_key: str = "source",
) -> dict[str, object]:
    """Return the global summary plus prespecified patient groups."""
    if not rows:
        raise ValueError("At least one case row is required")
    missing = [str(row.get("case_id", "<unknown>")) for row in rows if group_key not in row]
    if missing:
        raise ValueError(f"Missing {group_key!r} for cases: {missing}")
    groups: dict[str, list[Mapping[str, float | int | bool | str]]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)
    return {
        "overall": summarize_rows(rows),
        "group_key": group_key,
        "groups": {name: summarize_rows(group) for name, group in sorted(groups.items())},
    }


def validate_case_coverage(
    rows: Sequence[Mapping[str, float | int | bool | str]],
    expected_case_ids: Sequence[str],
) -> dict[str, object]:
    observed = [str(row["case_id"]) for row in rows]
    duplicates = sorted({case_id for case_id in observed if observed.count(case_id) > 1})
    expected = set(expected_case_ids)
    observed_set = set(observed)
    return {
        "expected_count": len(expected),
        "observed_count": len(observed),
        "missing_case_ids": sorted(expected - observed_set),
        "unexpected_case_ids": sorted(observed_set - expected),
        "duplicate_case_ids": duplicates,
        "complete": not duplicates and observed_set == expected and len(observed) == len(expected),
    }

