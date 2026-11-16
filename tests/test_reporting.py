import math

from mamaguardnet.evaluation.reporting import (
    summarize_rows,
    summarize_rows_by_group,
    validate_case_coverage,
)


def test_summary_exposes_nonfinite_metrics_and_failure_rates():
    rows = [
        {"case_id": "a", "dice": 0.8, "hd95_mm": 2.0, "failed_segmentation": False},
        {"case_id": "b", "dice": 0.6, "hd95_mm": math.inf, "failed_segmentation": True},
    ]
    summary = summarize_rows(rows)
    assert summary["case_count"] == 2
    assert summary["metrics"]["dice"]["mean"] == 0.7
    assert summary["metrics"]["hd95_mm"]["mean"] is None
    assert summary["metrics"]["hd95_mm"]["nonfinite_count"] == 1
    assert summary["metrics"]["hd95_mm"]["finite_mean"] == 2.0
    assert summary["failed_segmentation_count"] == 1
    assert summary["failed_segmentation_rate"] == 0.5


def test_summary_includes_dispersion_tail_and_source_groups():
    rows = [
        {"case_id": "a", "source": "DUKE", "dice": 0.2, "severe_failure": False},
        {"case_id": "b", "source": "DUKE", "dice": 0.6, "severe_failure": False},
        {"case_id": "c", "source": "ISPY1", "dice": 1.0, "severe_failure": False},
    ]
    report = summarize_rows_by_group(rows)
    dice = report["overall"]["metrics"]["dice"]
    assert dice["median"] == 0.6
    assert dice["standard_deviation"] == 0.4
    assert dice["p10"] == 0.28
    assert report["groups"]["DUKE"]["case_count"] == 2
    assert report["groups"]["ISPY1"]["case_count"] == 1


def test_case_coverage_reports_missing_unexpected_and_duplicates():
    rows = [{"case_id": "a"}, {"case_id": "a"}, {"case_id": "c"}]
    coverage = validate_case_coverage(rows, ["a", "b"])
    assert coverage["missing_case_ids"] == ["b"]
    assert coverage["unexpected_case_ids"] == ["c"]
    assert coverage["duplicate_case_ids"] == ["a"]
    assert coverage["complete"] is False
