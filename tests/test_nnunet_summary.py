import pytest

from mamaguardnet.evaluation.nnunet_summary import summarize_nnunet_validation


def test_nnunet_summary_reports_patient_distribution_sources_and_missing_metrics():
    summary = {
        "metric_per_case": [
            {
                "prediction_file": "/pred/DUKE_001.nii.gz",
                "metrics": {
                    "1": {
                        "Dice": 0.2,
                        "TP": 2,
                        "FP": 3,
                        "FN": 4,
                        "n_pred": 5,
                        "n_ref": 6,
                    }
                },
            },
            {
                "prediction_file": "/pred/ISPY1_002.nii.gz",
                "metrics": {
                    "1": {
                        "Dice": 0.8,
                        "TP": 8,
                        "FP": 1,
                        "FN": 2,
                        "n_pred": 9,
                        "n_ref": 10,
                    }
                },
            },
        ]
    }
    manifest = {
        "DUKE_001": {"patient_id": "DUKE_001", "source": "DUKE"},
        "ISPY1_002": {"patient_id": "ISPY1_002", "source": "ISPY1"},
    }
    report = summarize_nnunet_validation(summary, manifest, ["DUKE_001", "ISPY1_002"])
    assert report["coverage"]["complete"] is True
    assert report["overall"]["metrics"]["dice"]["mean"] == pytest.approx(0.5)
    assert report["overall"]["metrics"]["dice"]["p10"] == pytest.approx(0.26)
    assert set(report["groups"]) == {"DUKE", "ISPY1"}
    assert "hd95_mm" in report["not_available_in_nnunet_summary"]
    assert report["overall"]["metrics"]["tumor_sensitivity"]["mean"] == pytest.approx(
        (2 / 6 + 8 / 10) / 2
    )
    assert report["overall"]["metrics"]["relative_volume_error"]["mean"] == pytest.approx(
        (-1 / 6 - 1 / 10) / 2
    )
    assert report["overall"]["severe_failure_count"] == 0


def test_nnunet_summary_rejects_missing_patient():
    summary = {
        "metric_per_case": [
            {"prediction_file": "a.nii.gz", "metrics": {"1": {"Dice": 0.5}}},
        ]
    }
    with pytest.raises(ValueError, match="coverage"):
        summarize_nnunet_validation(
            summary,
            {"a": {"source": "DUKE"}},
            ["a", "b"],
        )
