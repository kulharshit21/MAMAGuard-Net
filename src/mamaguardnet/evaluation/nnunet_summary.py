from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mamaguardnet.evaluation.reporting import summarize_rows_by_group, validate_case_coverage


def case_id_from_prediction(value: str) -> str:
    name = Path(value).name
    return name.removesuffix(".nii.gz").removesuffix(".nii")


def _foreground_metrics(case: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = case.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("nnU-Net case lacks metrics mapping")
    foreground = [value for key, value in metrics.items() if str(key) not in {"0", "background"}]
    if len(foreground) != 1 or not isinstance(foreground[0], Mapping):
        raise ValueError("Expected exactly one non-background label in nnU-Net summary")
    return foreground[0]


def summarize_nnunet_validation(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Mapping[str, str]],
    expected_case_ids: Sequence[str],
) -> dict[str, object]:
    cases = summary.get("metric_per_case")
    if not isinstance(cases, list) or not cases:
        raise ValueError("nnU-Net summary has no metric_per_case records")
    rows: list[dict[str, float | int | bool | str]] = []
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("Invalid nnU-Net case record")
        prediction = case.get("prediction_file") or case.get("prediction")
        if not isinstance(prediction, str):
            raise ValueError("nnU-Net case lacks prediction file")
        case_id = case_id_from_prediction(prediction)
        if case_id not in manifest:
            raise ValueError(f"Prediction case absent from manifest: {case_id}")
        metric = _foreground_metrics(case)
        if "Dice" not in metric:
            raise ValueError(f"Dice missing for case: {case_id}")
        record = manifest[case_id]
        row: dict[str, float | int | bool | str] = {
            "case_id": case_id,
            "patient_id": record.get("patient_id", case_id),
            "source": record.get("source", "UNKNOWN"),
            "dice": float(metric["Dice"]),
        }
        for source_name, output_name in (
            ("TP", "true_positive_voxels"),
            ("FP", "false_positive_voxels"),
            ("FN", "false_negative_voxels"),
            ("n_pred", "predicted_voxels"),
            ("n_ref", "reference_voxels"),
        ):
            if source_name in metric:
                row[output_name] = int(metric[source_name])
        true_positive = int(metric.get("TP", 0))
        false_positive = int(metric.get("FP", 0))
        false_negative = int(metric.get("FN", 0))
        predicted = int(metric.get("n_pred", true_positive + false_positive))
        reference = int(metric.get("n_ref", true_positive + false_negative))
        row.update(
            {
                "tumor_sensitivity": (
                    true_positive / reference if reference else 1.0 if predicted == 0 else 0.0
                ),
                "precision": (
                    true_positive / predicted if predicted else 1.0 if reference == 0 else 0.0
                ),
                "relative_volume_error": (
                    (predicted - reference) / reference
                    if reference
                    else 0.0 if predicted == 0 else float("inf")
                ),
                "failed_segmentation": bool(reference > 0 and predicted == 0),
                "severe_failure": bool(reference > 0 and float(metric["Dice"]) < 0.1),
            }
        )
        rows.append(row)

    coverage = validate_case_coverage(rows, expected_case_ids)
    if not coverage["complete"]:
        raise ValueError(f"Validation prediction coverage failed: {coverage}")
    return {
        "coverage": coverage,
        "available_metrics": sorted(
            key for key in rows[0] if key not in {"case_id", "patient_id", "source"}
        ),
        "not_available_in_nnunet_summary": [
            "hd95_mm",
            "mean_surface_distance_mm",
            "surface_dice_2mm",
            "physical_volume_error_mm3",
        ],
        **summarize_rows_by_group(rows),
    }
