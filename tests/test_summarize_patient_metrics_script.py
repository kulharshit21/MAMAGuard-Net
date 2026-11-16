import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_patient_metric_summary_validates_manifest_and_sources(tmp_path: Path):
    metrics = tmp_path / "metrics.csv"
    manifest = tmp_path / "manifest.csv"
    expected = tmp_path / "expected.txt"
    output = tmp_path / "summary.json"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "dice", "severe_failure"])
        writer.writeheader()
        writer.writerows(
            [
                {"case_id": "case_a", "dice": 0.4, "severe_failure": False},
                {"case_id": "case_b", "dice": 0.8, "severe_failure": True},
            ]
        )
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "patient_id", "source"])
        writer.writeheader()
        writer.writerows(
            [
                {"case_id": "case_a", "patient_id": "patient_a", "source": "DUKE"},
                {"case_id": "case_b", "patient_id": "patient_b", "source": "ISPY1"},
            ]
        )
    expected.write_text("case_a\ncase_b\n", encoding="utf-8")
    environment = os.environ.copy()
    source = str((Path.cwd() / "src").resolve())
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source, environment.get("PYTHONPATH", "")) if value
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_patient_metrics.py",
            "--metrics-csv",
            str(metrics),
            "--manifest",
            str(manifest),
            "--expected-cases",
            str(expected),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["coverage"]["complete"] is True
    assert report["overall"]["metrics"]["dice"]["mean"] == pytest.approx(0.6)
    assert set(report["groups"]) == {"DUKE", "ISPY1"}
