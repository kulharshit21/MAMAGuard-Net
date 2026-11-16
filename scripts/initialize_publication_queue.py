#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from mamaguardnet.orchestration.task_registry import TaskRegistry
from mamaguardnet.utils.hashing import hash_object, sha256_file


def source_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    paths: list[Path] = []
    for directory in ("src", "scripts", "configs", "tests"):
        paths.extend(path for path in (root / directory).rglob("*") if path.is_file())
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in relative or relative.endswith((".pyc", ".pyo")):
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def task(
    task_id: str,
    task_type: str,
    hypothesis: str,
    *,
    status: str,
    code_hash: str,
    data_hash: str,
    config: dict,
    dependencies: list[str] | None = None,
    gpu_count: int = 0,
    launch_approved: bool = False,
    outputs: list[str] | None = None,
) -> dict:
    return {
        "task_id": task_id,
        "task_type": task_type,
        "scientific_hypothesis": hypothesis,
        "dependencies": dependencies or [],
        "priority": 1 if task_type == "ENGINEERING" else 2,
        "owner": "MAMAGuard-Net research team",
        "code_snapshot_hash": code_hash,
        "data_split_hash": data_hash,
        "configuration_hash": hash_object(config),
        "resources": {
            "cpu_cores": 32 if gpu_count == 0 else 16,
            "ram_gib": 128 if gpu_count == 0 else 64,
            "gpu_count": gpu_count,
        },
        "resource_authorized": gpu_count == 0,
        "expected_outputs": outputs or ["registered result and provenance report"],
        "test_requirements": [
            "unit correctness",
            "synthetic integration",
            "real MAMA-MIA compatibility",
            "geometry and provenance validation",
        ],
        "launch_approved": launch_approved,
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize the gated publication task registry")
    parser.add_argument("--output", type=Path, default=Path("registry/publication_tasks.json"))
    parser.add_argument("--split", type=Path, default=Path("manifests/splits_final.json"))
    parser.add_argument(
        "--split-hash",
        help="Verified SHA-256 to use when the DGX split file is intentionally absent locally",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing task registry: {args.output}")

    root = Path(__file__).resolve().parents[1]
    code_hash = source_tree_hash(root)
    if (root / args.split).is_file():
        data_hash = sha256_file(root / args.split)
        if args.split_hash and args.split_hash != data_hash:
            raise SystemExit("Provided split hash does not match the local split file")
    elif args.split_hash:
        if len(args.split_hash) != 64 or any(character not in "0123456789abcdef" for character in args.split_hash):
            raise SystemExit("--split-hash must be a lowercase SHA-256 digest")
        data_hash = args.split_hash
    else:
        raise SystemExit("Split file is unavailable; provide its previously verified --split-hash")
    registry = TaskRegistry(root / args.output)

    specs = [
        task(
            "ENG-A3-FOLD0-PATIENT-REPORT",
            "ENGINEERING",
            "A3 fold 0 has finite full-volume patient metrics that can be summarized without retraining.",
            status="READY_FOR_TEST",
            code_hash=code_hash,
            data_hash=data_hash,
            config={"script": "summarize_patient_metrics.py", "experiment": "A3", "fold": 0},
            outputs=["patient metrics JSON", "source-stratified metrics JSON"],
        ),
        task(
            "ENG-A3-REAL-XAI",
            "ENGINEERING",
            "A fixed, non-cherry-picked A3 validation case can produce finite geometry-aligned attribution.",
            status="WAITING_FOR_DEPENDENCY",
            code_hash=code_hash,
            data_hash=data_hash,
            config={"model": "A3", "fold": 0, "selection": "first lexical validation case"},
            dependencies=["ENG-A3-FOLD0-PATIENT-REPORT"],
            outputs=["Grad-CAM array", "overlay figure", "phase contribution report"],
        ),
        task(
            "ENG-A3-PREDICTED-3D",
            "ENGINEERING",
            "A3 prediction geometry can produce reproducible physical tumor measurements and mesh exports.",
            status="WAITING_FOR_DEPENDENCY",
            code_hash=code_hash,
            data_hash=data_hash,
            config={"model": "A3", "fold": 0, "measurement_source": "raw prediction mask"},
            dependencies=["ENG-A3-FOLD0-PATIENT-REPORT"],
            outputs=["VTP mesh", "PNG preview", "prediction-expert measurement JSON"],
        ),
        task(
            "E1-A5-INDEPENDENT-SMOOTH",
            "GPU_RESEARCH",
            "Independent smooth fields differ from shared smooth fields when all marginal field parameters are matched.",
            status="WAITING_FOR_APPROVAL",
            code_hash=code_hash,
            data_hash=data_hash,
            config={"candidate": "configs/candidates/a5_independent_smooth.yaml"},
            gpu_count=1,
        ),
        task(
            "E2-SOURCE-HELD-OUT",
            "GPU_RESEARCH",
            "A4 may improve tumor segmentation robustness when an entire source cohort is unseen in training.",
            status="WAITING_FOR_APPROVAL",
            code_hash=code_hash,
            data_hash=data_hash,
            config={"protocol": "leave-one-source-out", "sources": ["DUKE", "ISPY1", "ISPY2", "NACT"]},
            gpu_count=1,
        ),
        task(
            "E3-CONFIRMATORY-SEEDS",
            "GPU_RESEARCH",
            "Any observed A4-versus-A1 difference should be stable across prespecified random seeds.",
            status="WAITING_FOR_DEPENDENCY",
            code_hash=code_hash,
            data_hash=data_hash,
            config={"methods": ["A1", "A4"], "seed_count": "pending power review"},
            dependencies=["ENG-A3-FOLD0-PATIENT-REPORT"],
            gpu_count=1,
        ),
        task(
            "E4-PUBLISHED-COMPARATOR",
            "GPU_RESEARCH",
            "A verified published comparator provides a stronger benchmark than standard nnU-Net alone.",
            status="WAITING_FOR_APPROVAL",
            code_hash=code_hash,
            data_hash=data_hash,
            config={"comparator": "not selected", "protocol": "must be matched before approval"},
            gpu_count=1,
        ),
        task(
            "E5-EXPLANATION-ROBUSTNESS",
            "ENGINEERING",
            "A4 may change attribution stability under scanner-style intensity perturbations relative to A1.",
            status="WAITING_FOR_DEPENDENCY",
            code_hash=code_hash,
            data_hash=data_hash,
            config={"methods": ["A1", "A4"], "analysis": "paired attribution stability"},
            dependencies=["ENG-A3-REAL-XAI"],
            gpu_count=0,
        ),
        task(
            "E6-VOLUME-RELIABILITY",
            "ENGINEERING",
            "Segmentation differences may translate into different physical tumor-volume errors.",
            status="WAITING_FOR_DEPENDENCY",
            code_hash=code_hash,
            data_hash=data_hash,
            config={"methods": ["A0", "A1", "A2", "A3", "A4"], "analysis": "paired volume error"},
            dependencies=["ENG-A3-PREDICTED-3D"],
            gpu_count=0,
        ),
        task(
            "E7-YUNNAN-EXTERNAL",
            "EXTERNAL_EVALUATION",
            "The frozen selected method generalizes to the sealed independent Yunnan cohort.",
            status="WAITING_FOR_DEPENDENCY",
            code_hash=code_hash,
            data_hash="SEALED_EXTERNAL_NOT_ACCESSED",
            config={"dataset": "Yunnan", "single_use": True, "lock_required": True},
            dependencies=["E6-VOLUME-RELIABILITY"],
            gpu_count=1,
        ),
    ]
    for spec in specs:
        registry.register(spec)
    print(f"Registered {len(specs)} tasks in {root / args.output}")
    print(f"Code tree hash: {code_hash}")
    print(f"Split hash: {data_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
