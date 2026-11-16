from __future__ import annotations

import shlex
from typing import Any

TRAINERS = {
    "none": "nnUNetTrainerMAMAGuardNoDA",
    "standard": "nnUNetTrainerMAMAGuardStandard",
    "independent": "nnUNetTrainerMAMAGuardIndependent",
    "shared": "nnUNetTrainerMAMAGuardShared",
    "kinetics": "nnUNetTrainerMAMAGuardKinetics",
}


def build_train_command(config: dict[str, Any], *, resume: bool = False, validation_only: bool = False) -> list[str]:
    augmentation = config["augmentation"]
    trainer = config.get("trainer") or TRAINERS[augmentation]
    command = [
        "nnUNetv2_train",
        str(config["dataset"]),
        config.get("configuration", "3d_fullres"),
        str(config["fold"]),
        "-tr", trainer,
        "-num_gpus", str(config.get("num_gpus", 4)),
    ]
    if config.get("plans"):
        command.extend(["-p", str(config["plans"])])
    if config.get("save_probabilities", True):
        command.append("--npz")
    if resume:
        command.append("--c")
    if validation_only:
        command.append("--val")
    return command


def shell_join(command: list[str]) -> str:
    return shlex.join(command)

