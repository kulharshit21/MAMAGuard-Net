from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from mamaguardnet.utils.hashing import hash_object

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    pass


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ConfigError(f"Required environment variable is not set: {name}")

    return _ENV.sub(replace, value)


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration root must be a mapping: {source}")
    data = _expand(data)
    data["_meta"] = {"source": str(source.resolve()), "sha256": hash_object(data)}
    return data


def validate_experiment(config: dict[str, Any]) -> None:
    gpus = config.get("gpus", [1, 5, 6, 7])
    if not gpus or len(gpus) != len(set(gpus)) or any(not isinstance(g, int) or g < 0 for g in gpus):
        raise ConfigError("gpus must contain unique nonnegative device IDs")
    if config.get("num_gpus", 4) != len(gpus):
        raise ConfigError("num_gpus must match the number of selected GPUs")
    if config.get("status") == "planned_not_implemented":
        raise ConfigError(config.get("blocker", "Experiment is marked planned_not_implemented"))
    required = ["experiment", "dataset", "trainer", "fold", "augmentation"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"Missing experiment keys: {', '.join(missing)}")
    fold = config["fold"]
    if not (fold == "all" or isinstance(fold, int) or str(fold).isdigit()):
        raise ConfigError("fold must be an integer or 'all'")
    allowed = {"none", "standard", "independent", "shared", "kinetics"}
    if config["augmentation"] not in allowed:
        raise ConfigError(f"augmentation must be one of {sorted(allowed)}")

