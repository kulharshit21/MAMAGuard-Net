from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from mamaguardnet.utils.atomic import atomic_write_json


class ExternalTestLocked(RuntimeError):
    pass


def assert_external_allowed(state_path: Path, *, config_hash: str, checkpoint_hash: str) -> None:
    if not state_path.exists():
        raise ExternalTestLocked(
            "External evaluation is locked. Freeze the configuration and checkpoint first."
        )
    import json

    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("consumed_at_utc"):
        raise ExternalTestLocked("External evaluation token has already been consumed")
    if state.get("config_hash") != config_hash or state.get("checkpoint_hash") != checkpoint_hash:
        raise ExternalTestLocked("Frozen hashes do not match the requested evaluation")


def freeze_external(state_path: Path, *, config_hash: str, checkpoint_hash: str) -> None:
    if state_path.exists():
        raise ExternalTestLocked(f"External lock state already exists: {state_path}")
    atomic_write_json(
        state_path,
        {
            "config_hash": config_hash,
            "checkpoint_hash": checkpoint_hash,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "consumed_at_utc": None,
        },
    )


def consume_external(state_path: Path, *, config_hash: str, checkpoint_hash: str) -> None:
    assert_external_allowed(state_path, config_hash=config_hash, checkpoint_hash=checkpoint_hash)
    import json

    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["consumed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(state_path, state)
