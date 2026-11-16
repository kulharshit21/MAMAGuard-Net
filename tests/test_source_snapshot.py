import importlib.util
import json
import tarfile
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "create_source_snapshot.py"
SPEC = importlib.util.spec_from_file_location("create_source_snapshot", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_snapshot_includes_source_and_excludes_sensitive_bulky_files(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "registry").mkdir()
    (root / "reports").mkdir()
    (root / "src" / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "configs" / "run.yaml").write_text("seed: 4\n", encoding="utf-8")
    (root / "registry" / "tasks.json").write_text('{"tasks": []}\n', encoding="utf-8")
    (root / "reports" / "PUBLICATION_ENGINE_STATUS.md").write_text(
        "# Status\n", encoding="utf-8"
    )
    (root / "src" / "checkpoint.pth").write_bytes(b"secret weights")
    archive, manifest = MODULE.create_snapshot(root, tmp_path / "snapshot")

    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert document["file_count"] == 4
    assert len(document["archive_sha256"]) == 64
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        names = [member.name for member in members]
    assert names == [
        "configs/run.yaml",
        "registry/tasks.json",
        "reports/PUBLICATION_ENGINE_STATUS.md",
        "src/model.py",
    ]
    assert all(member.mode == 0o644 for member in members)
