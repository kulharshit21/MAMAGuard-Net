import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_validator_refuses_active_environment(monkeypatch):
    module = load('validate_secondary_cpu')
    monkeypatch.setattr(sys,'argv',['validate','--model-root','unused'])
    monkeypatch.setattr(sys,'prefix','/active/.venv')
    with pytest.raises(RuntimeError,match='isolated'):
        module.main()


def test_cpu_validator_defers_low_memory(tmp_path,monkeypatch):
    module = load('validate_secondary_cpu')
    for name in ('plans.json','dataset.json'):
        (tmp_path/name).write_text('{}')
    monkeypatch.setattr(sys,'argv',['validate','--model-root',str(tmp_path)])
    monkeypatch.setattr(sys,'prefix',str(tmp_path/'.venv-secondary'))
    monkeypatch.setitem(sys.modules,'psutil',SimpleNamespace(virtual_memory=lambda:SimpleNamespace(available=0)))
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:pytest.fail('Resource gate bypassed'))
    with pytest.raises(RuntimeError,match='RAM'):
        module.main()


def test_installer_idempotent_only_for_identical_isolated_plugin(tmp_path,monkeypatch):
    module = load('install_secondary_plugin')
    prefix = tmp_path/'.venv-secondary'
    package = prefix/'site-packages/nnunetv2'
    (package/'training/nnUNetTrainer/variants').mkdir(parents=True)
    monkeypatch.setattr(sys,'prefix',str(prefix))
    monkeypatch.setitem(sys.modules,'nnunetv2',SimpleNamespace(__file__=str(package/'__init__.py')))
    module.main()
    module.main()
    target = package/'training/nnUNetTrainer/variants/mamaguard_secondary_v1.py'
    target.write_text('different plugin')
    with pytest.raises(RuntimeError,match='overwrite'):
        module.main()
