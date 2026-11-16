import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('secondary_run', ROOT/'scripts/secondary_run.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_candidate_command_isolation_and_resume(tmp_path):
    c = yaml.safe_load((ROOT/'configs/candidates/a5_matched_v1.yaml').read_text())
    cmd, output = module.plan(c, tmp_path, 0)
    assert c['trainer'] in cmd and '--c' not in cmd
    assert output.parent == tmp_path/c['experiment']
    folder = output/'Dataset501_MAMAMIA3Phase'/(c['trainer']+'__'+c['plans']+'__'+c['configuration'])/'fold_0'
    folder.mkdir(parents=True)
    (folder/'checkpoint_latest.pth').touch()
    assert '--c' in module.plan(c, tmp_path, 0)[0]
    (folder/'checkpoint_final.pth').touch()
    cmd = module.plan(c, tmp_path, 0)[0]
    assert '--val' in cmd and '--c' not in cmd


def test_yunnan_rejected(tmp_path):
    with pytest.raises(ValueError):
        module.plan({'dataset': 502}, tmp_path, 0)


def test_missing_campaign_and_approval_block(tmp_path):
    from check_secondary_campaign import readiness
    report = readiness({}, tmp_path, tmp_path, ROOT)
    assert not report['ready'] and not report['launch_performed']
    assert 'explicit_campaign_approval not satisfied' in report['issues']


def test_every_original_fold_and_each_approval_are_required(tmp_path, monkeypatch):
    import hashlib

    import check_secondary_campaign as gate
    calls = []
    monkeypatch.setattr(gate, 'inspect_fold_gate', lambda **kw: calls.append(kw['fold']) or {'ready': True})
    required = ('artifacts_preserved', 'scientific_design_approved', 'gpu_authorized',
                'explicit_campaign_approval', 'training_smoke_passed', 'real_data_audit_passed')
    manifest = {k: True for k in required}
    config = ROOT/'configs/candidates/a5_matched_v1.yaml'
    manifest['experiments'] = [{'id':'A5', 'config':str(config.relative_to(ROOT)),
                               'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest()}]
    names = ['src/mamaguardnet/nnunet/secondary_trainers_v1.py',
             'src/mamaguardnet/nnunet/trainers.py',
             'src/mamaguardnet/augmentation/adaptive_v1.py',
             'src/mamaguardnet/augmentation/smooth_control.py',
             'scripts/secondary_run.py', 'scripts/check_secondary_campaign.py',
             'research/04_experiments/SECONDARY_PRESPECIFICATION_V1.md']
    manifest['frozen_files'] = {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    split = tmp_path/'Dataset501_MAMAMIA3Phase/splits_final.json'
    split.parent.mkdir()
    split.write_text('[]')
    manifest['split_sha256'] = hashlib.sha256(split.read_bytes()).hexdigest()
    assert gate.readiness(manifest,tmp_path,tmp_path,ROOT)['ready']
    assert calls == list(range(5))
    for key in required:
        assert not gate.readiness(dict(manifest, **{key:False}),tmp_path,tmp_path,ROOT)['ready']
    monkeypatch.setattr(gate, 'inspect_fold_gate', lambda **kw: {'ready':kw['fold'] != 4})
    assert not gate.readiness(manifest,tmp_path,tmp_path,ROOT)['ready']


def test_execute_blocked_registry_never_calls_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['secondary_run', '--config',str(ROOT/'configs/candidates/a5_matched_v1.yaml'),
        '--manifest',str(ROOT/'registry/secondary_campaign_v1.json'), '--original-results',str(tmp_path/'original'),
        '--preprocessed',str(tmp_path/'preprocessed'), '--output-root',str(tmp_path/'secondary'),
        '--repo',str(ROOT),'--gpu','0','--execute'])
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: pytest.fail('Blocked launch reached subprocess'))
    with pytest.raises(RuntimeError, match='Campaign gates'):
        module.main()


def test_failed_resource_preflight_prevents_training(tmp_path, monkeypatch):
    import hashlib
    import json
    import subprocess
    config = ROOT/'configs/candidates/a5_matched_v1.yaml'
    data = yaml.safe_load(config.read_text())
    manifest = tmp_path/'approval_fixture.json'
    manifest.write_text(json.dumps({'experiments':[{'id':data['experiment'],
        'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest()}]}))
    monkeypatch.setattr(sys, 'argv', ['secondary_run','--config',str(config),'--manifest',str(manifest),
        '--original-results',str(tmp_path/'original'),'--preprocessed',str(tmp_path/'preprocessed'),
        '--output-root',str(tmp_path/'secondary'),'--gpu','0','--execute'])
    monkeypatch.setattr(module, 'readiness', lambda *a: {'ready':True,'issues':[]})
    monkeypatch.setattr(sys, 'prefix',str(tmp_path/'.venv-secondary'))
    calls = []
    def failed_preflight(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(2,command)
    monkeypatch.setattr(module.subprocess,'run',failed_preflight)
    with pytest.raises(subprocess.CalledProcessError):
        module.main()
    assert len(calls) == 1 and 'gpu-preflight' in calls[0]


@pytest.mark.parametrize('output', ['original','original/nested','.'])
def test_overlapping_output_rejected(tmp_path, monkeypatch, output):
    monkeypatch.setattr(sys, 'argv', ['secondary_run','--config','unused','--manifest','unused',
        '--original-results',str(tmp_path/'original'),'--preprocessed',str(tmp_path/'preprocessed'),
        '--output-root',str(tmp_path/output),'--gpu','0','--execute'])
    with pytest.raises(ValueError,match='separate'):
        module.main()
