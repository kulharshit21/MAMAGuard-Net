"""Prepare or execute one approved candidate in a separately installed environment."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from check_secondary_campaign import readiness

from mamaguardnet.nnunet.commands import build_train_command


def plan(config, root, fold):
    if config['dataset'] != 501 or fold not in range(5):
        raise ValueError('Only development Dataset501 and folds 0..4 are permitted')
    if config['experiment'] not in ('A5_independent_smooth_matched',
                                    'A6_adaptive_input_ordering_v1', 'A7_shuffled_reliability_v1'):
        raise ValueError('Unknown candidate')
    c = dict(config, fold=fold, augmentation='secondary', num_gpus=1)
    output = root / c['experiment'] / ('seed_' + str(c['seed']))
    checkpoint_dir = output / 'Dataset501_MAMAMIA3Phase' / (
        c['trainer'] + '__' + c['plans'] + '__' + c['configuration']) / ('fold_' + str(fold))
    final = checkpoint_dir / 'checkpoint_final.pth'
    latest = checkpoint_dir / 'checkpoint_latest.pth'
    command = build_train_command(c, resume=latest.is_file() and not final.is_file(),
                                  validation_only=final.is_file())
    return command, output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--original-results', type=Path, required=True)
    p.add_argument('--preprocessed', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    original = a.original_results.resolve()
    secondary = a.output_root.resolve()
    if secondary == original or original in secondary.parents or secondary in original.parents:
        raise ValueError('Use a separate secondary results root')
    config = yaml.safe_load(a.config.read_text())
    manifest = json.loads(a.manifest.read_text())
    command, output = plan(config, a.output_root, a.fold)
    gate = readiness(manifest, a.original_results, a.preprocessed, a.repo)
    digest = hashlib.sha256(a.config.read_bytes()).hexdigest()
    entries = [e for e in manifest['experiments'] if e['id'] == config['experiment']]
    if len(entries) != 1 or entries[0].get('config_sha256') != digest:
        gate['ready'] = False
        gate['issues'].append('Selected config not frozen in approved registry')
    print(json.dumps(dict(command=command, output=str(output), gate=gate), indent=2))
    if not a.execute:
        return 0
    if not gate['ready']:
        raise RuntimeError('Campaign gates not satisfied')
    if Path(sys.prefix).name != '.venv-secondary':
        raise RuntimeError('Execution requires isolated .venv-secondary environment')
    # Always re-run real resource preflight; authorization alone is insufficient.
    subprocess.run([sys.executable, '-m', 'mamaguardnet.cli', 'gpu-preflight', '--gpus',
                    str(a.gpu), '--min-free-mib', '36000', '--max-utilization', '10'], check=True)
    params = dict(config.get('augmentation_parameters', config.get('parameters', {})))
    probability = params.pop('probability', .5)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), nnUNet_results=str(output),
               nnUNet_preprocessed=str(a.preprocessed), MAMAGUARD_SEED=str(config['seed']),
               MAMAGUARD_AUG_P=str(probability), MAMAGUARD_SECONDARY_PARAMETERS=json.dumps(params))
    executable = Path(sys.executable).parent / ('nnUNetv2_train.exe' if os.name == 'nt' else 'nnUNetv2_train')
    if not executable.is_file():
        raise RuntimeError('Isolated nnUNet training entry point missing')
    command[0] = str(executable)
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
