"""Read-only, fail-closed readiness inspection. This script cannot launch training."""
import argparse
import hashlib
import json
from pathlib import Path

from mamaguardnet.orchestration.fold_gate import inspect_fold_gate


def readiness(manifest, results, preprocessed, repo):
    issues = []
    originals = [repo / 'configs/experiments' / x for x in (
        'a0_no_augmentation.yaml', 'a1_nnunet_standard.yaml', 'a2_independent.yaml',
        'a3_shared.yaml', 'a4_kinetics.yaml')]
    for fold in range(5):
        try:
            gate = inspect_fold_gate(results_root=results, preprocessed_root=preprocessed,
                                     config_paths=originals, fold=fold, seed=20261031)
            if not gate['ready']:
                issues.append('Original fold ' + str(fold) + ' incomplete')
        except (OSError, ValueError, KeyError) as exc:
            issues.append('Original campaign evidence unavailable: ' + str(exc))
    for key in ('artifacts_preserved', 'scientific_design_approved', 'gpu_authorized',
                'explicit_campaign_approval', 'training_smoke_passed', 'real_data_audit_passed'):
        if manifest.get(key) is not True:
            issues.append(key + ' not satisfied')
    for entry in manifest.get('experiments', []):
        path = repo / entry['config']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry.get('config_sha256'):
            issues.append(entry['id'] + ' config missing or hash not frozen')
    if not manifest.get('experiments'):
        issues.append('No experiments registered')
    # Approval must bind executable inputs, not only editable boolean switches.
    frozen = manifest.get('frozen_files', {})
    required = {
        'src/mamaguardnet/nnunet/secondary_trainers_v1.py',
        'src/mamaguardnet/nnunet/trainers.py',
        'src/mamaguardnet/augmentation/adaptive_v1.py',
        'src/mamaguardnet/augmentation/smooth_control.py',
        'scripts/secondary_run.py', 'scripts/check_secondary_campaign.py',
        'research/04_experiments/SECONDARY_PRESPECIFICATION_V1.md',
    }
    if not required.issubset(frozen):
        issues.append('Executable source and scientific protocol not frozen')
    for name, digest in frozen.items():
        path = (repo / name).resolve()
        if repo.resolve() not in path.parents or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            issues.append('Frozen file missing or changed: ' + name)
    split = preprocessed / 'Dataset501_MAMAMIA3Phase/splits_final.json'
    if not split.is_file() or hashlib.sha256(split.read_bytes()).hexdigest() != manifest.get('split_sha256'):
        issues.append('Development split not frozen')
    return {'ready': not issues, 'issues': issues, 'launch_performed': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--preprocessed', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    a = p.parse_args()
    report = readiness(json.loads(a.manifest.read_text()), a.results, a.preprocessed, a.repo)
    print(json.dumps(report, indent=2))
    return 0 if report['ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
