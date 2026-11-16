"""Exploratory patient-paired fold-0 audit from existing summaries, no image reads."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    records = {r['case_id']: r for r in csv.DictReader(a.manifest.open())}
    methods = {}
    prediction_paths = {}
    for name in ['A0_no_augmentation', 'A1_nnunet_standard', 'A2_independent_phase_intensity',
                 'A3_shared_affine', 'A4_kinetics_constrained_primary']:
        paths = list((a.results/name/'seed_20261031').glob('Dataset501*/*/fold_0/validation/summary.json'))
        if len(paths) != 1:
            raise ValueError('Expected exactly one completed summary: '+name)
        data = json.loads(paths[0].read_text())
        rows = {}
        for row in data['metric_per_case']:
            case = Path(row['reference_file']).name.removesuffix('.nii.gz')
            if case in rows or records[case]['dataset_role'] != 'development':
                raise ValueError('Duplicate case or non-development input')
            rows[case] = row['metrics']['1']
            prediction_paths[case] = paths[0].parent / (case+'.nii.gz')
        methods[name[:2]] = rows
    ids = sorted(methods['A4'])
    if any(set(x) != set(ids) for x in methods.values()):
        raise ValueError('Unmatched patient coverage')
    report = {'status': 'PRELIMINARY_INTERNAL', 'n': len(ids), 'paired': {}, 'cases': []}
    rng = np.random.default_rng(20261031)
    for method in ['A0', 'A1', 'A2', 'A3']:
        delta = np.array([methods['A4'][c]['Dice']-methods[method][c]['Dice'] for c in ids])
        boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
        groups = sorted({records[c]['source'] for c in ids})
        report['paired']['A4_minus_'+method] = {
            'mean': float(delta.mean()), 'median': float(np.median(delta)),
            'exploratory_bootstrap95': np.quantile(boot, [.025, .975]).tolist(),
            'by_source': {g: float(np.mean([delta[i] for i,c in enumerate(ids) if records[c]['source']==g])) for g in groups}}
    for c in ids:
        import nibabel as nib
        voxel_mm3 = abs(float(np.linalg.det(nib.load(prediction_paths[c]).affine[:3, :3])))
        ref = methods['A4'][c]
        volume = (ref['TP'] + ref['FN']) * voxel_mm3 / 1000
        report['cases'].append({'case_id': c, 'source': records[c]['source'],
                                'reference_volume_cm3': volume,
                                'size_group': '<1' if volume < 1 else ('1-10' if volume <= 10 else '>10'),
                                'dice': {m: rows[c]['Dice'] for m,rows in methods.items()},
                                'absolute_volume_error_cm3': {m: abs(rows[c]['FP']-rows[c]['FN'])*voxel_mm3/1000 for m,rows in methods.items()},
                                'severe_failure': {m: rows[c]['Dice'] < .1 for m,rows in methods.items()}})
    report['size_strata'] = {}
    for group in ('<1', '1-10', '>10'):
        subset = [c for c in report['cases'] if c['size_group'] == group]
        report['size_strata'][group] = {'n': len(subset), 'mean_dice': {
            m: float(np.mean([c['dice'][m] for c in subset])) if subset else None for m in methods}}
    report['limitations'] = ['Fold 0 used for hypothesis development; intervals are exploratory.',
                            'Volume combines summary TP/FN/FP counts with saved prediction affine voxel determinant; original geometry validation remains required.',
                            'Enhancement strata not measured; MRI intensity images are not in the mirror.']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}, indent=2))


if __name__ == '__main__':
    main()
