"""Bounded local MRI input audit; no training or external cohort access."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from mamaguardnet.augmentation.adaptive_v1 import adaptive_affine, reliability_map
from mamaguardnet.augmentation.smooth_control import sample_smooth_affine_fields


def audit(folder, seeds=32):
    paths = sorted(folder.glob('*_000[012].nii.gz'))
    if len(paths) != 3 or not all(p.name.startswith('NACT_53_') for p in paths):
        raise ValueError('Only the explicitly permitted NACT_53 development fixture is supported')
    images = [nib.load(p) for p in paths]
    if any(i.shape != images[0].shape or not np.allclose(i.affine, images[0].affine) for i in images):
        raise ValueError('Phase geometry mismatch')
    # A deterministic central crop; not a tumor selection or representative cohort.
    slices = tuple(slice(max(0, n//2-16), min(n, n//2+16)) for n in images[0].shape)
    full = [np.asarray(i.dataobj, dtype=np.float32) for i in images]
    raw = np.stack([x[slices] for x in full])
    normalized = np.stack([((x-x.mean())/max(float(x.std()), 1e-6))[slices] for x in full])
    normalization_changes = []
    for a,b in ((0,1),(0,2),(1,2)):
        difference = raw[a]-raw[b]
        eligible = np.abs(difference) > 1e-6
        normalization_changes.extend((np.sign(difference[eligible]) != np.sign((normalized[a]-normalized[b])[eligible])).tolist())
    rows = []
    for space, x in [('raw', raw), ('independent_zscore', normalized)]:
        r = reliability_map(x)
        margin = np.diff(np.sort(x, axis=0), axis=0).min(axis=0)
        modes = {}
        for mode in ('adaptive', 'shifted', 'constant', 'shuffled'):
            values = []
            for seed in range(seeds):
                out = adaptive_affine(x, rng=np.random.default_rng(seed), mode=mode)
                changes = []
                for a,b in ((0,1),(0,2),(1,2)):
                    d = x[a]-x[b]
                    eligible = np.abs(d) > 1e-6
                    changes.extend((np.sign(d[eligible]) != np.sign((out.image[a]-out.image[b])[eligible])).tolist())
                original_tv = np.mean([np.abs(np.diff(r, axis=a)).mean() for a in range(3)])
                gate_tv = np.mean([np.abs(np.diff(out.reliability, axis=a)).mean() for a in range(3)])
                values.append([np.mean(np.abs(out.residual)), np.mean(changes), gate_tv-original_tv,
                               np.sqrt(np.mean((out.image-x)**2))])
            modes[mode] = dict(zip(('residual_abs_mean', 'sign_violation_rate', 'nonperiodic_tv_change', 'perturbation_rms'), np.mean(values,axis=0).tolist(), strict=True))
        rows.append({'space': space, 'reliability_mean': float(r.mean()),
                     'reliability_margin_correlation': float(np.corrcoef(r.ravel(), margin.ravel())[0,1]),
                     'modes': modes})
    # Matched marginal fields and effective RMS on the same normalized input.
    comparison = {}
    for shared in (True,False):
        values = []
        for seed in range(seeds):
            f = sample_smooth_affine_fields(normalized.shape[1:], rng=np.random.default_rng(seed), shared_across_phases=shared)
            values.append([np.log(f.gain).std(), f.offset.std(), np.abs(np.diff(f.offset,axis=1)).mean(),
                           np.sqrt(np.mean((normalized*f.gain+f.offset-normalized)**2))])
        comparison['A4' if shared else 'A5'] = np.mean(values,axis=0).tolist()
    return {'status':'PRELIMINARY_INTERNAL', 'case':'NACT_53', 'seeds':seeds,
            'normalization_sign_violation_rate':float(np.mean(normalization_changes)),
            'crop_shape':list(raw.shape), 'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'spaces':rows, 'a4_a5_columns':['log_gain_sd','offset_sd','offset_spatial_difference','perturbation_rms'],
            'a4_a5':comparison, 'limitations':['One central crop, not tumor-selected; not a cohort validation.',
             'Independent whole-volume zscore is an audit intervention, not a reconstruction of nnUNet preprocessing.',
             'Reliability-margin association is partly mathematical by construction, not calibration against clinical truth.',
             'Circular shift preserves histogram but may introduce seams; no fair-control approval follows from this audit.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
