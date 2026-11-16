import numpy as np

from mamaguardnet.temporal_analysis import (
    audit_kinetics_domains,
    sample_a4_smooth_affine,
    zscore_per_phase,
)


def test_audit_separates_normalization_from_augmented_domain():
    phases = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [10.0, 11.0, 12.0, 100.0],
            [20.0, 21.0, 22.0, 101.0],
        ],
        dtype=np.float32,
    )
    normalized = zscore_per_phase(phases)
    normalized_3d = np.broadcast_to(normalized[:, :, None, None], (3, 4, 4, 4)).copy()
    original_3d = np.broadcast_to(phases[:, :, None, None], (3, 4, 4, 4)).copy()
    augmentation = sample_a4_smooth_affine(
        normalized_3d,
        rng=np.random.default_rng(2),
        grid=2,
    )
    report = audit_kinetics_domains(original_3d, normalized_3d, augmentation.image)
    assert report["original_to_normalized"]["sign_violations"] > 0
    assert report["normalized_to_augmented"]["valid"]
    assert report["normalized_to_augmented"]["sign_violations"] == 0
    assert report["normalized_to_augmented"]["ordering_violations"] == 0
    assert report["interpretation"]["proven_domain"] == "normalized model-input domain"
    assert np.all(augmentation.gain > 0)
    assert augmentation.gain.std() > 0


def test_masked_zscore_and_audit_validate_shapes():
    phases = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    mask = np.asarray([[True, True, False, False], [True, True, False, False]])
    normalized = zscore_per_phase(phases, mask)
    augmented = 1.1 * normalized + 0.2
    report = audit_kinetics_domains(phases, normalized, augmented, mask=mask)
    assert report["selected_voxels"] == 4
    assert report["normalized_to_augmented"]["valid"]
