import numpy as np
import pytest

from mamaguardnet.evaluation.geometry import (
    GeometryMismatchError,
    ImageGeometry,
    assert_same_geometry,
    resample_label_to_geometry,
)


def test_geometry_requires_equal_shape_and_affine():
    reference = ImageGeometry((5, 5, 5), np.eye(4))
    assert_same_geometry(reference, ImageGeometry((5, 5, 5), np.eye(4)))
    with pytest.raises(GeometryMismatchError, match="shape"):
        assert_same_geometry(ImageGeometry((4, 5, 5), np.eye(4)), reference)
    shifted = np.eye(4)
    shifted[0, 3] = 2.0
    with pytest.raises(GeometryMismatchError, match="affine"):
        assert_same_geometry(ImageGeometry((5, 5, 5), shifted), reference)


def test_nearest_neighbour_label_resampling_preserves_discrete_values():
    source_geometry = ImageGeometry((3, 3, 3), np.diag([2.0, 2.0, 2.0, 1.0]))
    target_geometry = ImageGeometry((6, 6, 6), np.eye(4))
    source = np.zeros(source_geometry.shape, dtype=np.uint8)
    source[1, 1, 1] = 3
    result = resample_label_to_geometry(source, source_geometry, target_geometry)
    assert result.shape == target_geometry.shape
    assert set(np.unique(result)) <= {0, 3}
    assert result[2, 2, 2] == 3


def test_geometry_reports_oblique_voxel_volume_and_spacing():
    affine = np.asarray([[2.0, 0.5, 0.0, 10.0], [0.0, 3.0, 0.0, -2.0], [0.0, 0.0, 4.0, 1.0], [0, 0, 0, 1]])
    geometry = ImageGeometry((2, 2, 2), affine)
    assert geometry.voxel_volume_mm3 == pytest.approx(24.0)
    assert geometry.spacing == pytest.approx((2.0, np.sqrt(9.25), 4.0))
