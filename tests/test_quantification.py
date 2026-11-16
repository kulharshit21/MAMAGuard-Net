import numpy as np
import pytest

from mamaguardnet.evaluation.geometry import ImageGeometry
from mamaguardnet.quantification.measurements import quantify_mask


def test_anisotropic_cuboid_measurements_are_in_physical_units():
    geometry = ImageGeometry((5, 5, 5), np.diag([2.0, 3.0, 4.0, 1.0]))
    mask = np.zeros(geometry.shape, dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    result = quantify_mask(mask, geometry)
    assert result["voxel_count"] == 8
    assert result["volume_mm3"] == pytest.approx(192.0)
    assert result["volume_cm3"] == pytest.approx(0.192)
    assert result["centroid_mm"] == pytest.approx([3.0, 4.5, 6.0])
    assert result["axis_aligned_extents_mm"] == pytest.approx([4.0, 6.0, 8.0])
    assert sorted(result["principal_dimensions_mm"]) == pytest.approx([4.0, 6.0, 8.0])
    assert result["surface_area_mm2"] == pytest.approx(208.0)
    assert result["component_count"] == 1
    assert result["largest_component_fraction"] == 1.0


def test_components_translation_and_empty_mask_are_handled():
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = [10.0, 20.0, 30.0]
    geometry = ImageGeometry((4, 4, 4), affine)
    mask = np.zeros(geometry.shape, dtype=bool)
    mask[0, 0, 0] = True
    mask[3, 3, 3] = True
    result = quantify_mask(mask, geometry)
    assert result["centroid_mm"] == pytest.approx([11.5, 21.5, 31.5])
    assert result["component_count"] == 2
    assert result["largest_component_fraction"] == 0.5

    empty = quantify_mask(np.zeros(geometry.shape), geometry)
    assert empty["volume_mm3"] == 0.0
    assert empty["centroid_mm"] is None
    assert empty["component_count"] == 0
