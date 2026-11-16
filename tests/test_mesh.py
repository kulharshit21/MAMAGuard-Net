import numpy as np
import pytest

from mamaguardnet.visualization.mesh import mask_to_mesh, mesh_surface_area_mm2, save_mesh

pytest.importorskip("skimage")


def test_mesh_vertices_are_transformed_to_world_coordinates():
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    affine = np.asarray(
        [[2.0, 0, 0, 10], [0, 3.0, 0, 20], [0, 0, 4.0, 30], [0, 0, 0, 1]]
    )
    mesh = mask_to_mesh(mask, affine)
    assert len(mesh.vertices_world_mm) > 0
    assert mesh.vertices_world_mm[:, 0].min() == pytest.approx(11.0)
    assert mesh.vertices_world_mm[:, 0].max() == pytest.approx(15.0)
    assert mesh.vertices_world_mm[:, 1].min() == pytest.approx(21.5)
    assert mesh.vertices_world_mm[:, 1].max() == pytest.approx(27.5)
    assert mesh_surface_area_mm2(mesh) > 0


def test_empty_mask_returns_empty_mesh():
    mesh = mask_to_mesh(np.zeros((3, 3, 3), dtype=bool), np.eye(4))
    assert mesh.vertices_world_mm.shape == (0, 3)
    assert mesh.triangular_faces.shape == (0, 3)
    assert mesh_surface_area_mm2(mesh) == 0.0


def test_export_reports_missing_pyvista(tmp_path):
    try:
        import pyvista  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="PyVista"):
            save_mesh(mask_to_mesh(np.ones((2, 2, 2), dtype=bool), np.eye(4)), tmp_path / "x.vtk")
    else:
        pytest.skip("PyVista is installed; dependency-error path is not applicable")
