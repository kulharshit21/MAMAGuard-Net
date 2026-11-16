from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Mesh:
    vertices_world_mm: np.ndarray
    triangular_faces: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_world_mm, dtype=float)
        faces = np.asarray(self.triangular_faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices_world_mm must have shape (N, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("triangular_faces must have shape (M, 3)")
        if len(vertices) and (faces.min(initial=0) < 0 or faces.max(initial=0) >= len(vertices)):
            raise ValueError("Face indices are outside the vertex array")
        if not np.isfinite(vertices).all():
            raise ValueError("Mesh vertices contain non-finite values")


def mask_to_mesh(mask: np.ndarray, affine: np.ndarray) -> Mesh:
    """Extract an isosurface and transform vertices into NIfTI world millimetres."""
    try:
        from skimage.measure import marching_cubes
    except ImportError as error:  # pragma: no cover - dependency-specific path
        raise RuntimeError("scikit-image is required; install the analysis extra") from error

    binary = np.asarray(mask, dtype=bool)
    transform = np.asarray(affine, dtype=float)
    if binary.ndim != 3:
        raise ValueError("mask must be three-dimensional")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("affine must be a finite 4x4 matrix")
    if not binary.any():
        return Mesh(np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64))

    padded = np.pad(binary.astype(np.uint8), 1)
    vertices, faces, _, _ = marching_cubes(padded, level=0.5)
    voxel_vertices = vertices - 1.0
    homogeneous = np.column_stack((voxel_vertices, np.ones(len(voxel_vertices))))
    world = (transform @ homogeneous.T).T[:, :3]
    return Mesh(world, faces.astype(np.int64, copy=False))


def mesh_surface_area_mm2(mesh: Mesh) -> float:
    if not len(mesh.triangular_faces):
        return 0.0
    triangles = mesh.vertices_world_mm[mesh.triangular_faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def save_mesh(mesh: Mesh, output: Path) -> None:
    """Save a physical-coordinate mesh through PyVista (VTK/PLY/STL/OBJ by suffix)."""
    try:
        import pyvista as pv
    except ImportError as error:  # pragma: no cover - dependency-specific path
        raise RuntimeError("PyVista is required for mesh export; install the analysis extra") from error
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    faces = np.column_stack(
        (np.full(len(mesh.triangular_faces), 3, dtype=np.int64), mesh.triangular_faces)
    ).ravel()
    pv.PolyData(mesh.vertices_world_mm, faces).save(output)

