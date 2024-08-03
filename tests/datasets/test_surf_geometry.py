from itertools import product

import numpy as np
import pytest

from fugw.datasets.surf_geometry import (
    fetch_surf_geometry,
    _check_mesh,
    _check_resolution,
    _fetch_geometry_full_rank,
    _fetch_geometry_low_rank,
)

valid_surf_meshes = [
    "pial_left",
    "pial_right",
    "infl_left",
    "infl_right",
]


def test_check_mesh():
    valid_mesh = valid_surf_meshes[0]
    invalid_mesh = "pial"

    assert _check_mesh(valid_mesh) is None
    with pytest.raises(ValueError):
        _check_mesh(invalid_mesh)


def test_check_resolution():
    valid_resolution = "fsaverage3"
    invalid_resolution = "fsaverage1"

    assert _check_resolution(valid_resolution) is None
    with pytest.raises(ValueError):
        _check_resolution(invalid_resolution)


@pytest.mark.parametrize(
    "mesh,method",
    product(
        valid_surf_meshes,
        ["geodesic", "euclidean"],
    ),
)
def test_fetch_geometry_full_rank(mesh, method):
    resolution = "fsaverage3"
    n_vertices = 642  # fsaverage3 has 642 vertices
    geometry, d_max = _fetch_geometry_full_rank(mesh, resolution, method)

    assert isinstance(d_max, float)
    assert isinstance(geometry, np.ndarray)
    assert geometry.shape == (n_vertices, n_vertices)


@pytest.mark.parametrize(
    "mesh,method,rank,n_landmarks",
    product(
        valid_surf_meshes,
        ["geodesic"],
        [3],
        [100],
    ),
)
def test_fetch_geometry_low_rank(mesh, method, rank, n_landmarks):
    resolution = "fsaverage3"
    n_vertices = 642  # fsaverage3 has 642 vertices
    geometry, d_max = _fetch_geometry_low_rank(
        mesh, resolution, method, rank, n_landmarks, verbose=False
    )

    assert isinstance(d_max, float)
    assert isinstance(geometry, np.ndarray)
    assert geometry.shape == (n_vertices, rank)


@pytest.mark.parametrize(
    "mesh",
    valid_surf_meshes,
)
def test_fetch_surf_geometry(mesh):
    resolution = "fsaverage3"
    n_vertices = 642

    rank = -1
    geometry, d_max = fetch_surf_geometry(mesh, resolution, rank=rank)

    assert isinstance(d_max, float)
    assert isinstance(geometry, np.ndarray)
    assert geometry.shape == (n_vertices, n_vertices)

    rank = 3
    n_landmarks = 100
    geometry, d_max = fetch_surf_geometry(
        mesh, resolution, rank=rank, n_landmarks=n_landmarks, verbose=False
    )

    assert isinstance(d_max, float)
    assert isinstance(geometry, np.ndarray)
    assert geometry.shape == (n_vertices, rank)
