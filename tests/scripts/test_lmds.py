import numpy as np
import pytest
import torch

from fugw.scripts import lmds
from nilearn import datasets, surface

np.random.seed(0)
torch.manual_seed(0)

n_landmarks = 10
k = 3

numpy_inputs = [True, False]


def test_compute_geodesic_distances():
    fsaverage3 = datasets.fetch_surf_fsaverage(mesh="fsaverage3")
    coordinates, triangles = surface.load_surf_mesh(fsaverage3.pial_left)

    distances = lmds.compute_geodesic_distances(coordinates, triangles, 0)

    assert distances.shape == (642,)
    assert distances.dtype == torch.float64


def test_compute_geodesic_distances_edges():
    fsaverage3 = datasets.fetch_surf_fsaverage(mesh="fsaverage3")
    coordinates, triangles = surface.load_surf_mesh(fsaverage3.pial_left)
    adjacency = lmds.adjacency_matrix_from_triangles(
        coordinates.shape[0], triangles
    )
    print(adjacency.shape)
    print(coordinates.shape)

    distances = lmds.compute_geodesic_distances_edges(
        coordinates, adjacency, 0
    )

    assert distances.shape == (642,)
    assert distances.dtype == torch.float64


@pytest.mark.parametrize("numpy_inputs", numpy_inputs)
def test_lmds(numpy_inputs):
    fsaverage3 = datasets.fetch_surf_fsaverage(mesh="fsaverage3")
    coordinates, triangles = surface.load_surf_mesh(fsaverage3.pial_left)

    if numpy_inputs is False:
        coordinates = torch.from_numpy(coordinates)
        triangles = torch.from_numpy(triangles)

    X = lmds.compute_lmds(
        coordinates,
        triangles,
        n_landmarks=n_landmarks,
        k=k,
        n_jobs=2,
    )

    assert X.shape == (coordinates.shape[0], k)
