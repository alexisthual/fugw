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
methods = ["geodesic", "euclidean"]


def test_compute_geodesic_distances_from_volume():
    coordinates = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    field = lmds.compute_distance_field(coordinates)

    distance_field = lmds.compute_geodesic_distances_from_volume(
        field, coordinates, 0
    )

    assert distance_field.shape == (3,)
    assert distance_field.dtype == torch.float64


def test_compute_euclidean_distance():
    coordinates = torch.rand(642, 3)
    distance_field = lmds.compute_euclidean_distance(coordinates, 0)

    cdist = torch.cdist(coordinates, coordinates[0].reshape(1, -1)).flatten()

    assert distance_field.shape == (642,)
    assert distance_field.dtype == torch.float64
    assert torch.equal(distance_field, cdist)


@pytest.mark.parametrize("numpy_inputs", numpy_inputs)
def test_lmds_mesh(numpy_inputs):
    fsaverage3 = datasets.fetch_surf_fsaverage(mesh="fsaverage3")
    coordinates, triangles = surface.load_surf_mesh(fsaverage3.pial_left)

    if numpy_inputs is False:
        coordinates = torch.from_numpy(coordinates)
        triangles = torch.from_numpy(triangles)

    X = lmds.compute_lmds_mesh(
        coordinates,
        triangles,
        n_landmarks=n_landmarks,
        k=k,
        n_jobs=2,
    )

    assert X.shape == (coordinates.shape[0], k)


@pytest.mark.parametrize("method", methods)
def test_lmds_volume(method):
    segmentation = np.ones((20, 20, 20))

    X = lmds.compute_lmds_volume(
        segmentation,
        method=method,
        anisotropy=(1, 1, 1),
        n_landmarks=n_landmarks,
        k=3,
        n_jobs=2,
    )

    assert X.shape == (segmentation.size, 3)
