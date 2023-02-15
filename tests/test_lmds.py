import meshzoo
import numpy as np
import pytest
import torch

from fugw.scripts import lmds

np.random.seed(0)
torch.manual_seed(0)

n_landmarks = 10
k = 3

numpy_inputs = [True, False]


@pytest.mark.parametrize("numpy_inputs", numpy_inputs)
def test_lmds(numpy_inputs):
    coordinates, triangles = meshzoo.octa_sphere(6)

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
