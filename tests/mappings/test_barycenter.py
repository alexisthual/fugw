import numpy as np
import pytest
import torch

from fugw.mappings import FUGWBarycenter
from fugw.mappings.utils import init_mock_distribution

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))


@pytest.mark.parametrize("device", devices)
def test_fugw_barycenter(device):
    np.random.seed(0)
    n_subjects = 4
    n_voxels = 100
    n_features = 10

    # Generate random training data for n subjects
    features_ = []
    geometry_ = []
    weights_ = []

    for _ in range(n_subjects):
        weights, features, geometry, _ = init_mock_distribution(
            n_features, n_voxels
        )
        weights_.append(weights)
        features_.append(features)
        geometry_.append(geometry)

    fugw_barycenter = FUGWBarycenter()
    fugw_barycenter.fit(weights_, features_, geometry_, device=device)
