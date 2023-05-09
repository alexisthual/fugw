import numpy as np
import pytest
import torch

from fugw.mappings import FUGWBarycenter
from fugw.utils import _init_mock_distribution

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
    features_list = []
    geometry_list = []
    weights_list = []

    for _ in range(n_subjects):
        weights, features, geometry, _ = _init_mock_distribution(
            n_features, n_voxels
        )
        weights_list.append(weights)
        features_list.append(features)
        geometry_list.append(geometry)

    fugw_barycenter = FUGWBarycenter()
    fugw_barycenter.fit(
        weights_list, features_list, geometry_list, device=device
    )
