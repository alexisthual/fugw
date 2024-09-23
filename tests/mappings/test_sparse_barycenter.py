from itertools import product

import numpy as np
import pytest
import torch

from fugw.mappings import FUGWSparseBarycenter
from fugw.utils import _init_mock_distribution

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))

callbacks = [None, lambda x: x["plans"]]


@pytest.mark.skip_if_no_mkl
@pytest.mark.parametrize(
    "device, callback",
    product(devices, callbacks),
)
def test_fugw_sparse_barycenter(device, callback):
    """Tests the FUGW sparse barycenter fitting on toy data."""
    np.random.seed(0)
    n_subjects = 4
    n_voxels = 100
    n_features = 10
    nits_barycenter = 3

    mesh_sample = np.random.randint(0, n_voxels, size=10)

    # Generate random training data for n subjects
    features_list = []
    weights_list = []

    for _ in range(n_subjects):
        weights, features, geometry_embedding, _ = _init_mock_distribution(
            n_features, n_voxels, should_normalize=True
        )
        weights_list.append(weights)
        features_list.append(features)

    geometry_embedding_normalized = (
        geometry_embedding / geometry_embedding.norm()
    )
    fugw_sparse_barycenter = FUGWSparseBarycenter()

    # Fit the barycenter
    (
        barycenter_weights,
        barycenter_features,
        plans,
        losses_each_bar_step,
    ) = fugw_sparse_barycenter.fit(
        weights_list,
        features_list,
        geometry_embedding_normalized,
        mesh_sample=mesh_sample,
        coarse_mapping_solver_params={"nits_bcd": 2, "nits_uot": 5},
        fine_mapping_solver_params={"nits_bcd": 2, "nits_uot": 5},
        nits_barycenter=nits_barycenter,
        device=device,
        callback_barycenter=callback,
    )

    assert isinstance(barycenter_weights, torch.Tensor)
    assert barycenter_weights.shape == (n_voxels,)
    assert isinstance(barycenter_features, torch.Tensor)
    assert not torch.isnan(barycenter_features).any()
    assert barycenter_features.shape == (n_features, n_voxels)
    assert len(plans) == n_subjects
    assert len(losses_each_bar_step) == nits_barycenter
