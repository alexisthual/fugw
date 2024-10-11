from itertools import product

import numpy as np
import pytest
import torch

from fugw.mappings import FUGWBarycenter
from fugw.utils import _init_mock_distribution

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))

callbacks = [None, lambda x: x["plans"]]
alphas = [0.0, 0.5, 1.0]


@pytest.mark.parametrize(
    "device, callback",
    product(devices, callbacks),
)
def test_fugw_barycenter(device, callback):
    """Tests the FUGW barycenter fitting on toy data."""
    np.random.seed(0)
    n_subjects = 4
    n_voxels = 100
    n_features = 10
    nits_barycenter = 3

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

    # Fit the barycenter
    (
        barycenter_weights,
        barycenter_features,
        barycenter_geometry,
        plans,
        _,
        losses_each_bar_step,
    ) = fugw_barycenter.fit(
        weights_list,
        features_list,
        geometry_list,
        solver_params={"nits_bcd": 2, "nits_uot": 5},
        nits_barycenter=nits_barycenter,
        device=device,
        callback_barycenter=callback,
        init_barycenter_geometry=geometry_list[0],
    )

    assert isinstance(barycenter_weights, torch.Tensor)
    assert barycenter_weights.shape == (n_voxels,)
    assert isinstance(barycenter_features, torch.Tensor)
    assert barycenter_features.shape == (n_features, n_voxels)
    assert isinstance(barycenter_geometry, torch.Tensor)
    assert barycenter_geometry.shape == (n_voxels, n_voxels)
    assert len(plans) == n_subjects
    assert len(losses_each_bar_step) == nits_barycenter


@pytest.mark.parametrize(
    "alpha",
    alphas,
)
def test_identity_case(alpha):
    """Test the case where all subjects are the same."""
    torch.manual_seed(0)
    n_subjects = 3
    n_features = 10
    n_voxels = 100
    nits_barycenter = 2

    geometry = _init_mock_distribution(n_features, n_voxels)[2]
    features = torch.rand(n_features, n_voxels)

    geometry_list = [geometry for _ in range(n_subjects)]
    features_list = [features for _ in range(n_subjects)]
    weights_list = [torch.ones(n_voxels) / n_voxels for _ in range(n_subjects)]

    fugw_barycenter = FUGWBarycenter(alpha=alpha, eps=1e-6, rho=float("inf"))
    (
        barycenter_weights,
        barycenter_features,
        barycenter_geometry,
        plans,
        _,
        _,
    ) = fugw_barycenter.fit(
        weights_list,
        features_list,
        geometry_list,
        solver_params={"nits_bcd": 5, "nits_uot": 100},
        nits_barycenter=nits_barycenter,
        device=torch.device("cpu"),
        init_barycenter_geometry=geometry_list[0],
        init_barycenter_features=features_list[0],
    )

    # Check that the barycenter is the same as the input
    assert torch.allclose(barycenter_weights, torch.ones(n_voxels) / n_voxels)
    assert torch.allclose(barycenter_geometry, geometry_list[0])

    # In the case alpha=1.0, the features can be permuted
    # since the GW distance is invariant under isometries
    if alpha != 1.0:
        assert torch.allclose(barycenter_features, features)
        # Check that all the plans are the identity matrix divided
        # by the number of voxels
        for plan in plans:
            assert torch.allclose(plan, torch.eye(n_voxels) / n_voxels)
