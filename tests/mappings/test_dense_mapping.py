from itertools import product

import numpy as np
import pytest
import torch

from fugw.mappings import FUGW
from fugw.utils import _init_mock_distribution

np.random.seed(0)
torch.manual_seed(0)

n_voxels_source = 105
n_voxels_target = 95
n_features_train = 10
n_features_test = 5

return_numpys = [True, False]

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))

solvers = ["sinkhorn", "mm", "ibpp"]

callbacks = [None, lambda x: x["gamma"]]


@pytest.mark.parametrize(
    "device,return_numpy,solver,callback", product(devices, return_numpys, solvers, callbacks)
)
def test_dense_mapping(device, return_numpy, solver, callback):
    # Generate random training data for source and target
    _, source_features_train, source_geometry, _ = _init_mock_distribution(
        n_features_train, n_voxels_source, return_numpy=return_numpy
    )
    _, target_features_train, target_geometry, _ = _init_mock_distribution(
        n_features_train, n_voxels_target, return_numpy=return_numpy
    )

    fugw = FUGW()
    fugw.fit(
        source_features=source_features_train,
        target_features=target_features_train,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
        solver=solver,
        solver_params={
            "nits_bcd": 3,
            "ibpp_eps_base": 1e8,
        },
        callback_bcd=callback,
        device=device,
    )

    # Use trained model to transport new features
    # 1. with numpy arrays
    source_features_test = np.random.rand(n_features_test, n_voxels_source)
    target_features_test = np.random.rand(n_features_test, n_voxels_target)
    source_features_on_target = fugw.transform(source_features_test)
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, np.ndarray)
    target_features_on_source = fugw.inverse_transform(target_features_test)
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, np.ndarray)

    source_features_test = np.random.rand(n_voxels_source)
    target_features_test = np.random.rand(n_voxels_target)
    source_features_on_target = fugw.transform(source_features_test)
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, np.ndarray)
    target_features_on_source = fugw.inverse_transform(target_features_test)
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, np.ndarray)

    # 2. with torch tensors
    source_features_test = torch.rand(n_features_test, n_voxels_source)
    target_features_test = torch.rand(n_features_test, n_voxels_target)
    source_features_on_target = fugw.transform(source_features_test)
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, torch.Tensor)
    target_features_on_source = fugw.inverse_transform(target_features_test)
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, torch.Tensor)

    source_features_test = torch.rand(n_voxels_source)
    target_features_test = torch.rand(n_voxels_target)
    source_features_on_target = fugw.transform(source_features_test)
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, torch.Tensor)
    target_features_on_source = fugw.inverse_transform(target_features_test)
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, torch.Tensor)
