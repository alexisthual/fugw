from itertools import product

import numpy as np
import pytest
import torch

from fugw.scripts import coarse_to_fine
from fugw.mappings import FUGW, FUGWSparse
from fugw.mappings.utils import init_mock_distribution

np.random.seed(0)
torch.manual_seed(0)

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))


return_numpys = [False, True]


@pytest.mark.parametrize(
    "device,return_numpy", product(devices, return_numpys)
)
def test_coarse_to_fine(device, return_numpy):
    n_voxels_source = 105
    n_samples_source = 50
    n_voxels_target = 95
    n_samples_target = 45
    n_features_train = 10
    n_features_test = 5

    _, source_features, _, source_embeddings = init_mock_distribution(
        n_features_train, n_voxels_source, return_numpy=return_numpy
    )
    _, target_features, _, target_embeddings = init_mock_distribution(
        n_features_train, n_voxels_target, return_numpy=return_numpy
    )

    coarse_model = FUGW()
    fine_model = FUGWSparse()

    coarse_model_fit_params = {
        "uot_solver": "mm",
    }

    fine_model_fit_params = {
        "uot_solver": "mm",
    }

    coarse_to_fine.fit(
        coarse_model=coarse_model,
        coarse_model_fit_params=coarse_model_fit_params,
        fine_model=fine_model,
        fine_model_fit_params=fine_model_fit_params,
        source_sample_size=n_samples_source,
        target_sample_size=n_samples_target,
        source_features=source_features,
        target_features=target_features,
        source_geometry_embeddings=source_embeddings,
        target_geometry_embeddings=target_embeddings,
        device=device,
    )

    assert coarse_model.pi.shape == (n_samples_source, n_samples_target)
    assert fine_model.pi.shape == (n_voxels_source, n_voxels_target)

    # Use trained model to transport new features
    # 1. with numpy arrays
    source_features_test = np.random.rand(n_features_test, n_voxels_source)
    target_features_test = np.random.rand(n_features_test, n_voxels_target)
    source_features_on_target = fine_model.transform(source_features_test)
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, np.ndarray)
    target_features_on_source = fine_model.inverse_transform(
        target_features_test
    )
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, np.ndarray)

    source_features_test = np.random.rand(n_voxels_source)
    target_features_test = np.random.rand(n_voxels_target)
    source_features_on_target = fine_model.transform(source_features_test)
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, np.ndarray)
    target_features_on_source = fine_model.inverse_transform(
        target_features_test
    )
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, np.ndarray)

    # 2. with torch tensors
    source_features_test = torch.rand(n_features_test, n_voxels_source)
    target_features_test = torch.rand(n_features_test, n_voxels_target)
    source_features_on_target = fine_model.transform(source_features_test)
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, torch.Tensor)
    target_features_on_source = fine_model.inverse_transform(
        target_features_test
    )
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, torch.Tensor)

    source_features_test = torch.rand(n_voxels_source)
    target_features_test = torch.rand(n_voxels_target)
    source_features_on_target = fine_model.transform(source_features_test)
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, torch.Tensor)
    target_features_on_source = fine_model.inverse_transform(
        target_features_test
    )
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, torch.Tensor)
