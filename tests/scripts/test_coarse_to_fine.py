from itertools import product

import numpy as np
import pytest
import torch

from fugw.scripts import coarse_to_fine
from fugw.mappings import FUGW, FUGWSparse
from fugw.utils import init_mock_distribution

np.random.seed(0)
torch.manual_seed(0)

n_voxels_source = 105
n_samples_source = 50
n_voxels_target = 95
n_samples_target = 45
n_features_train = 10
n_features_test = 5

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))

return_numpys = [False, True]


@pytest.mark.parametrize("return_numpy", product(return_numpys))
def test_random_normalizing(return_numpy):
    _, _, _, embeddings = init_mock_distribution(
        n_features_train, n_voxels_source, return_numpy=return_numpy
    )

    embeddings_normalized, d_max = coarse_to_fine.random_normalizing(
        embeddings
    )
    assert isinstance(d_max, float)
    assert embeddings_normalized.shape == embeddings.shape


@pytest.mark.parametrize(
    "device,return_numpy", product(devices, return_numpys)
)
def test_coarse_to_fine(device, return_numpy):
    _, source_features, _, source_embeddings = init_mock_distribution(
        n_features_train, n_voxels_source, return_numpy=return_numpy
    )
    _, target_features, _, target_embeddings = init_mock_distribution(
        n_features_train, n_voxels_target, return_numpy=return_numpy
    )

    coarse_mapping = FUGW()
    fine_mapping = FUGWSparse()

    coarse_mapping_fit_params = {
        "uot_solver": "mm",
    }

    fine_mapping_fit_params = {
        "uot_solver": "mm",
    }

    source_sample, target_sample = coarse_to_fine.fit(
        coarse_mapping=coarse_mapping,
        coarse_mapping_fit_params=coarse_mapping_fit_params,
        fine_mapping=fine_mapping,
        fine_mapping_fit_params=fine_mapping_fit_params,
        source_sample_size=n_samples_source,
        target_sample_size=n_samples_target,
        source_features=source_features,
        target_features=target_features,
        source_geometry_embeddings=source_embeddings,
        target_geometry_embeddings=target_embeddings,
        device=device,
    )

    assert coarse_mapping.pi.shape == (n_samples_source, n_samples_target)
    assert fine_mapping.pi.shape == (n_voxels_source, n_voxels_target)

    assert source_sample.shape == (n_samples_source,)
    assert target_sample.shape == (n_samples_target,)

    # Use trained model to transport new features
    # 1. with numpy arrays
    source_features_test = np.random.rand(n_features_test, n_voxels_source)
    target_features_test = np.random.rand(n_features_test, n_voxels_target)
    source_features_on_target = fine_mapping.transform(source_features_test)
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, np.ndarray)
    target_features_on_source = fine_mapping.inverse_transform(
        target_features_test
    )
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, np.ndarray)

    source_features_test = np.random.rand(n_voxels_source)
    target_features_test = np.random.rand(n_voxels_target)
    source_features_on_target = fine_mapping.transform(source_features_test)
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, np.ndarray)
    target_features_on_source = fine_mapping.inverse_transform(
        target_features_test
    )
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, np.ndarray)

    # 2. with torch tensors
    source_features_test = torch.rand(n_features_test, n_voxels_source)
    target_features_test = torch.rand(n_features_test, n_voxels_target)
    source_features_on_target = fine_mapping.transform(source_features_test)
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, torch.Tensor)
    target_features_on_source = fine_mapping.inverse_transform(
        target_features_test
    )
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, torch.Tensor)

    source_features_test = torch.rand(n_voxels_source)
    target_features_test = torch.rand(n_voxels_target)
    source_features_on_target = fine_mapping.transform(source_features_test)
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, torch.Tensor)
    target_features_on_source = fine_mapping.inverse_transform(
        target_features_test
    )
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, torch.Tensor)
