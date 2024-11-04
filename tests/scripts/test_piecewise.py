from itertools import product

import numpy as np
import pytest
import torch

from fugw.scripts import piecewise
from fugw.mappings import FUGWSparse
from fugw.utils import _init_mock_distribution

np.random.seed(0)
torch.manual_seed(0)

n_voxels = 100
n_samples_source = 50
n_samples_target = 45
n_features_train = 10
n_features_test = 5
n_pieces = 10

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))

return_numpys = [False, True]


@pytest.mark.skip_if_no_mkl
def test_one_hot_encoding():
    labels = torch.randint(0, n_pieces, (n_voxels,))
    one_hot = piecewise.one_hot_encoding(labels)
    assert one_hot.shape == (n_voxels, n_pieces)


@pytest.mark.skip_if_no_mkl
@pytest.mark.parametrize(
    "device",
    devices,
)
def test_compute_sparsity_mask(device):
    labels = torch.tensor([0, 1, 1], device=device)
    mask = piecewise.compute_sparsity_mask(labels, device=device)
    assert mask.shape == (3, 3)
    assert mask.is_sparse
    assert torch.allclose(
        mask.to_dense(),
        torch.tensor(
            [[1.0, 0, 0], [0, 1.0, 1.0], [0, 1.0, 1.0]], device=device
        ),
    )

    labels = torch.randint(0, n_pieces, (n_voxels,))
    sparsity_mask = piecewise.compute_sparsity_mask(labels)
    assert sparsity_mask.shape == (n_voxels, n_voxels)


@pytest.mark.skip_if_no_mkl
@pytest.mark.parametrize(
    "device,return_numpy",
    product(devices, return_numpys),
)
def test_piecewise(device, return_numpy):
    source_weights, source_features, source_geometry, source_embeddings = (
        _init_mock_distribution(
            n_features_train, n_voxels, return_numpy=return_numpy
        )
    )
    target_weights, target_features, target_geometry, target_embeddings = (
        _init_mock_distribution(
            n_features_train, n_voxels, return_numpy=return_numpy
        )
    )

    labels = torch.randint(0, n_pieces, (n_voxels,))
    init_plan = piecewise.compute_sparsity_mask(
        labels=labels,
        device=device,
    )

    piecewise_mapping = FUGWSparse()
    piecewise_mapping.fit(
        source_features,
        target_features,
        source_geometry_embedding=source_embeddings,
        target_geometry_embedding=target_embeddings,
        source_weights=source_weights,
        target_weights=target_weights,
        init_plan=init_plan,
        device=device,
        verbose=True,
    )

    assert piecewise_mapping.pi.shape == (n_voxels, n_voxels)

    # Use trained model to transport new features
    # 1. with numpy arrays
    source_features_test = np.random.rand(n_features_test, n_voxels)
    target_features_test = np.random.rand(n_features_test, n_voxels)
    source_features_on_target = piecewise_mapping.transform(
        source_features_test
    )
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, np.ndarray)
    target_features_on_source = piecewise_mapping.inverse_transform(
        target_features_test
    )
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, np.ndarray)

    source_features_test = np.random.rand(n_voxels)
    target_features_test = np.random.rand(n_voxels)
    source_features_on_target = piecewise_mapping.transform(
        source_features_test
    )
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, np.ndarray)
    target_features_on_source = piecewise_mapping.inverse_transform(
        target_features_test
    )
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, np.ndarray)

    # 2. with torch tensors
    source_features_test = torch.rand(n_features_test, n_voxels)
    target_features_test = torch.rand(n_features_test, n_voxels)
    source_features_on_target = piecewise_mapping.transform(
        source_features_test
    )
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, torch.Tensor)
    target_features_on_source = piecewise_mapping.inverse_transform(
        target_features_test
    )
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, torch.Tensor)

    source_features_test = torch.rand(n_voxels)
    target_features_test = torch.rand(n_voxels)
    source_features_on_target = piecewise_mapping.transform(
        source_features_test
    )
    assert source_features_on_target.shape == target_features_test.shape
    assert isinstance(source_features_on_target, torch.Tensor)
    target_features_on_source = piecewise_mapping.inverse_transform(
        target_features_test
    )
    assert target_features_on_source.shape == source_features_test.shape
    assert isinstance(target_features_on_source, torch.Tensor)
