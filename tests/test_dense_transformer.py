import numpy as np
import torch

from fugw import FUGW
from fugw.utils import init_mock_distribution

np.random.seed(0)
torch.manual_seed(0)

n_voxels_source = 105
n_voxels_target = 95
n_features_train = 10
n_features_test = 5


def test_fugw():
    # Generate random training data for source and target
    _, source_features_train, source_geometry, _ = init_distribution(
        n_features_train, n_voxels_source
    )
    _, target_features_train, target_geometry, _ = init_distribution(
        n_features_train, n_voxels_target
    )

    fugw = FUGW()
    fugw.fit(
        source_features_train,
        target_features_train,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
    )

    # Use trained model to transport new features
    source_features_test = np.random.rand(n_features_test, n_voxels_source)
    target_features_test = np.random.rand(n_features_test, n_voxels_target)
    source_features_on_target = fugw.transform(source_features_test)
    assert source_features_on_target.shape == target_features_test.shape
    target_features_on_source = fugw.inverse_transform(target_features_test)
    assert target_features_on_source.shape == source_features_test.shape


def test_fugw_with_weights():
    # Generate random training data for source and target
    source_weights, source_features, source_geometry, _ = init_distribution(
        n_features_train, n_voxels_source
    )
    target_weights, target_features, target_geometry, _ = init_distribution(
        n_features_train, n_voxels_target
    )

    fugw = FUGW()
    fugw.fit(
        source_features,
        target_features,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
        source_weights=source_weights,
        target_weights=target_weights,
    )

    assert fugw.pi.shape == (n_voxels_source, n_voxels_target)

    # Use trained model to transport new features
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


def test_fugw_with_torch_tensors():
    # Generate random training data for source and target
    source_weights, source_features, source_geometry, _ = init_distribution(
        n_features_train, n_voxels_source
    )
    target_weights, target_features, target_geometry, _ = init_distribution(
        n_features_train, n_voxels_target
    )

    fugw = FUGW()
    fugw.fit(
        source_features,
        target_features,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
        source_weights=source_weights,
        target_weights=target_weights,
    )

    assert fugw.pi.shape == (n_voxels_source, n_voxels_target)

    # Use trained model to transport new features
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
