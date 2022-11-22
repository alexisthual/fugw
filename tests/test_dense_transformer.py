import numpy as np
from sklearn.metrics import pairwise_distances
import torch

from fugw import FUGW


np.random.seed(100)
n_voxels_source = 105
n_voxels_target = 95
n_features_train = 10
n_features_test = 5


def init_distribution(n_features, n_voxels):
    weights = np.ones(n_voxels) / n_voxels
    features = np.random.rand(n_features, n_voxels)
    embeddings = np.random.rand(n_voxels, 3)
    geometry = pairwise_distances(embeddings)

    return weights, features, geometry, embeddings


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

    transformed_data = fugw.transform(source_features_test)
    assert transformed_data.shape == target_features_test.shape

    # Compute score
    s = fugw.score(source_features_test, target_features_test)
    assert isinstance(s, int) or isinstance(s, float)


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

    transformed_data = fugw.transform(source_features_test)
    assert transformed_data.shape == target_features_test.shape

    # Compute score
    s = fugw.score(source_features_test, target_features_test)
    assert isinstance(s, int) or isinstance(s, float)


def test_fugw_with_torch_tensors():
    # Generate random training data for source and target
    source_weights, source_features, source_geometry, _ = init_distribution(
        n_features_train, n_voxels_source
    )
    target_weights, target_features, target_geometry, _ = init_distribution(
        n_features_train, n_voxels_target
    )

    source_weights = torch.tensor(source_weights)
    source_features = torch.tensor(source_features)
    source_geometry = torch.tensor(source_geometry)

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

    transformed_data = fugw.transform(source_features_test)
    assert transformed_data.shape == target_features_test.shape

    # Compute score
    s = fugw.score(source_features_test, target_features_test)
    assert isinstance(s, int) or isinstance(s, float)


# TODO: at some point, it would be nice that this test
# passes so that our model really is a Scikit learn transformer
# def test_fugw_sklearn_transform_api():
#     fugw = FUGW()
#     check_estimator(fugw)
