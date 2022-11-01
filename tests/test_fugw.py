import numpy as np
import torch
from fugw import FUGW
from sklearn.metrics import pairwise_distances


def init_distribution(n_features, n_voxels):
    weights = np.ones(n_voxels) / n_voxels
    features = np.random.rand(n_features, n_voxels)
    embedding = np.random.rand(n_voxels, 3)
    geometry = pairwise_distances(embedding)

    return weights, features, geometry


def test_fugw():
    # Generate random training data for source and target
    np.random.seed(100)
    n_voxels = 100
    n_features_train = 10
    n_features_test = 50

    _, source_features_train, source_geometry = init_distribution(
        n_features_train, n_voxels
    )
    _, target_features_train, target_geometry = init_distribution(
        n_features_train, n_voxels
    )

    fugw = FUGW()
    fugw.fit(
        source_features_train,
        target_features_train,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
    )

    # Use trained model to transport new features
    source_features_test = np.random.rand(n_features_test, n_voxels)
    target_features_test = np.random.rand(n_features_test, n_voxels)

    transformed_data = fugw.transform(source_features_test)
    assert transformed_data.shape == source_features_test.shape

    # Compute score
    s = fugw.score(source_features_test, target_features_test)
    assert isinstance(s, int) or isinstance(s, float)


def test_fugw_with_weights():
    # Generate random training data for source and target
    np.random.seed(100)
    n_voxels = 100
    n_features = 10

    source_weights, source_features, source_geometry = init_distribution(
        n_features, n_voxels
    )
    target_weights, target_features, target_geometry = init_distribution(
        n_features, n_voxels
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

    assert fugw.pi.shape == (n_voxels, n_voxels)


def test_fugw_with_torch_tensors():
    # Generate random training data for source and target
    np.random.seed(100)
    n_voxels = 100
    n_features = 10

    source_weights, source_features, source_geometry = init_distribution(
        n_features, n_voxels
    )
    target_weights, target_features, target_geometry = init_distribution(
        n_features, n_voxels
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

    assert fugw.pi.shape == (n_voxels, n_voxels)


# TODO: at some point, it would be nice that this test
# passes so that our model really is a Scikit learn transformer
# def test_fugw_sklearn_transform_api():
#     fugw = FUGW()
#     check_estimator(fugw)
