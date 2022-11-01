import numpy as np
from fugw import FUGW
from sklearn.metrics import pairwise_distances


def test_fugw_alignment():
    # Generate random training data for source and target
    np.random.seed(100)
    n_voxels = 100
    n_train = 10
    n_test = 50

    source_features_train = np.random.rand(n_train, n_voxels)
    target_features_train = np.random.rand(n_train, n_voxels)

    source_embedding = np.random.rand(n_voxels, 3)
    target_embedding = np.random.rand(n_voxels, 3)
    source_geometry = pairwise_distances(source_embedding)
    target_geometry = pairwise_distances(target_embedding)

    fugw = FUGW()
    fugw.fit(
        source_features_train,
        target_features_train,
        source_geometry=source_geometry,
        target_geometry=target_geometry,
    )

    # Use trained model to transport new features
    source_features_test = np.random.rand(n_test, n_voxels)
    target_features_test = np.random.rand(n_test, n_voxels)

    transformed_data = fugw.transform(source_features_test)
    assert transformed_data.shape == source_features_test.shape

    # Compute score
    s = fugw.score(source_features_test, target_features_test)
    assert isinstance(s, int) or isinstance(s, float)


# TODO: at some point, it would be nice that this test
# passes so that our model really is a Scikit learn transformer
# def test_fugw_sklearn_transform_api():
#     fugw = FUGW()
#     check_estimator(fugw)
