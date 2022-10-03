from fugw.fugw import FUGW
import numpy as np
from sklearn.metrics import pairwise_distances


def test_fugw_alignment():
    # Generate random data for source and target for training
    np.random.seed(100)
    n_voxels = 1000
    n_train = 10
    n_test = 50

    source_data_train = np.random.rand(n_train, n_voxels)
    target_data_train = np.random.rand(n_train, n_voxels)
    source_embedding = np.random.rand(n_voxels, 3)
    target_embedding = np.random.rand(n_voxels, 3)
    source_kernel = pairwise_distances(source_embedding)
    target_kernel = pairwise_distances(target_embedding)

    print(
        f"""Source and target data shapes:
{source_data_train.shape}
{target_data_train.shape}"""
    )

    print(
        f"""Source and target kernel shapes:
{source_kernel.shape}
{target_kernel.shape}"""
    )

    fugw = FUGW()
    fugw.fit(source_data_train, target_data_train, source_kernel, target_kernel)

    # Use trained model to transport data
    source_data_test = np.random.rand(n_test, n_voxels)
    target_data_test = np.random.rand(n_test, n_voxels)

    transformed_data = fugw.transform(source_data_test)
    assert transformed_data.shape == source_data_test.shape

    # Compute score
    s = fugw.score(source_data_test, target_data_test)
    assert isinstance(s, int) or isinstance(s, float)


# TODO: at some point, it would be nice that this test
# passes so that our model really is a Scikit learn transformer
# def test_fugw_sklearn_transform_api():
#     fugw = FUGW()
#     check_estimator(fugw)
