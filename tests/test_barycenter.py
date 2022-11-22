import numpy as np
from sklearn.metrics import pairwise_distances

from fugw import FUGWBarycenter


def test_fugw_barycenter():
    np.random.seed(100)
    n_subjects = 4
    n_voxels = 100
    n_train = 10

    # Generate random training data for n subjects
    features_ = []
    geometry_ = []
    weights_ = []
    for _ in range(n_subjects):
        features_.append(np.random.rand(n_train, n_voxels))
        weights_.append(np.ones(n_voxels) / n_voxels)
        embeddings = np.random.rand(n_voxels, 3)
        geometry_.append(pairwise_distances(embeddings))

    fugw_barycenter = FUGWBarycenter()
    fugw_barycenter.fit(weights_, features_, geometry_)
