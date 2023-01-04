import numpy as np

from fugw import FUGWBarycenter

from .utils import init_distribution


def test_fugw_barycenter():
    np.random.seed(100)
    n_subjects = 4
    n_voxels = 100
    n_features = 10

    # Generate random training data for n subjects
    features_ = []
    geometry_ = []
    weights_ = []

    for _ in range(n_subjects):
        weights, features, geometry, _ = init_distribution(
            n_features, n_voxels
        )
        weights_.append(weights)
        features_.append(features)
        geometry_.append(geometry)

    fugw_barycenter = FUGWBarycenter()
    fugw_barycenter.fit(weights_, features_, geometry_)
