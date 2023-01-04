import numpy as np
from sklearn.metrics import pairwise_distances


def init_distribution(n_features, n_voxels, should_normalize=True):
    weights = np.ones(n_voxels) / n_voxels
    features = np.random.rand(n_features, n_voxels)
    embeddings = np.random.rand(n_voxels, 3)
    geometry = pairwise_distances(embeddings)

    # Normalize outputs if need be
    features_normalized = features / np.linalg.norm(features, axis=1).reshape(
        -1, 1
    )
    geometry_normalized = geometry / np.max(geometry)
    embeddings_normalized = embeddings / np.max(geometry)

    if should_normalize:
        return (
            weights,
            features_normalized,
            geometry_normalized,
            embeddings_normalized,
        )
    else:
        return weights, features, geometry, embeddings
