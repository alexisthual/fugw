import torch


def init_distribution(n_features, n_voxels, should_normalize=True):
    weights = torch.ones(n_voxels) / n_voxels
    features = torch.rand(n_features, n_voxels)
    embeddings = torch.rand(n_voxels, 3)
    geometry = torch.cdist(embeddings, embeddings)

    # Normalize outputs if need be
    features_normalized = features / torch.linalg.norm(
        features, dim=1
    ).reshape(-1, 1)
    geometry_normalized = geometry / geometry.max()
    embeddings_normalized = embeddings / geometry.max()

    if should_normalize:
        return (
            weights,
            features_normalized,
            geometry_normalized,
            embeddings_normalized,
        )
    else:
        return (
            weights.numpy(),
            features.numpy(),
            geometry.numpy(),
            embeddings.numpy(),
        )
